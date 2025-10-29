from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
import torch
import os
import json
import random
import csv
import re
from datetime import datetime
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import gc

# Configuration
local_path = "models/InternVL3_5-14B-HF"
model_name = "OpenGVLab/InternVL3_5-14B-HF"
logged_model_name = re.sub(r'^[\\/]*models[\\/]+', '', local_path)
videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
json_file_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/WLASL_v0.3.json"
output_dir = "output/"

# Load model, tokenizer and processor
model = AutoModelForImageTextToText.from_pretrained(
    local_path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_video_frames(video_path, num_frames=32):
    """Load and sample video frames uniformly."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    frames = [Image.fromarray(vr[int(idx)].asnumpy()) for idx in indices]
    return frames


def predict_gloss(video_path, fps=25):
    """Predict gloss from an ASL video using InternVL model."""
    
    frames = load_video_frames(video_path, num_frames=32)
    
    system_content = (
        "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed."
        "Provide the gloss of the sign language video, output only the gloss, no other text."
        "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions."
        "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    )
    question = "What is the ASL gloss for this sign? Output only the single English word or phrase."
    
    # Build message with image placeholders
    image_content = [{"type": "image"} for _ in frames]
    text_content = {"type": "text", "text": f"{system_content}\n{question}"}
    messages = [{"role": "user", "content": image_content + [text_content]}]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(text=prompt, images=frames, return_tensors="pt").to("cuda")
    
    # Generate and decode
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    result = output_text[0] if output_text else ""
    
    # Aggressive memory cleanup
    del frames, inputs, generated_ids, generated_ids_trimmed, output_text
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
    return result

def process_dataset(json_file_path, videos_path, output_dir):
    """Process a dataset of videos and save results to CSV and log files."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file_path = os.path.join(output_dir, f"wlasl_predictions_{logged_model_name}_{timestamp}.csv")
    log_file_path = os.path.join(output_dir, f"wlasl_processing_{logged_model_name}_{timestamp}.log")
    
    # Load JSON data
    print("Loading WLASL JSON data...")
    with open(json_file_path, 'r') as f:
        wlasl_data = json.load(f)
    
    print(f"Loaded {len(wlasl_data)} glosses from JSON")
    
    # Collect all test videos with their metadata
    test_videos = []
    
    for gloss_idx, gloss_entry in enumerate(wlasl_data):
        gloss_name = gloss_entry['gloss']
        gloss_folder = str(gloss_idx)  # Folder named by index
        gloss_folder_path = os.path.join(videos_path, gloss_folder)
        
        # Check if this gloss has a test folder
        if not os.path.exists(gloss_folder_path):
            continue
            
        # Get all instances marked as test split
        for instance in gloss_entry['instances']:
            if instance.get('split') == 'test':
                video_id = instance['video_id']
                video_filename = f"{video_id}.mp4"
                video_path = os.path.join(gloss_folder_path, video_filename)
                
                # Check if video file exists
                if os.path.exists(video_path):
                    test_videos.append({
                        'video_path': video_path,
                        'ground_truth_gloss': gloss_name,
                        'gloss_id': gloss_idx,
                        'video_id': video_id,
                        'fps': instance.get('fps', 25)  # Use fps from instance, default to 25
                    })
    
    print(f"Found {len(test_videos)} test videos to process")
    
    if len(test_videos) == 0:
        print("No test videos found! Check the videos_path and JSON structure.")
        return
    
    # Set random seed for reproducible order
    random.seed(42)
    random.shuffle(test_videos)
    
    # Initialize CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ground_truth_gloss', 'gloss_id', 'video_id', 'predicted_gloss', 'video_fps', 'processing_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each video
        processed_count = 0
        errors_count = 0
        
        for i, video_info in enumerate(test_videos):
            start_time = datetime.now()
            
            try:
                print(f"Processing video {i+1}/{len(test_videos)}: {video_info['video_id']} - {video_info['ground_truth_gloss']}")
                
                # Use FPS from video metadata (WLASL videos are typically 25 FPS)
                video_fps = video_info['fps']
                
                # Predict gloss using InternVL
                predicted_gloss = predict_gloss(video_info['video_path'], fps=video_fps)
                
                # Clean up the predicted gloss (remove extra whitespace, newlines, etc.)
                predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                # Also remove multiple consecutive spaces
                predicted_gloss = ' '.join(predicted_gloss.split())
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Write to CSV
                writer.writerow({
                    'ground_truth_gloss': video_info['ground_truth_gloss'],
                    'gloss_id': video_info['gloss_id'],
                    'video_id': video_info['video_id'],
                    'predicted_gloss': predicted_gloss,
                    'video_fps': video_fps,
                    'processing_time': processing_time
                })
                
                # Write to log file
                log_message = f"{datetime.now().isoformat()} - MODEL: {logged_model_name} - SUCCESS - Video: {video_info['video_id']} - GT: {video_info['ground_truth_gloss']} - Predicted: {predicted_gloss} - Time: {processing_time:.2f}s\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(log_message)
                
                processed_count += 1
                print(f"  → Predicted: '{predicted_gloss}' (took {processing_time:.2f}s)")
                
            except Exception as e:
                error_message = f"{datetime.now().isoformat()} - MODEL: {logged_model_name} - ERROR - Video: {video_info['video_id']} - GT: {video_info['ground_truth_gloss']} - Error: {str(e)}\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(error_message)
                
                print(f"  → ERROR: {str(e)}")
                errors_count += 1
                
                # Write error row to CSV
                writer.writerow({
                    'ground_truth_gloss': video_info['ground_truth_gloss'],
                    'gloss_id': video_info['gloss_id'],
                    'video_id': video_info['video_id'],
                    'predicted_gloss': f"ERROR: {str(e)}",
                    'video_fps': video_fps,
                    'processing_time': 0
                })
            
            # Flush files periodically
            if (i + 1) % 10 == 0:
                csvfile.flush()
                print(f"Progress: {i+1}/{len(test_videos)} videos processed")
    
    # Final summary
    summary_message = f"""
=== PROCESSING COMPLETE ===
Model: {logged_model_name}
Total videos: {len(test_videos)}
Successfully processed: {processed_count}
Errors: {errors_count}
CSV output: {csv_file_path}
Log output: {log_file_path}
    """
    
    print(summary_message)
    
    # Write summary to log
    with open(log_file_path, 'a', encoding='utf-8') as logfile:
        logfile.write(f"\n{datetime.now().isoformat()} - SUMMARY:\n{summary_message}\n")


if __name__ == "__main__":
    print(f"Start date and time = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    process_dataset(json_file_path, videos_path, output_dir)
    print(f"End date and time = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

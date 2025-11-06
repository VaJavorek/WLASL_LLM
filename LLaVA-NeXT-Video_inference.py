from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import torch
import os
import json
import random
import csv
import re
from datetime import datetime
import av
import numpy as np

# Configuration
# Options: "llava-hf/LLaVA-NeXT-Video-7B-hf", "llava-hf/LLaVA-NeXT-Video-34B-hf", "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"
local_path = "models/LLaVA-NeXT-Video-7B-hf"  # Optional: use local path if model is downloaded
use_local = True  # Set to True if using local model path

logged_model_name = model_name.split('/')[-1] if not use_local else re.sub(r'^[\\/]*models[\\/]+', '', local_path)
videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
json_file_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/WLASL_v0.3.json"
output_dir = "output/"

# Load model and processor
print(f"Loading {model_name} model...")
model_path = local_path if use_local else model_name

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

processor = LlavaNextVideoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

print("Model loaded successfully!")


def read_video_pyav(video_path, num_frames=8):
    """
    Read video frames using PyAV library.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract evenly from the video
    
    Returns:
        List of numpy arrays representing frames
    """
    container = av.open(video_path)
    
    # Get total number of frames
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        # If frames count is not available, count manually
        total_frames = sum(1 for _ in container.decode(video=0))
        container.close()
        container = av.open(video_path)
    
    # Calculate indices for evenly spaced frames
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    video_stream = container.streams.video[0]
    
    for i, frame in enumerate(container.decode(video_stream)):
        if i in indices:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        
        if len(frames) >= num_frames:
            break
    
    container.close()
    
    return frames


def predict_gloss(video_path, num_frames=8):
    """Predict gloss from an ASL video using LLaVA-NeXT-Video model."""
    
    # Read video frames
    video_frames = read_video_pyav(video_path, num_frames=num_frames)
    
    if len(video_frames) == 0:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    
    # Prepare the conversation with system-like instruction and user query
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "You are an expert ASL (American Sign Language) gloss annotator. Watch this ASL video carefully and identify the single English word or phrase (gloss) that represents the sign being performed. Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc. What is the ASL gloss for this sign? Output only the single English word or phrase, no other text or description."},
            ],
        },
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        text=prompt,
        videos=video_frames,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,  # Use greedy decoding for consistency
        )
    
    # Decode the generated text
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


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
        fieldnames = ['ground_truth_gloss', 'gloss_id', 'video_id', 'predicted_gloss', 'num_frames', 'processing_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each video
        processed_count = 0
        errors_count = 0
        
        for i, video_info in enumerate(test_videos):
            start_time = datetime.now()
            
            try:
                print(f"Processing video {i+1}/{len(test_videos)}: {video_info['video_id']} - {video_info['ground_truth_gloss']}")
                
                # Use 8 frames for video processing
                num_frames = 8
                
                # Predict gloss using LLaVA-NeXT-Video
                predicted_gloss = predict_gloss(video_info['video_path'], num_frames=num_frames)
                
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
                    'num_frames': num_frames,
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
                    'num_frames': 0,
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


from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

path = "models/Llama-3.1-Nemotron-Nano-VL-8B-V1"
model = AutoModel.from_pretrained(path, trust_remote_code=True, device_map="cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(path)
image_processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True, device="cuda")

import decord
# sys.path.append("frame_selection")
from frame_selection.data_handlers import get_adaptive_frames, get_equally_distributed_frames
from PIL import Image
import numpy as np
import torch.nn.functional as F

def predict_gloss(video_path, from_frames=False, cut_edges=False, fps=None, required_frames=64, max_w=224, max_h=224, use_img_input=False):
    """Predict gloss from an ASL video using Qwen model."""

    system_content = (
        "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed."
        "Provide the gloss of the sign language video, output only the gloss, no other text."
        "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions."
        "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    )
    
    prompt = "What is the ASL gloss for this sign? Output only the single English word or phrase."
    
    # if video is preprocessed and put into model as array(tensor)
    if from_frames:
        
        vr = decord.VideoReader(video_path)
        if cut_edges is True:
            frames = get_adaptive_frames(vr, required_frames=required_frames)
            if fps is None:
                fps = vr.get_avg_fps()
        else:
            frames = [Image.fromarray(frame.asnumpy()) for frame in vr]  # list of PIL images
            frames = np.stack(frames)
            
            if fps is not None:
                frames = get_equally_distributed_frames(frames, fps, required_frames=required_frames)
            else:
                fps = vr.get_avg_fps()
        if frames.shape[1] > max_h or frames.shape[2] > max_w:
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            frames = F.interpolate(frames, size=(max_h, max_w), mode='bilinear', align_corners=False)
            
        
        # if input is image
        if use_img_input:
            frames = frames.permute(0, 2, 3, 1).cpu().numpy()
            pil_images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
            image_features = image_processor(pil_images)
            generation_config = dict(max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.eos_token_id)
            response = model.chat(tokenizer=tokenizer, question=prompt, system=system_content, generation_config=generation_config, **image_features)
            return output_text[0].split()[0].replace("*", "") if output_text else ""

def process_dataset(json_file_path, videos_path, output_dir):
    """Process a dataset of videos and save results to CSV and log files."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file_path = os.path.join(output_dir, f"wlasl_predictions_{timestamp}.csv")
    log_file_path = os.path.join(output_dir, f"wlasl_processing_{timestamp}.log")
    
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
                
                # Use FPS from the video metadata, but cap it at reasonable value for processing
                # video_fps = min(video_info['fps'], 4)
                video_fps = 4
                # Predict gloss using Qwen
                predicted_gloss = predict_gloss(video_info['video_path'], from_frames=False, fps=video_fps)
                
                # Clean up the predicted gloss (remove extra whitespace, newlines, etc.)
                predicted_gloss = predicted_gloss.strip().replace('\n', '').replace('\r', '')
                predicted_gloss = predicted_gloss.replace('.', '').replace('\"', '').replace(")", "")
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
                log_message = f"{datetime.now().isoformat()} - SUCCESS - Video: {video_info['video_id']} - GT: {video_info['ground_truth_gloss']} - Predicted: {predicted_gloss} - Time: {processing_time:.2f}s\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(log_message)
                
                processed_count += 1
                print(f"  → Predicted: '{predicted_gloss}' (took {processing_time:.2f}s)")
                
            except Exception as e:
                error_message = f"{datetime.now().isoformat()} - ERROR - Video: {video_info['video_id']} - GT: {video_info['ground_truth_gloss']} - Error: {str(e)}\n"
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



process_dataset(json_file_path, videos_path, output_dir)
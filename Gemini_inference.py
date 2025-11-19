import os
import json
import random
import csv
import argparse
from datetime import datetime
import cv2
from google import genai
from google.genai import types

# Configuration
videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
json_file_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/WLASL_v0.3.json"
output_dir = "output/"

def extract_frames_from_video(video_path, num_frames=8):
    """Extract evenly spaced frames from a video for processing."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {idx} from {video_path}")
    
    cap.release()
    
    if not frames:
        raise ValueError(f"Could not extract any frames from: {video_path}")
    
    return frames

def predict_gloss(client, video_path, model_name="gemini-2.0-flash", num_frames=8, max_output_tokens=4096):
    """Predict gloss from an ASL video using Google Gemini API."""
    
    system_instruction = (
        "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed. "
        "Provide the gloss of the sign language video, output only the gloss, no other text. "
        "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions. "
        "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    )
    
    # Extract frames from video
    frames = extract_frames_from_video(video_path, num_frames=num_frames)
    
    # Prepare content parts
    contents = [
        "What is the ASL gloss for this sign shown in these sequential frames? Output only the single English word or phrase."
    ]
    
    # Add each frame as image bytes
    for frame in frames:
        # Encode frame to JPEG bytes
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
        image_bytes = buffer.tobytes()
        
        # Create a Part object from bytes
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    
    # Prepare configuration
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=0.0, # Low temperature for deterministic output
    )
    
    try:
        # Call Gemini API
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        
        # Extract response text
        if response.text:
            return response.text.strip()
        else:
            return "EMPTY_RESPONSE"
            
    except Exception as e:
        # Check for safety refusals or other API errors
        if "refusal" in str(e).lower() or "safety" in str(e).lower():
             return f"REFUSED: {str(e)}"
        raise e

def process_dataset(json_file_path, videos_path, output_dir, api_key, model_name="gemini-2.0-flash", num_frames=8, max_output_tokens=4096):
    """Process a dataset of videos and save results to CSV and log files."""
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    csv_file_path = os.path.join(output_dir, f"wlasl_predictions_{safe_model_name}_{timestamp}.csv")
    log_file_path = os.path.join(output_dir, f"wlasl_processing_{safe_model_name}_{timestamp}.log")
    
    # Load JSON data
    print("Loading WLASL JSON data...")
    with open(json_file_path, 'r') as f:
        wlasl_data = json.load(f)
    
    print(f"Loaded {len(wlasl_data)} glosses from JSON")
    
    # Collect all test videos with their metadata
    test_videos = []
    
    for gloss_idx, gloss_entry in enumerate(wlasl_data):
        gloss_name = gloss_entry['gloss']
        gloss_folder = str(gloss_idx)
        gloss_folder_path = os.path.join(videos_path, gloss_folder)
        
        if not os.path.exists(gloss_folder_path):
            continue
            
        for instance in gloss_entry['instances']:
            if instance.get('split') == 'test':
                video_id = instance['video_id']
                video_filename = f"{video_id}.mp4"
                video_path = os.path.join(gloss_folder_path, video_filename)
                
                if os.path.exists(video_path):
                    test_videos.append({
                        'video_path': video_path,
                        'ground_truth_gloss': gloss_name,
                        'gloss_id': gloss_idx,
                        'video_id': video_id,
                        'fps': instance.get('fps', 25)
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
        
        processed_count = 0
        errors_count = 0
        
        for i, video_info in enumerate(test_videos):
            start_time = datetime.now()
            
            try:
                print(f"Processing video {i+1}/{len(test_videos)}: {video_info['video_id']} - {video_info['ground_truth_gloss']}")
                
                predicted_gloss = predict_gloss(
                    client,
                    video_info['video_path'],
                    model_name=model_name,
                    num_frames=num_frames,
                    max_output_tokens=max_output_tokens
                )
                
                # Clean up predicted gloss
                predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                predicted_gloss = ' '.join(predicted_gloss.split())
                
                # Retry mechanism
                if predicted_gloss == "EMPTY_RESPONSE":
                    retry_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY - Video: {video_info['video_id']} - First attempt returned empty response, retrying...\n"
                    with open(log_file_path, 'a', encoding='utf-8') as logfile:
                        logfile.write(retry_message)
                    print(f"  → First attempt returned empty response, retrying...")
                    
                    predicted_gloss = predict_gloss(
                        client,
                        video_info['video_path'],
                        model_name=model_name,
                        num_frames=num_frames,
                        max_output_tokens=max_output_tokens
                    )
                    
                    predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                    predicted_gloss = ' '.join(predicted_gloss.split())
                    
                    if predicted_gloss == "EMPTY_RESPONSE":
                        retry_fail_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY_FAILED - Video: {video_info['video_id']} - Second attempt also returned empty response\n"
                        with open(log_file_path, 'a', encoding='utf-8') as logfile:
                            logfile.write(retry_fail_message)
                        print(f"  → Second attempt also returned empty response")
                    else:
                        retry_success_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY_SUCCESS - Video: {video_info['video_id']} - Second attempt succeeded: {predicted_gloss}\n"
                        with open(log_file_path, 'a', encoding='utf-8') as logfile:
                            logfile.write(retry_success_message)
                        print(f"  → Second attempt succeeded: '{predicted_gloss}'")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                writer.writerow({
                    'ground_truth_gloss': video_info['ground_truth_gloss'],
                    'gloss_id': video_info['gloss_id'],
                    'video_id': video_info['video_id'],
                    'predicted_gloss': predicted_gloss,
                    'num_frames': num_frames,
                    'processing_time': processing_time
                })
                
                log_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - SUCCESS - Video: {video_info['video_id']} - GT: {video_info['ground_truth_gloss']} - Predicted: {predicted_gloss} - Time: {processing_time:.2f}s\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(log_message)
                
                processed_count += 1
                print(f"  → Predicted: '{predicted_gloss}' (took {processing_time:.2f}s)")
                
            except Exception as e:
                error_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - ERROR - Video: {video_info['video_id']} - GT: {video_info['ground_truth_gloss']} - Error: {str(e)}\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(error_message)
                
                print(f"  → ERROR: {str(e)}")
                errors_count += 1
                
                writer.writerow({
                    'ground_truth_gloss': video_info['ground_truth_gloss'],
                    'gloss_id': video_info['gloss_id'],
                    'video_id': video_info['video_id'],
                    'predicted_gloss': f"ERROR: {str(e)}",
                    'num_frames': num_frames,
                    'processing_time': 0
                })
            
            if (i + 1) % 10 == 0:
                csvfile.flush()
                print(f"Progress: {i+1}/{len(test_videos)} videos processed")
    
    summary_message = f"""
=== PROCESSING COMPLETE ===
Model: {model_name}
Total videos: {len(test_videos)}
Successfully processed: {processed_count}
Errors: {errors_count}
CSV output: {csv_file_path}
Log output: {log_file_path}
    """
    
    print(summary_message)
    
    with open(log_file_path, 'a', encoding='utf-8') as logfile:
        logfile.write(f"\n{datetime.now().isoformat()} - SUMMARY:\n{summary_message}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASL Gloss Prediction using Google Gemini API')
    parser.add_argument('--api-key', type=str, required=True, help='Google GenAI API key')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Model name to use (default: gemini-2.5-flash)')
    parser.add_argument('--num-frames', type=int, default=8, help='Number of frames to extract from each video (default: 8)')
    parser.add_argument('--max-output-tokens', type=int, default=4096, help='Maximum output tokens (default: 4096)')
    parser.add_argument('--videos-path', type=str, default=videos_path, help='Path to videos directory')
    parser.add_argument('--json-path', type=str, default=json_file_path, help='Path to WLASL JSON file')
    parser.add_argument('--output-dir', type=str, default=output_dir, help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"Start date and time = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Max output tokens: {args.max_output_tokens}")
    
    process_dataset(
        args.json_path,
        args.videos_path,
        args.output_dir,
        args.api_key,
        model_name=args.model,
        num_frames=args.num_frames,
        max_output_tokens=args.max_output_tokens
    )
    
    print(f"End date and time = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


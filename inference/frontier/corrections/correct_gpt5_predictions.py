import os
import csv
import argparse
import base64
from datetime import datetime
import cv2
from openai import OpenAI
import time

# Configuration
default_videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
default_output_dir = "output/"

def extract_frames_from_video(video_path, num_frames=8):
    """Extract evenly spaced frames from a video for GPT processing."""
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

def encode_frame_to_base64(frame):
    """Encode a cv2 frame to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def predict_gloss(client, video_path, model_name="gpt-5-nano", num_frames=8, max_completion_tokens=4096):
    """Predict gloss from an ASL video using OpenAI GPT API."""
    
    system_content = (
        "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed. "
        "Provide the gloss of the sign language video, output only the gloss, no other text. "
        "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions. "
        "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    )
    
    # Extract frames from video
    frames = extract_frames_from_video(video_path, num_frames=num_frames)
    
    # Prepare message content with frames
    content = [
        {
            "type": "text",
            "text": "What is the ASL gloss for this sign shown in these sequential frames? Output only the single English word or phrase."
        }
    ]
    
    # Add each frame as an image
    for frame in frames:
        base64_frame = encode_frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}"
            }
        })
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content}
    ]
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens
    )
    
    # Extract the response text
    choice = response.choices[0]
    message = choice.message
    output_text = message.content
    
    # Check if there's a refusal
    if hasattr(message, 'refusal') and message.refusal:
        return f"REFUSED: {message.refusal}"
    
    # Handle empty responses
    if not output_text:
        return "EMPTY_RESPONSE"
    
    return output_text.strip()

def process_corrections(input_csv_path, videos_path, output_dir, api_key, model_name="gpt-5-nano", num_frames=8, max_completion_tokens=4096):
    """Process a dataset of videos and save results to CSV and log files."""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    output_csv_name = os.path.basename(input_csv_path).replace('.csv', f'_corrected_{timestamp}.csv')
    output_csv_path = os.path.join(output_dir, output_csv_name)
    log_file_path = os.path.join(output_dir, f"wlasl_correction_{safe_model_name}_{timestamp}.log")
    
    print(f"Reading input CSV: {input_csv_path}")
    rows = []
    with open(input_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows. Scanning for EMPTY_RESPONSE...")
    
    error_rows = [row for row in rows if row['predicted_gloss'] == 'EMPTY_RESPONSE']
    print(f"Found {len(error_rows)} rows to correct.")
    
    if len(error_rows) == 0:
        print("No corrections needed.")
        return

    # Initialize CSV file with all rows initially
    # We will update rows as we process them and write the final result at the end?
    # Or better: write progress as we go to a temp list, then write everything.
    # To be safe against crashes, let's write processed rows as we go? 
    # But we need to output the FULL file including good rows.
    # Strategy: Create a copy of rows, update them in place, and save intermediate results or final result.
    # Given the retry logic, it's better to process and update the list, then write the full file.
    
    # However, to keep a log of progress, we'll write to the log file.
    
    processed_count = 0
    success_count = 0
    still_empty_count = 0
    error_count = 0

    for i, row in enumerate(rows):
        if row['predicted_gloss'] == 'EMPTY_RESPONSE':
            gloss_id = row['gloss_id']
            video_id = row['video_id']
            ground_truth = row['ground_truth_gloss']
            
            # Construct video path
            # Assuming structure: videos_path/gloss_id/video_id.mp4
            video_path = os.path.join(videos_path, str(gloss_id), f"{video_id}.mp4")
            
            if not os.path.exists(video_path):
                print(f"Warning: Video not found at {video_path}")
                log_msg = f"{datetime.now().isoformat()} - ERROR - Video not found: {video_path}\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(log_msg)
                continue
                
            print(f"Correcting video {processed_count+1}/{len(error_rows)}: {video_id} - {ground_truth}")
            processed_count += 1
            
            start_time = datetime.now()
            
            try:
                # First attempt
                predicted_gloss = predict_gloss(
                    client,
                    video_path,
                    model_name=model_name,
                    num_frames=num_frames,
                    max_completion_tokens=max_completion_tokens
                )
                
                # Clean up
                predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                predicted_gloss = ' '.join(predicted_gloss.split())
                
                # Retry mechanism
                if predicted_gloss == "EMPTY_RESPONSE":
                    retry_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY - Video: {video_id} - First attempt returned empty response, retrying...\n"
                    with open(log_file_path, 'a', encoding='utf-8') as logfile:
                        logfile.write(retry_message)
                    print(f"  → First attempt returned empty response, retrying...")
                    
                    # Second attempt
                    predicted_gloss = predict_gloss(
                        client,
                        video_path,
                        model_name=model_name,
                        num_frames=num_frames,
                        max_completion_tokens=max_completion_tokens
                    )
                    
                    # Clean up
                    predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                    predicted_gloss = ' '.join(predicted_gloss.split())
                    
                    if predicted_gloss == "EMPTY_RESPONSE":
                        fail_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY_FAILED - Video: {video_id} - Second attempt also returned empty response\n"
                        with open(log_file_path, 'a', encoding='utf-8') as logfile:
                            logfile.write(fail_msg)
                        print(f"  → Second attempt also returned empty response")
                        still_empty_count += 1
                    else:
                        success_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY_SUCCESS - Video: {video_id} - Second attempt succeeded: {predicted_gloss}\n"
                        with open(log_file_path, 'a', encoding='utf-8') as logfile:
                            logfile.write(success_msg)
                        print(f"  → Second attempt succeeded: '{predicted_gloss}'")
                        success_count += 1
                else:
                    success_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - SUCCESS - Video: {video_id} - Correction succeeded: {predicted_gloss}\n"
                    with open(log_file_path, 'a', encoding='utf-8') as logfile:
                        logfile.write(success_msg)
                    print(f"  → Correction succeeded: '{predicted_gloss}'")
                    success_count += 1
                
                # Update row
                processing_time = (datetime.now() - start_time).total_seconds()
                row['predicted_gloss'] = predicted_gloss
                row['processing_time'] = processing_time # Update time to the correction time
                
            except Exception as e:
                error_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - ERROR - Video: {video_id} - Error during correction: {str(e)}\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(error_msg)
                print(f"  → ERROR: {str(e)}")
                error_count += 1
                # Keep as EMPTY_RESPONSE or ERROR
                row['predicted_gloss'] = f"ERROR: {str(e)}"

    # Write final CSV
    print(f"Writing corrected CSV to: {output_csv_path}")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = f"""
=== CORRECTION COMPLETE ===
Total rows checked: {len(rows)}
Errors found: {len(error_rows)}
Successfully corrected: {success_count}
Still empty: {still_empty_count}
Processing errors: {error_count}
Output file: {output_csv_path}
    """
    print(summary)
    with open(log_file_path, 'a', encoding='utf-8') as logfile:
        logfile.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correct ASL Gloss Predictions (EMPTY_RESPONSE) using OpenAI GPT API')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to input CSV file with EMPTY_RESPONSEs')
    parser.add_argument('--model', type=str, default='gpt-5-nano', help='Model name to use (default: gpt-5-nano)')
    parser.add_argument('--num-frames', type=int, default=8, help='Number of frames to extract from each video (default: 8)')
    parser.add_argument('--max-completion-tokens', type=int, default=4096, help='Maximum completion tokens (default: 4096)')
    parser.add_argument('--videos-path', type=str, default=default_videos_path, help='Path to videos directory')
    parser.add_argument('--output-dir', type=str, default=default_output_dir, help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"Start correction = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Model: {args.model}")
    
    process_corrections(
        args.input_csv,
        args.videos_path,
        args.output_dir,
        args.api_key,
        model_name=args.model,
        num_frames=args.num_frames,
        max_completion_tokens=args.max_completion_tokens
    )
    
    print(f"End correction = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


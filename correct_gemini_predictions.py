import os
import json
import csv
import argparse
import time
from datetime import datetime
import cv2
from google import genai
from google.genai import types

# Configuration
default_videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
default_output_dir = "output/"

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
        # Check for specific errors that should be retried in the main loop
        error_str = str(e).lower()
        if "resource_exhausted" in error_str or "429" in error_str:
             raise Exception(f"RESOURCE_EXHAUSTED: {str(e)}")
        if "internal" in error_str or "500" in error_str:
             raise Exception(f"INTERNAL_ERROR: {str(e)}")
             
        # Check for safety refusals or other API errors
        if "refusal" in error_str or "safety" in error_str:
             return f"REFUSED: {str(e)}"
        raise e

def process_corrections(input_csv_path, videos_path, output_dir, api_key, model_name="gemini-2.0-flash", num_frames=8, max_output_tokens=4096):
    """Process corrections for a dataset of videos and save results to CSV and log files."""
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
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
    
    print(f"Loaded {len(rows)} rows. Scanning for EMPTY_RESPONSE, RESOURCE_EXHAUSTED, and INTERNAL errors...")
    
    # Function to check if a row needs correction
    def needs_correction(row):
        val = row['predicted_gloss']
        return (val == 'EMPTY_RESPONSE' or 
                'RESOURCE_EXHAUSTED' in val or 
                'INTERNAL' in val or 
                val.startswith('ERROR:'))

    error_rows = [row for row in rows if needs_correction(row)]
    print(f"Found {len(error_rows)} rows to correct.")
    
    if len(error_rows) == 0:
        print("No corrections needed.")
        return

    processed_count = 0
    success_count = 0
    still_error_count = 0
    
    for i, row in enumerate(rows):
        if needs_correction(row):
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
            predicted_gloss = "EMPTY_RESPONSE"
            
            try:
                # First attempt
                try:
                    predicted_gloss = predict_gloss(
                        client,
                        video_path,
                        model_name=model_name,
                        num_frames=num_frames,
                        max_output_tokens=max_output_tokens
                    )
                except Exception as e:
                    predicted_gloss = f"ERROR: {str(e)}"
                    # We want to retry on specific errors, so suppress exception here to hit retry logic
                    if "RESOURCE_EXHAUSTED" in str(e) or "INTERNAL_ERROR" in str(e):
                         pass
                    else:
                         # Log other errors but maybe don't crash the whole script
                         pass

                # Clean up
                if not predicted_gloss.startswith("ERROR:") and predicted_gloss != "EMPTY_RESPONSE":
                    predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                    predicted_gloss = ' '.join(predicted_gloss.split())
                
                # Generalized Retry mechanism
                should_retry = False
                retry_reason = ""
                
                if predicted_gloss == "EMPTY_RESPONSE":
                    should_retry = True
                    retry_reason = "EMPTY_RESPONSE"
                elif "RESOURCE_EXHAUSTED" in predicted_gloss:
                    should_retry = True
                    retry_reason = "RESOURCE_EXHAUSTED"
                    print("  → Resource exhausted, sleeping for 5 seconds...")
                    time.sleep(5)
                elif "INTERNAL_ERROR" in predicted_gloss:
                    should_retry = True
                    retry_reason = "INTERNAL_ERROR"
                    print("  → Internal error, sleeping for 2 seconds...")
                    time.sleep(2)
                elif predicted_gloss.startswith("ERROR:"):
                    # Could retry other errors too, but let's stick to these for now or treat all ERRORs as retriable?
                    # The prompt asked to generalize retry for errors. Let's retry all errors once.
                    should_retry = True
                    retry_reason = "ERROR"
                
                if should_retry:
                    retry_message = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY ({retry_reason}) - Video: {video_id} - First attempt failed, retrying...\n"
                    with open(log_file_path, 'a', encoding='utf-8') as logfile:
                        logfile.write(retry_message)
                    print(f"  → First attempt failed ({retry_reason}), retrying...")
                    
                    # Second attempt
                    try:
                        predicted_gloss = predict_gloss(
                            client,
                            video_path,
                            model_name=model_name,
                            num_frames=num_frames,
                            max_output_tokens=max_output_tokens
                        )
                        
                        if predicted_gloss != "EMPTY_RESPONSE":
                            predicted_gloss = predicted_gloss.strip().replace('\n', ' ').replace('\r', ' ')
                            predicted_gloss = ' '.join(predicted_gloss.split())
                            
                    except Exception as e:
                        predicted_gloss = f"ERROR: {str(e)}"
                    
                    # Check result
                    if needs_correction({'predicted_gloss': predicted_gloss}):
                        fail_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY_FAILED - Video: {video_id} - Second attempt failed: {predicted_gloss}\n"
                        with open(log_file_path, 'a', encoding='utf-8') as logfile:
                            logfile.write(fail_msg)
                        print(f"  → Second attempt failed: {predicted_gloss}")
                        still_error_count += 1
                    else:
                        success_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - RETRY_SUCCESS - Video: {video_id} - Second attempt succeeded: {predicted_gloss}\n"
                        with open(log_file_path, 'a', encoding='utf-8') as logfile:
                            logfile.write(success_msg)
                        print(f"  → Second attempt succeeded: '{predicted_gloss}'")
                        success_count += 1
                else:
                    # First attempt succeeded
                    success_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - SUCCESS - Video: {video_id} - Correction succeeded: {predicted_gloss}\n"
                    with open(log_file_path, 'a', encoding='utf-8') as logfile:
                        logfile.write(success_msg)
                    print(f"  → Correction succeeded: '{predicted_gloss}'")
                    success_count += 1
                
                # Update row
                processing_time = (datetime.now() - start_time).total_seconds()
                row['predicted_gloss'] = predicted_gloss
                row['processing_time'] = processing_time
                
            except Exception as e:
                error_msg = f"{datetime.now().isoformat()} - MODEL: {model_name} - CRITICAL_ERROR - Video: {video_id} - Error during correction: {str(e)}\n"
                with open(log_file_path, 'a', encoding='utf-8') as logfile:
                    logfile.write(error_msg)
                print(f"  → CRITICAL ERROR: {str(e)}")
                still_error_count += 1
                row['predicted_gloss'] = f"CRITICAL_ERROR: {str(e)}"

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
Still errors: {still_error_count}
Output file: {output_csv_path}
    """
    print(summary)
    with open(log_file_path, 'a', encoding='utf-8') as logfile:
        logfile.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correct ASL Gloss Predictions (EMPTY/ERRORS) using Gemini API')
    parser.add_argument('--api-key', type=str, required=True, help='Google GenAI API key')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to input CSV file with errors')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Model name to use (default: gemini-2.5-flash)')
    parser.add_argument('--num-frames', type=int, default=8, help='Number of frames to extract from each video (default: 8)')
    parser.add_argument('--max-output-tokens', type=int, default=4096, help='Maximum output tokens (default: 4096)')
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
        max_output_tokens=args.max_output_tokens
    )
    
    print(f"End correction = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


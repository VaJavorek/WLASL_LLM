# %load_ext autoreload
# %autoreload 2
import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO
import random
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

import sys
sys.path.append("BAGEL/")

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file

model_path = "/auto/plzen4-ntis/home/honzikj/git/WLASL_LLM/models/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
json_file_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/WLASL_v0.3.json"
output_dir = "output/"

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')

import json
import decord
# sys.path.append("frame_selection")
from frame_selection.data_handlers import get_adaptive_frames, get_equally_distributed_frames
from PIL import Image
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import csv
import re

max_length = 512

def predict_gloss(video_path, from_frames=True, cut_edges=False, fps=16, required_frames=64, max_w=224, max_h=224, use_img_input=True):
    """Predict gloss from an ASL video using Qwen model."""
    
    # system_content = (
    #     "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed."
    #     "Provide the gloss of the sign language video, output only the gloss, no other text."
    #     "Glosses are from WLASL dataset."
    #     "Do not say: The ASL gloss from WLASL300 for this sign is.. Only use one word."
    #     # "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions."
    #     "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    # )

    system_content = (
        "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed."
        "Provide the gloss of the sign language video, output only the gloss, no other text."
        "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions."
        "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    )


    
    # system_content = (
    #     # "You are an expert ASL (American Sign Language) gloss annotator. Your task is to watch ASL videos and predict the EXACT single English gloss that represents the sign being performed."
    #     "You have a task to explain what ASL (American Sign Language) gloss is shown on the video."
    #     "Firstly explain your thought process what happens in the video and in the end of your message, provide the gloss of the sign language video."
    #     "The final gloss should be one word."
    #     # "Do NOT output phrases like 'woman signing' or 'hand movements' or descriptions."
    #     "Examples of correct ASL glosses: BOOK, RUN, HAPPY, WATER, SCHOOL, COMPUTER, DANCE, etc."
    # )

    
    prompt = "What is the ASL gloss from WLASL300 for this sign? Firstly, explain what is in the video and then say what ASL gloss it is. The final word must represent the ASL gloss."
    prompt = "What is the ASL gloss from WLASL300 for this sign? Note that the video should match description of a gloss given earlier."# Firstly, explain what is in the video and then say what ASL gloss it is. The final word must represent the ASL gloss."
    # prompt = "What is the ASL gloss for this sign? Note that the video should match description of a gloss given earlier."
    prompt = "What is the ASL gloss for this sign?"

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
            full_prompt = f"{system_content}\n{prompt}"
            result = model.chat(tokenizer=tokenizer,
                        new_token_ids=new_token_ids,
                        image_transform=vit_transform, #vit_transform, #vae_transform
                        images=pil_images,
                        prompt=prompt,
                        max_length=max_length)
            # return result if result else ""
            # return result.split()[0].replace("*", "") if result else ""
            return result.split()[-1].replace("*", "") if result else ""
        else:
            messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": [
                        {"type": "video", "fps": fps},
                        {"type": "text", "text": prompt},
                        ]
                    },
                ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
            inputs = processor(
                text=text,
                videos=frames,
                # images=pil_images,
                fps=fps, 
                padding=True,
                return_tensors="pt"
            ).to(model.device)
        
    # if model loads from file
    if not from_frames:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "video", "video": f"file://{video_path}", "fps": fps},
                {"type": "text", "text": prompt},
                # {"type": "file", "file": open(glosses_csv, "rb")}
            ]}
        ]
        
        # Apply chat template and process vision inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True) 
        
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            if len(video_kwargs["fps"]) == 1:
                video_kwargs["fps"] = video_kwargs["fps"][0]
            else:
                raise ValueError(f"Unexpected multiple fps values: {video_kwargs['fps']}")
        # print(len(video_inputs))
        # print(video_inputs[0].shape, video_tensor.dtype, video_tensor.device)
        # print(video_inputs[0].shape)
        # Prepare model inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,  # Include video-specific kwargs
        )
        inputs = inputs.to("cuda")
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.1)
    # generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=1)
    # print(processor.batch_decode(generated_ids, skip_special_tokens=True))
    # Trim prompt tokens and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    # print(output_text[0].split()[-1].replace("*", ""))

    # return output_text[0] if output_text else ""
    return output_text[0].split()[-1].replace("*", "") if output_text else ""

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
                predicted_gloss = predict_gloss(video_info['video_path'], from_frames=True, fps=video_fps)
                
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
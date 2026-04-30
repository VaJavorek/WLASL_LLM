import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import random
import csv
import re
from datetime import datetime

# Configuration
# --- Set seeds ---
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# local_path = "models/Qwen3-VL-235B-A22B-Instruct"
local_path = "../models/Qwen3-VL-30B-A3B-Instruct"
model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# local_path = "../models/Qwen3-VL-32B-Instruct"
# model_name = "Qwen/Qwen3-VL-32B-Instruct"
logged_model_name = re.sub(r'^[\\/]*models[\\/]+', '', local_path)
videos_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/test"
json_file_path = "/auto/plzen4-ntis/projects/korpusy_cv/WLASL/WLASL300/WLASL_v0.3.json"
min_pixels = 256 * 40 * 40
max_pixels = 1080 * 40 * 40
output_dir = "/auto/plzen4-ntis/projects/cv/WLASL_LLM/gloss_descriptions/output/"

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    local_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True
)


# Load JSON data
print("Loading WLASL JSON data...")
with open(json_file_path, 'r') as f:
    wlasl_data = json.load(f)

with open("../internet_wlasl300_descriptions.json") as f:
    wlasl300_descriptions = json.load(f)

print(f"Loaded {len(wlasl_data)} glosses from JSON")

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

#compares video with a description
def compare_vid_and_desc(video_path, description, system_content=None, fps=4):
    if system_content is None:
        system_content = (
                "You are given description of an ASL gloss. Does it fully fit the given video?"
                "Firstly describe hand movements, handshape of both hands in the video. Compare it with the given description. Lastly, give a YES or NO answer. If everything fits, say YES. If there is something that does not fit, say NO. Be sceptical."
                "The last word has to be YES or NO, nothing else."
            )
    
    prompt = description
    
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
    # Trim prompt tokens and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # return output_text[0] if output_text else ""
    return output_text[0].split()[-1].replace("*", "").replace(".", "") if output_text else ""


# Configuration
# --- Set seeds ---
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




#more sceptical
system_content = (
            "You are given description of an ASL gloss. Does it fully fit the given video? Focus on every detail like orientation of hands handshapes and movement. "
            "Firstly think about how you would describe hand movements, handshape of both hands in the video. After that, explain in what ways the description might NOT fit the video. Then explain in what ways it might fit. Lastly, give a YES or NO answer. If everything fits, say YES. If there is something that does not fit, say NO. "
        "Do not repeat the description if it matches and just answer. "    
        "The last word has to be YES or NO, nothing else."
        )
# #less sceptical
# system_content = (
#             "You are given description of an ASL gloss. Does it fully fit the given video? Focus on every detail like orientation of hands handshapes and movement. "
#             "Firstly describe hand movements, handshape of both hands in the video. Compare it with the given description. Lastly, give a YES or NO answer. If everything fits, say YES. If there is something that does not fit, say NO. "
#             "The last word has to be YES or NO, nothing else."
#             # "Give a YES or NO answer. And Explain why"
#         )
with_description = True
#no description
# system_content = (
#             "You are given an ASL gloss in the prompt. Is this gloss shown in the video? Focus on every detail like orientation of hands handshapes and movement. "
#             "Firstly describe hand movements, handshape of both hands in the video. Compare it with your understanding of the given gloss. Lastly, give a YES or NO answer. If everything fits, say YES. If there is something that does not fit, say NO. "
#             "The last word has to be YES or NO, nothing else."
#             # "Give a YES or NO answer. And Explain why"
#         )
# with_description = False
all_tp = 0
all_fp = 0
all_fn = 0
all_tn = 0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(os.path.join(output_dir, f"glosses_YesNo_{timestamp}.log"), "a") as f:
    f.write(model_name + "\n")
    # f.write("gemini 2.5" + "\n")
    f.write(system_content + "\n")
    for i, gloss in enumerate(wlasl300_descriptions):
        print(gloss)
        # gloss_videos = [v for v in val_videos if v['ground_truth_gloss'] == gloss] + [v for v in train_videos if v['ground_truth_gloss'] == gloss]
        # videos = [v for v in val_videos if v['ground_truth_gloss'] != gloss]
        gloss_videos = [v for v in test_videos if v['ground_truth_gloss'] == gloss]
        videos = [v for v in test_videos if v['ground_truth_gloss'] != gloss]
        samples = random.sample(range(len(videos)), len(gloss_videos))
        other_gloss_videos = [videos[i] for i in samples]
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        exceptions = []
        if with_description:
            description = wlasl300_descriptions[gloss] #qwens_text_description #video_description
        else:
            description = gloss
        for j, video_info in enumerate(gloss_videos):
            try:
                answer = compare_vid_and_desc(video_info['video_path'], description, system_content)
            except Exception as e:
                answer = e
            video_gloss = video_info['ground_truth_gloss']
            # print(str(i)+video_gloss+answer )
            if video_gloss == gloss and answer == "YES":
                tp += 1
            elif video_gloss != gloss and answer == "YES":
                fp += 1
            elif video_gloss == gloss and answer == "NO":
                fn += 1
            elif video_gloss != gloss and answer == "NO":
                tn += 1
            else:
                exceptions.append(video_gloss + ": " + str(answer))
        for j, video_info in enumerate(other_gloss_videos):
            try:
                answer = compare_vid_and_desc(video_info['video_path'], description, system_content)
            except Exception as e:
                answer = e
            video_gloss = video_info['ground_truth_gloss']
            # print(str(i)+video_gloss+answer )
            if video_gloss == gloss and answer == "YES":
                tp += 1
            elif video_gloss != gloss and answer == "YES":
                fp += 1
            elif video_gloss == gloss and answer == "NO":
                fn += 1
            elif video_gloss != gloss and answer == "NO":
                tn += 1
            else:
                exceptions.append(video_gloss + ": " + str(answer))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        
        f.write(gloss + "\n")
        f.write(description + "\n")
        f.write("true positive: " + str(tp) + "\n")
        f.write("false negative: " + str(fn) + "\n")
        f.write("false positive: " + str(fp) + "\n")
        f.write("true negative: " + str(tn) + "\n")
        f1 = 2*tp/(2*tp+fp+fn)
        f.write("f1: " + str(f1) + "\n")
        f.write("number of exceptions = "  + str(len(exceptions)) + "\n")
        for exception in exceptions:
            f.write(exception + "\n")
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tn += tn
    f.write("all glosses" + "\n")
    f.write("true positive: " + str(all_tp) + "\n")
    f.write("false negative: " + str(all_fn) + "\n")
    f.write("false positive: " + str(all_fp) + "\n")
    f.write("true negative: " + str(all_tn) + "\n")
    all_f1 = 2*all_tp/(2*all_tp+all_fp+all_fn)
    f.write("f1: " + str(all_f1) + "\n")

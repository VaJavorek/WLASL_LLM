# WLASL_LLM
Using LLMs to tackle Sign Language Recognition with the WLASL dataset

## Installation

```bash
conda create -n qwen3 python=3.10
conda activate qwen3

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate safetensors pillow

pip install qwen-vl-utils
pip install qwen-omni-utils
pip install flash-attn --no-build-isolation
pip install decord
```

### For frontier models also add:
```bash
pip install google-genai
pip install openai
```
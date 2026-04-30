This folder provides scripts for collecting descriptions of ASL glosses and using them for experiments.

## get_descriptions_from_internet.ipynb
This notebook collects textual desctiptions of WLASL300 glosses from https://www.handspeak.com.

Additionaly, glosses that have different names in the website from WLASL300 labels that contain the same gloss, the glosses are manualy added to the list of gloss descriptions.

## gloss_descriptions.ipynb
This notebook uses the gloss descriptions gained from get_descriptions_from_internet.ipynb and gives them to the model in the prompt.
The model compares these descriptions with the video in the input and decides if the video fits the description.

The script has pipeline for Qwen3-VL and for Gemini api.

The settings for experiments with nonskeptical, skeptical and no description scenarios for Qwen3-VL are showed in the following scripts. \
nonskeptical: gloss_descriptions_qwen_nonskept.py \
skeptical: gloss_descriptions_qwen_skept.py \
no description: gloss_descriptions_qwen_nodesc.py

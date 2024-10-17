#!/usr/bin/env python3

import os

import requests

from imfind import image_and_text_to_text
from imfind.etc import default_prompt

# Try a test run of loading big model (LLaVa 1.5) into memory.
# If success please set IMFIND_USE_LLAVA to "True"
 
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = requests.get(url, stream=True).raw
os.environ["USE_MMRY_CACHE"] = "False"
description = image_and_text_to_text(image, default_prompt)

#!/usr/bin/env python3

### Adapted from notarealdeveloper/kern repository 
### https://github.com/notarealdeveloper/kern/blob/master/src/kern/image/lib/image_and_text.py 

__all__ = [
    'image_and_text_to_text', 'load_model_image_and_text_to_text'
]

import gc
import io
import os
import re
from functools import lru_cache

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_and_text_to_text(image, text, use_cache=True):

    import assure
    from mmry import Cache

    from imfind import easyocr

    image_bytes = assure.bytes(image)
    text_bytes = text.encode()
    bytes = image_bytes + text_bytes

    cache = Cache('image_and_text_to_text')
    if use_cache and cache.have_blob(bytes):
        return cache.load_blob(bytes).decode()

    text = image_and_text_to_text_nocache(image_bytes, text)

    ocr_text = easyocr(image_bytes)
    text += '\n' + ocr_text + '\n'
    cache.save_blob(bytes, text)
    return text

def image_and_text_to_text_nocache(image_bytes, text):

    import PIL.Image
    from pillow_heif import register_heif_opener
    register_heif_opener()

    file = io.BytesIO(image_bytes)
    image = PIL.Image.open(file)

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
            ],
        },
    ]

    processor, model = load_model_image_and_text_to_text()

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    encoding = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)

    generate_ids = model.generate(**encoding, max_new_tokens=1024, do_sample=False)
    output = processor.decode(generate_ids[0], skip_special_tokens=True)

    del encoding # frees up any gpu memory taken by this
    torch.cuda.empty_cache()
    _ = gc.collect()

    m = re.search("ASSISTANT:", output)
    return output[m.end():].strip()

@lru_cache(maxsize=1)
def load_model_image_and_text_to_text():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(name)

    # Using device_map='auto' with single GPU tends to slow it down significantly (by ~4 times)
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        model = LlavaForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
    else:
        model = LlavaForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True,).to(device)
    return (processor, model)

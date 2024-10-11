#!/usr/bin/env python3

### Adapted from notarealdeveloper/kern repository 
### https://github.com/notarealdeveloper/kern/blob/master/src/kern/image/lib/image.py 

__all__ = [
    'image_to_text',
]

import io
import os
from functools import lru_cache
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_text(image):

    import assure
    from mmry import Cache

    bytes = assure.bytes(image)

    cache = Cache('image_to_text')
    use_cache = os.environ.get("USE_MMRY_CACHE") # is None if not set, then by default use cache. If set, needs to be either "True" or "False"
    if eval(use_cache or "True") and cache.have_blob(bytes):
        return cache.load_blob(bytes).decode()

    text = image_to_text_nocache(bytes)
    cache.save_blob(bytes, text.encode())
    return text


def image_to_text_nocache(bytes):

    import PIL.Image
    from pillow_heif import register_heif_opener
    register_heif_opener()

    file = io.BytesIO(bytes)
    image = PIL.Image.open(file).convert('RGB')

    processor, model = load_model_image_to_text()
    
    encoding = processor(image, return_tensors="pt").to(device)

    generate_ids = model.generate(**encoding, max_new_tokens=1024)
    output =  processor.decode(generate_ids[0], skip_special_tokens=True)
    return output


@lru_cache(maxsize=1)
def load_model_image_to_text():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name).to(device)
    return (processor, model)

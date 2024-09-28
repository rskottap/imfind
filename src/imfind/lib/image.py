#!/usr/bin/env python3

### Adapted from notarealdeveloper/kern repository 
### https://github.com/notarealdeveloper/kern/blob/master/src/kern/image/lib/image.py 

__all__ = [
    'image_to_text',
]

import io
import os
from functools import lru_cache


def image_to_text(image):

    import assure
    from mmry import Cache

    bytes = assure.bytes(image)

    cache = Cache('image_to_text')
    if eval(os.environ.get("USE_MMRY_CACHE")) and cache.have_blob(bytes):
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
    
    encoding = processor(image, return_tensors="pt")

    generate_ids = model.generate(**encoding, max_new_tokens=512)
    output =  processor.decode(generate_ids[0], skip_special_tokens=True)
    return output


@lru_cache(maxsize=1)
def load_model_image_to_text():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    
    return (processor, model)

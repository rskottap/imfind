#!/usr/bin/env python3

### Adapted from notarealdeveloper/kern repository 
### https://github.com/notarealdeveloper/kern/blob/master/src/kern/image/lib/image_and_text.py 

__all__ = [
    'image_and_text_to_text',
]

import io
import os
from functools import lru_cache
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_and_text_to_text(image, text):

    import assure
    from mmry import Cache

    image_bytes = assure.bytes(image)
    text_bytes = text.encode()
    bytes = image_bytes + text_bytes

    cache = Cache('image_and_text_to_text')
    use_cache = os.environ.get("USE_MMRY_CACHE") # if not set, by default is True, so uses the cache
    if eval(use_cache if use_cache else "True") and cache.have_blob(bytes):
        return cache.load_blob(bytes).decode()

    text = image_and_text_to_text_nocache(image_bytes, text)
    cache.save_blob(bytes, text)
    return text

def image_and_text_to_text_nocache(image_bytes, text):

    import PIL.Image
    from pillow_heif import register_heif_opener
    register_heif_opener()

    file = io.BytesIO(image_bytes)
    image = PIL.Image.open(file)

    template = f"USER:  \n{text}\nASSISTANT:\n"
    processor, model = load_model_image_and_text_to_text()
    encoding = processor(image, template.format(text), return_tensors="pt").to(device)
 
    import time
    s = time.time()
    generate_ids = model.generate(**encoding, max_new_tokens=1024)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    e = time.time()
    print(f"Time taken for generation and decoding: {e-s} seconds")
    return output

@lru_cache(maxsize=1)
def load_model_image_and_text_to_text():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(name)
    model = LlavaForConditionalGeneration.from_pretrained(name).to(device)
    return (processor, model)

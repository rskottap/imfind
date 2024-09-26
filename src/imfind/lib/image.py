#!/usr/bin/env python3

### Adapted from notarealdeveloper/kern repository 
### https://github.com/notarealdeveloper/kern/blob/master/src/kern/image/lib/image.py 

__all__ = [
    'image_to_text',
]

import io
from functools import lru_cache


def image_to_text(image):

    import assure
    from mmry import Cache

    bytes = assure.bytes(image)

    cache = Cache('image_to_text')
    if cache.have_blob(bytes):
        return cache.load_blob(bytes).decode()

    text = image_to_text_nocache(bytes)
    cache.save_blob(bytes, text.encode())
    return text


def image_to_text_nocache(bytes):

    import PIL.Image
    file = io.BytesIO(bytes)
    image = PIL.Image.open(file).convert('RGB')

    processor, model = load_model_image_to_text()
    
    encoding = processor(image, return_tensors="pt")

    outputs = model.generate(**encoding, max_new_tokens=1024)
    return processor.decode(outputs[0], skip_special_tokens=True)
   
@lru_cache(maxsize=1)
def load_model_image_to_text():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    
    return (processor, model)

#!/usr/bin/env python3

### Adapted from notarealdeveloper/kern repository 
### https://github.com/notarealdeveloper/kern/blob/master/src/kern/image/lib/image_and_text.py 

__all__ = [
    'image_and_text_to_text',
]

import io
from functools import lru_cache

def image_and_text_to_text(image, text):

    import assure
    from mmry import Cache

    image_bytes = assure.bytes(image)
    text_bytes = text.encode()
    bytes = image_bytes + text_bytes

    cache = Cache('image_and_text_to_text')
    if cache.have_blob(bytes):
        return cache.load_blob(bytes).decode()

    text = image_and_text_to_text_nocache(image_bytes, text)
    cache.save_blob(bytes, text)
    return text

def image_and_text_to_text_nocache(image_bytes, text):

    import PIL.Image
    file = io.BytesIO(image_bytes)
    image = PIL.Image.open(file)

    processor, model = load_model_image_and_text_to_text()
    encoding = processor(image, text, return_tensors="pt")

    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

@lru_cache(maxsize=1)
def load_model_image_and_text_to_text():
    from transformers import ViltProcessor, ViltForQuestionAnswering
    name = "dandelin/vilt-b32-finetuned-vqa"
    processor = ViltProcessor.from_pretrained(name)
    model = ViltForQuestionAnswering.from_pretrained(name)
    return (processor, model)

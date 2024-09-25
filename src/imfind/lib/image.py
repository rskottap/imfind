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
    image = PIL.Image.open(file)

    model = load_model_image_to_text()

    results = model(image)
    outputs = []
    for result in results:
        output = result['generated_text'].strip()
        outputs.append(output)

    if isinstance(file, (list, tuple)):
        return outputs
    elif len(outputs) == 1:
        return outputs[0]
    else:
        raise TypeError(file)


@lru_cache(maxsize=1)
def load_model_image_to_text():
    from transformers import pipeline
    model = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning"
    )
    return model



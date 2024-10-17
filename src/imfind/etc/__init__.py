default_prompt = """
What is shown in this image?
Generate a detailed description of this image.
Include any specific details on background colors, patterns, themes, settings/context (for example if it's a search page results, texting platform screenshot, pic of scenery etc.,), what might be going in in the picture (activities, conversations), what all and how many objects, animals and people are present, their orientations and activities, etc.,
Besides a general description, include any details that might help uniquely identify the image.
"""

file_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'heic']

# See here for full available list https://www.jaided.ai/easyocr/
easyocr_languages = ['en', 'ch_tra']

imfind_use_llava_path = '~/.cache/imfind/use_llava.txt'

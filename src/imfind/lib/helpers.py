#!/usr/bin/env python3

from __future__ import annotations

__all__=['find_all_image_paths', 'describe_images_and_cache', 'image_search', 'load_easyocr']

from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def load_easyocr():
    import easyocr
    from imfind.etc import easyocr_languages
    # this needs to run only once to load the model into memory
    reader = easyocr.Reader(easyocr_languages)
    return reader


def find_all_image_paths(directory: Path, file_types: list[str], include_hidden=False) -> list[Path]:
    """
    Find all image files in the given directory. By default, excludes any hidden directories,
    ones starting with '.' and '__', and their contents.
    """

    def should_include(file: Path) -> bool:
        # Skip hidden directories
        if not include_hidden:
            for parent in file.parents:
                if parent.name.startswith('.') or parent.name.startswith('__'):
                    return False
        return True

    # Find all image files
    image_paths = []
    for file_type in file_types:
        image_paths.extend([file for file in directory.rglob(f'*.{file_type}') if should_include(file)])
    
    return image_paths
    

def describe_images_and_cache(images: list[Path], prompt: str) -> dict[str]:
    """
        images: List of paths to image files
        prompt: prompt used to generate descriptions
        
        Note: in image_and_text_to_text the bytes cached include both the image and text bytes, so if prompt is unchanged then cache can be reused.
    """
    
    import os
    import textwrap
    import torch
    from imfind import image_and_text_to_text, image_to_text
    from collections import defaultdict

    # maps from image abs paths to their descriptions
    descriptions = defaultdict(str)

    # if gpu is available, only then use the bigger LLaVa model. By default, use smaller BLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_llava_success = True

    for img_path in images:
        k = str(img_path)
        try:
            if device != 'cpu' and use_llava_success:
                try:
                    descriptions[k] = image_and_text_to_text(img_path, prompt)
                except Exception as e:
                    msg = textwrap.dedent(f"""\
                    Torch detected gpu, tried to use LLaVa1.5 for image-to-text but inference failed due to the following error:
                    {e}
                    {"#"*42}
                    Using smaller but faster Salesforce/BLIP (image-captioning-large) model instead.\n""")
                    print(msg)

                    use_llava_success = False
                    descriptions[k] = image_to_text(img_path)
            else:
                descriptions[k] = image_to_text(img_path)
              
        except Exception as e:
            descriptions[k] = img_path.name
            print(f"Could not describe image '{k}'. Using file name for description instead.")
            print(e)

    return descriptions


def image_search(user_img_desc: str, gen_desc_prompt: str, directory: Path, file_types: list[str], include_hidden=False, embed_size='large') -> list[str]:

    from embd import Space, EmbedFlag, List
    
    images = find_all_image_paths(directory, file_types, include_hidden)
    descriptions = describe_images_and_cache(images, gen_desc_prompt)
    space = Space(EmbedFlag(embed_size))
    
    user_desc_embed = space.think(user_img_desc)

    # for each image_path:desc pair, embed the descriptions and dot product them with the desired image description embedding to get similarity score for each image
    
    desc_embeds_scores = {path: user_desc_embed @ space.think(desc).T for path, desc in descriptions.items()}
    sorted_path_scores = sorted(desc_embeds_scores.items(), key=lambda item: item[1], reverse=True)
    
    # return img paths (like find command does) sorted by most to least familiar
    return [pair[0] for pair in sorted_path_scores]


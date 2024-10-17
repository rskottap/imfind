#!/usr/bin/env python3

from __future__ import annotations

__all__=['find_all_image_paths', 'describe_images_and_cache', 'image_search', 'easyocr', 'gpu_mem_avail']

import gc
import logging
import os
import textwrap
from functools import lru_cache
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

line_break = '#'*80
llava_error_msg = textwrap.dedent("""
{}
Torch detected gpu, tried to use LLaVa1.5 for image-to-text but inference failed due to the following error:
{}

Using smaller but faster Salesforce/BLIP (image-captioning-large) model instead.
{}
""").strip()

def gpu_mem_avail():

    available_memory = None
    if torch.cuda.is_available():
        available_memory = 0
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            available_memory += total_memory - reserved_memory

    return available_memory


@lru_cache(maxsize=1)
def load_easyocr():
    import easyocr

    from imfind.etc import easyocr_languages

    # this needs to run only once to load the model into memory
    reader = easyocr.Reader(easyocr_languages)
    return reader


def easyocr(image):
    """
    image: bytes, numpy array or str path

    EasyOCR has some limits, so wrap this in a try except.
    - File extension support: png, jpg, tiff.
    - File size limit: 2 Mb.
    - Image dimension limit: 1500 pixel.
    - Possible Language Code Combination: Languages sharing the same written script (e.g. latin) can be used together.
      English can be used with any language.
    """
    text = ""
    try:
        reader = load_easyocr()
        text = ' '.join(reader.readtext(image, detail=0))
        if text: text = 'Text extracted from image:\n' + text
    except Exception as e:
        logging.error(f"Could not run EasyOCR due to the following error:\n{e}")

    return text

@lru_cache(maxsize=1)
def check_image_and_text():
    # In a separate process try to load the LLava model and
    # run single inference to see if model can be used successfully.
    # Tries to use EasyOCR for both models, but doesn't fail if ocr fails for any reason.
    """
    If the model cannot be loaded into memory successfully and raises an exception mid-load,
    i.e., some layers of model are loaded onto GPU but the full model cannot be loaded,
    a CUDA OOM error is thrown but the partially loaded model is not deleted from memory.
    And since an exception is raised before the model can be loaded, we don't have access to
    the model variable to delete it and free up memory. Hence this hack.
    """
    import subprocess

    import imfind

    try:
        # if this file exists, already checked on that device, so return True
        _ = Path(imfind.etc.imfind_use_llava_path).expanduser().resolve(strict=True)
        return True

    except FileNotFoundError as e:
        logging.info(f"File {imfind.etc.imfind_use_llava_path} not found.\n\
                     Running test script to determine if llava can be loaded and used successfully.")

        test_path = os.path.join(imfind.__path__[0], 'lib/test_load_llava.py')
        try:
            _ = subprocess.run(
                ["python3", test_path],
                capture_output=True,
                check=True,
            )

            file_path = Path(imfind.etc.imfind_use_llava_path).expanduser()
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write("True")
            logging.info("Was able to successfully load and use LLaVa model.\n")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(llava_error_msg.format(line_break, e.stderr, line_break))

    torch.cuda.empty_cache()
    _ = gc.collect()
    return False


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
    from collections import defaultdict

    from imfind import image_and_text_to_text, image_to_text

    # maps from image abs paths to their descriptions
    descriptions = defaultdict(str)
    # if gpu is available, only then use the bigger LLaVa model. By default, use smaller BLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    use_llava_success = False
    if device != 'cpu': use_llava_success = check_image_and_text()

    for img_path in images:
        k = str(img_path)
        try:
            if device != 'cpu' and use_llava_success:
                try:
                    descriptions[k] = image_and_text_to_text(img_path, prompt)
                except Exception as e:
                    logging.error(llava_error_msg.format(line_break, e, line_break))
                    use_llava_success = False
                    torch.cuda.empty_cache()
                    _ = gc.collect()

                    descriptions[k] = image_to_text(img_path)
            else:
                descriptions[k] = image_to_text(img_path)
              
        except Exception as e:
            descriptions[k] = img_path.name
            logging.warning(f"Could not describe image '{k}' due to the following error:\n{e}\nUsing file name for description instead.\n")

    return descriptions


def image_search(user_img_desc: str, gen_desc_prompt: str, directory: Path, file_types: list[str], include_hidden=False, embed_size='large') -> list[str]:

    from embd import EmbedFlag, List, Space
    
    images = find_all_image_paths(directory, file_types, include_hidden)
    descriptions = describe_images_and_cache(images, gen_desc_prompt)
    space = Space(EmbedFlag(embed_size))
    
    user_desc_embed = space.think(user_img_desc)

    # for each image_path:desc pair, embed the descriptions and dot product them with the desired image description embedding to get similarity score for each image
    
    desc_embeds_scores = {path: user_desc_embed @ space.think(desc).T for path, desc in descriptions.items()}
    sorted_path_scores = sorted(desc_embeds_scores.items(), key=lambda item: item[1], reverse=True)
    
    # return img paths (like find command does) sorted by most to least familiar
    return [pair[0] for pair in sorted_path_scores]


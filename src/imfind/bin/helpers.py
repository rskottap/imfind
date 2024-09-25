#!/usr/bin/env python3

### Set of helper functions to generate descriptions of images, embed and cache them. Embed user description and similarity with image embedding caches.

__all__=['find_all_image_paths']


from __future__ import annotations
import subprocess

# Find all image files in given directory. By default excludes any hidden or cache directories, ones starting with '.' and '__' (for ex, .cache, .config, __pycache__ etc.,)
"""
TODO: Fix to work without shell=True

Do not use any special shell syntax like \(\) or ! to allow more cross-platform compatability (sorry Windows!)
    
find /home/ramya -type d -iname ".*" -prune -iname "__*" -prune -type f -iname "*.png" -o -iname "*.jpg" | wc -l
"""
def find_all_image_paths(directory: str, file_types: list[str], include_hidden=False) -> list[str]:
    
    exclude_hidden_command = ' ! -path "*/.*" ! -path "*/__*" '

    find_command = f'find {directory} -type f \\( {' -o '.join([f'-iname "*.{t}"' for t in file_types])} \\)' + (exclude_hidden_command if not include_hidden else '')
    
    output = subprocess.run(find_command, shell=True, capture_output=True, text=True)
    
    return output.stdout.splitlines()
    

def describe_images_and_cache(images, prompt) -> dict[str]:
    """
        images: List of paths to image files
        prompt: prompt used to generate descriptions
        
        Note: in image_and_text_to_text the bytes cached include both the image and text bytes, so if prompt is unchanged then cache can be reused.
    """
    
    from imfind import image_and_text_to_text
    from collections import defaultdict

    # maps from image abs paths to their descriptions
    descriptions = defaultdict(str) 
    return descriptions


def image_search(prompt, directory, file_types, include_hidden=False, embed_size='large') -> list[str]:

    from embd import Space, EmbedFlag, List
    
    images = find_all_image_paths(directory, file_types, include_hidden)
    descriptions = describe_images_and_cache(images, prompt)
    space = Space(EmbedFlag(embed_size))
    
    prompt_embed = space.think(prompt)

    # for each image_path:desc pair, embed the descriptions and dot product them with the prompt embedding to get similarity score for each image
    desc_embeds_scores = {path: prompt_embed @ space.think(desc).T for path, desc in descriptions.items()}

    sorted_path_scores = dict(sorted(desc_embeds_scores.items(), key=lambda item: item[1]))
    return sorted_path_scores.keys()
 

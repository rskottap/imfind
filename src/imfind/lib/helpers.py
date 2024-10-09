#!/usr/bin/env python3

from __future__ import annotations

__all__=['find_all_image_paths', 'describe_images_and_cache', 'image_search']


def find_all_image_paths(directory: str, file_types: list[str], include_hidden=False) -> list[str]:    
    """
    Find all image files in given directory. By default excludes any hidden or cache directories, ones starting with '.' and '__' (for ex, .cache, .config, __pycache__ etc.,)
    
    TODO: Fix to work without shell=True
    
    Do not use any special shell syntax like \(\) or ! to allow more cross-platform compatability (sorry Windows!)
        
    find /home/ramya -type d -iname ".*" -prune -iname "__*" -prune -type f -iname "*.png" -o -iname "*.jpg" | wc -l
    """
        
    import subprocess
    
    exclude_hidden_command = ' ! -path "*/.*" ! -path "*/__*" '

    find_command = f'find -L "{directory}" -type f \\( {' -o '.join([f'-iname "*.{t}"' for t in file_types])} \\)' + (exclude_hidden_command if not include_hidden else '')
    
    output = subprocess.run(find_command, shell=True, capture_output=True, text=True)
    return output.stdout.splitlines()
    

def describe_images_and_cache(images: list[str], prompt: str) -> dict[str]:
    """
        images: List of paths to image files
        prompt: prompt used to generate descriptions
        
        Note: in image_and_text_to_text the bytes cached include both the image and text bytes, so if prompt is unchanged then cache can be reused.
    """
    
    import os
    import torch
    from imfind import image_and_text_to_text, image_to_text
    from collections import defaultdict

    # maps from image abs paths to their descriptions
    descriptions = defaultdict(str)

    # if gpu is available, only then use the bigger LLaVa model. By default, use smaller BLIP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for img_path in images:
        try:
            if device != 'cpu':
                descriptions[img_path] = image_and_text_to_text(img_path, prompt)
            else:
                descriptions[img_path] = image_to_text(img_path)
              
        except Exception as e:
            descriptions[img_path] = os.path.basename(img_path)
            print(f"Could not describe image '{img_path}'. Using file name for description instead.")
            print(e)

    return descriptions


def image_search(user_img_desc: str, gen_desc_prompt: str, directory: str, file_types: list[str], include_hidden=False, embed_size='large') -> list[str]:

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


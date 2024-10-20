#!/usr/bin/env python3

__all__ = ['main']

import argparse
import os
import sys
import platform
from pathlib import Path

from imfind import etc


def parse_args(argv):

    def check_nonneg(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"Threshold needs to be an integer greater than or equal to 0 (int n>=0). Passed {value}\n")
        return ivalue

    parser = argparse.ArgumentParser('imfind')
    parser.add_argument('description', type=str, help="Description of the image to find.")
    parser.add_argument('-d', '--directory', type=str, help="Directory to search in. Defaults to $HOME.")
    parser.add_argument('-n', '--threshold', type=check_nonneg, default=10, help="Top n (>=0) results to output. By default lists the top 10 images from most similar/relevant to least. Pass n=0 to list ALL the images (from most to least similar).")
    parser.add_argument('-p', '--prompt', type=str, help="Additional user prompt that gets appended to default prompt. Used to generate description of images.")
    parser.add_argument('-t', '--types', type=str, nargs='*', default=etc.file_types, help=f"Additional image types to search. By default searches for images with  extensions {etc.file_types}")
    parser.add_argument('--no-cache', action='store_true', help="Do not read from existing cache. Overwrites cache with new model generations.")
    parser.add_argument('--include-hidden', action='store_true', help="Include hidden directories like ones starting with '.' or '__'. For example, within .cache or __pycache__ etc., Excludes these by default.")
    args = parser.parse_args(argv)
    return args

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    description = args.description.strip()
    if not description:
        raise Exception("Error: Empty description. Please provided a detailed description of the image to find.")

    directory = ''
    try:
        if not args.directory:
            HOME = "HOME"
            if platform.system() == "Windows": HOME = "USERPROFILE"
            directory = Path(os.environ[HOME]).expanduser().resolve(strict=True) # make it home directory
        else:
            directory = Path(args.directory.strip()).expanduser().resolve(strict=True)
    except Exception as e:
        raise e
    
    file_types = etc.file_types + (args.types or [])
    file_types += [e.capitalize() for e in file_types]
    file_types = list(set(file_types))

    thres = args.threshold
    final_prompt = etc.default_prompt + (args.prompt or '').strip()

    use_cache = False if args.no_cache else True
    include_hidden = bool(args.include_hidden)

    # do the image search
    from imfind import image_search

    top = image_search(user_img_desc=description, gen_desc_prompt=final_prompt, 
                       directory=directory, file_types=file_types, use_cache=use_cache, include_hidden=include_hidden)

    if thres==0:
        print('\n'.join(top))
    else:
        print('\n'.join(top[:thres]))

if __name__ == '__main__':
    main(sys.argv[1:])

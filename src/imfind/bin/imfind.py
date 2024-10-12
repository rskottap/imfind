#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import argparse
from pathlib import Path
from . import config

def parse_args(argv):

    def check_nonneg(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"Threshold needs to be an integer greater than or equal to 1 (n>=1).\n Passed {value}.")
        return ivalue

    parser = argparse.ArgumentParser('imfind')
    parser.add_argument('description', type=str, help="Description of the image to find.")
    parser.add_argument('-d', '--directory', type=str, help="Directory to search in. Defaults to $HOME.")
    parser.add_argument('-n', '--threshold', type=check_nonneg, default=0, help="Top n (>=1) results to output. By default lists all images found (like the find command) sorted from most relevant/similar to least.")
    parser.add_argument('-p', '--prompt', type=str, help="Additional user prompt that gets appended to default prompt. Used to generate description of images.")
    parser.add_argument('-t', '--types', type=str, nargs='*', default=config.file_types, help=f"Additional image types to search. By default searches for images with  extensions {config.file_types}")
    parser.add_argument('--no-cache', action='store_true', help="Do not read from existing cache. Overwrites cache with new model generations. (Internally sets and uses USE_MMRY_CACHE environment variable).")
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
            directory = Path(os.environ["HOME"]).expanduser().resolve() # make it home directory 
        else:
            directory = Path(args.directory.strip()).expanduser().resolve()
            if not directory.exists():
                raise FileNotFoundError(f"{directory}: No such file or folder exists.")
    except Exception as e:
        print("Make sure $HOME is set (export HOME='/home/<user>') or provide an existing directory to search within.")
        raise e
    
    file_types = config.file_types + (args.types or [])
    file_types += [e.capitalize() for e in file_types]
    file_types = list(set(file_types))

    thres = args.threshold
    final_prompt = config.default_prompt + (args.prompt or '').strip()

    os.environ["USE_MMRY_CACHE"] = "False" if args.no_cache else "True"
    include_hidden = bool(args.include_hidden)

    # do the image search
    from imfind import image_search

    top = image_search(user_img_desc=description, gen_desc_prompt=final_prompt, 
                       directory=directory, file_types=file_types, include_hidden=include_hidden)

    # delete variable
    __ = os.environ.pop("USE_MMRY_CACHE")

    if thres:
        print('\n'.join(top[:thres]))
    else:
        print('\n'.join(top))

if __name__ == '__main__':
    main(sys.argv[1:])

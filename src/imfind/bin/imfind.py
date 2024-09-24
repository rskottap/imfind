#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import subprocess
import argparse
import json
from .config import default_prompt, file_types

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser('Find image')
    parser.add_argument('description', type=str, help="Description of the image to find.")
    parser.add_argument('-d', '--directory', type=str, help="Directory to search in. Defaults to $HOME.")
    parser.add_argument('-n', '--threshold', type=int, default=10, help="Top n results to output. Defaults to top 10. Use -1 to list all images sorted by relevance.")
    parser.add_argument('-p', '--prompt', type=str, help="Additional user prompt that gets appended to default prompt. Used to generate description of images.")
    parser.add_argument('-t', '--types', type=str, nargs='*', default=file_types, help=f"Additional image types to search. By default searches for images with  extensions {file_types}")
    parser.add_argument('--include-hidden', action='store_true', help="Include hidden directories like ones starting with '.' or '__'. For example, within .cache or __pycache__ etc., Excludes these by default.")
    args = parser.parse_args(argv)
    
    description = args.description.strip()
    if not description:
        raise Exception("Error: Empty description. Please provided a detailed description of the image to find.")

    directory = ''
    try:
        if not args.directory:
            directory = os.environ["HOME"] # make home directory 
        else:
            directory = os.path.expanduser(args.directory.strip())
            if not os.path.exists(directory):
                raise FileNotFoundError(f"{directory}: No such file or folder exists.")
    except Exception as e:
        print("Make sure $HOME is set or provide an existing directory to search within.")
        raise e
    
    im_types = list(set(file_types + (args.types or [])))
    thres = args.threshold
    final_prompt = default_prompt + (args.prompt or '').strip()
    include_hidden = bool(args.include_hidden)
    
    d = {"description": description, "directory": directory, "im_types": im_types, "n": thres, "final_prompt": final_prompt, "include_hidden": include_hidden}

    print("Here are the final arguments:")
    print(json.dumps(d, indent=4))
    print("-"*42)

    #### Find all image files in given directory
    # Do not use any special shell syntax like \(\) or ! to allow more cross-platform compatability (sorry Windows!)
    
    # find /home/ramya -type d -iname ".*" -prune -iname "__*" -prune -type f -iname "*.png" -o -iname "*.jpg" | wc -l
    #TODO: Fix to work without shell=True

   # find_command = ['find', directory]

   # if not include_hidden:
   #     base_find_command += ['-type', 'd', '-iname', '".*"', '-prune', '-iname', '"__*"', '-prune']
   # 
   # base_find_command += ['-type', 'f']

   # for t in file_types:
   #     base_find_command = base_find_command + ['-iname', f'"*.{t}"', '-o']
   # base_find_command.pop()

   # print(" ".join(base_find_command))
    exclude_hidden_command = ' ! -path "*/.*" ! -path "*/__*" '
    find_command = f"find {directory} -type f {'-o'.join(['-iname "*.{t}"' for t in file_types])}" + (exclude_hidden_command if not include_hidden else '')
    output = subprocess.run(find_command, shell=True, capture_output=True, text=True)
    
    __import__("IPython").embed()    
    print(output.stdout)
    
if __name__ == '__main__':
    main(sys.argv[1:])

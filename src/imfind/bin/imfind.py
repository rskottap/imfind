#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import argparse
import json

default_prompt = """Generate a detailed description of this image. Include an overall description that can help identify and distinguish the image amongst many other similar or different images.
Include any specific details on background colors, patterns, themes, settings/context (for example if it's a search page results, texting platform screenshot, pic of scenery etc.,), what might be going in in the picture (activities, conversations), what all objects/animals/people are present, their orientations, any proper nouns mentioned, dates etc., 
Besides a general description, include any details that might help uniquely identify the image.
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser('Find image')
    parser.add_argument('description', type=str, help="Description of the image to find.")
    parser.add_argument('-d', '--directory', type=str, help="Directory to search in. Defaults to $HOME.")
    parser.add_argument('-n', '--threshold', type=int, default=10, help="Top n results to output. Defaults to top 10. Use -1 to list all images sorted by relevance.")
    parser.add_argument('-p', '--prompt', type=str, help="Additional user prompt that gets appended to default prompt. Used to generate description of images.")
    
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

    thres = args.threshold
    final_prompt = default_prompt + (args.prompt or '').strip()
    
    d = {"description": description, "directory": directory, "n": thres, "final_prompt": final_prompt}

    print("Here are the final arguments:")
    print(json.dumps(d, indent=4))

if __name__ == '__main__':
    main(sys.argv[1:])

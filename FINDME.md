## ImFind 🖼️ 

Remember what an image, pic or screenshot looks like but can't for the life of you remember where you put it? 

Or what you named it? 

Or there's just too many damn screenshots to go through all of them? 

<img src="include/imfind-comic.png" alt="Comic" width="60%"/> 

Been there. Too many times.

So here's a simple tool to help with that.


---

### Usage

- See in `tests` for examples.
TODO: Insert a screenshot of example here. 

- Do `imfind --help` to see usage.

#### Demo

TODO: Make demo video

---

### How it works

Given a description of an image you want to find, this library: 


1. Embeds the given user description using an embedding model. 


2. Finds all images in the given directory and (if not already previously done (via caching)) generates a description of the each of the images and embeds them. Uses an image-to-text like model with a default prompt to generate a decription of the image, optionally adding a user custom prompt if provided. 


3. Finds the most similar images based on similarity between the user description and the generated image description embeddings (nothing fancy, just your simple dot products). 

---

#### Models used 

- Embedding model:
- image-to-text model:

---
#### Notes

- Be as descriptive and detailed as you can when providing a description of the image you want to search for.

- The first time you run the command on a new directory it will take some time to download the models, generate the descriptions of all the images and embed and cache them. Any subsequent runs should be quick thanks to the awesome cacheing implementation in [embd](https://github.com/notarealdeveloper/embd) using [mmry](https://github.com/notarealdeveloper/mmry). 

- If you want to clear the cache of all the text and image embeddings (for re-runs, new models, debugging etc.,) simply do `rm -rf ~/.cache/mmry`.
 
- If you remember part of what you named the file, or some other unique file name identifier, don't forget your simple (but awesome) unix tools like `locate`. 

```
locate comic
> include/imfind-comic.png
```


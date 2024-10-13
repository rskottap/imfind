from pathlib import Path
import assure
from imfind import image_to_text, image_and_text_to_text, load_easyocr, image_search
from imfind import etc

def test_image_search():
    # CPU/GPU usage determined within
    path = Path("./include/examples/").expanduser().resolve()
    user_description = "raccooon and cars" # embedding model can handle typo's too
    img_paths = image_search(user_description, etc.default_prompt, path, etc.file_types)
    top_two_paths = ' '.join(img_paths[:2])
    assert "racoon" in top_two_paths
    assert "misc-3" in top_two_paths


def test_image_to_text():
    text = image_to_text("./include/examples/animal-pics/puppy-pic-1.jpeg")
    text2 = image_to_text("./include/examples/misc/food-1.png")
    assert "puppies" in text
    assert "pancakes" in text2


def example_image_and_text_to_text():
    """ Example usage of LLaVa model which also takes additional generation prompt as input.
        Only recommended to run on GPU, extremely slow on CPU due to large model """
    
    text = image_and_text_to_text("./include/examples/animal-pics/puppy-pic-1.jpeg", etc.default_prompt)
    print(text)

def test_easyocr():
    reader = load_easyocr()
    
    # read with image bytes
    image1 = Path("./include/examples/texts/text-1.jpeg").expanduser().resolve()
    image_bytes1 = assure.bytes(image1)
    ocr_text1 = ' '.join(reader.readtext(image_bytes1, detail=0))

    # read with image path as str
    image2 = "./include/examples/texts/text-2.jpeg"
    ocr_text2 = ' '.join(reader.readtext(image2, detail=0))
    
    assert "Privet Drive" in ocr_text1
    assert "Half Blood Prince" in ocr_text2


import os
import argparse
import shutil
import requests

from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor, BlipForConditionalGeneration

import os
import shutil
import csv

import re

parser = argparse.ArgumentParser(description="Dataset script")

parser.add_argument("--path", "-p", type=str,
                    help='Set path to your data set')

args = parser.parse_args()


def rename_files(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Sort the files to maintain order
    files.sort()

    # Rename files sequentially
    for i, filename in enumerate(files):
        # Create a new name with leading zeros
        new_name = f"{i + 1:02}.jpg"  # Change extension if needed

        # Full paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {filename} to {new_name}")
    return True



def images_train_test_split(dir_path):

    # Specify the directory containing the files
    if not rename_files(dir_path):
        print(f"File at {dir_path} are rename properly")

    # Directories for split datasets
    train_dir = os.path.join(dir_path, "train")
    test_dir = os.path.join(dir_path, "test")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all images
    all_images = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    # Split into train and test sets
    try:
        train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"{e} raise, The image folder is empty.")
    # Move images to respective directories

    for img in train_images:
        shutil.move(os.path.join(dir_path, img), os.path.join(train_dir, img))

    for img in test_images:
        shutil.move(os.path.join(dir_path, img), os.path.join(test_dir, img))

    print(f"Train path: {train_dir} \n Training images: {len(train_images)}")
    print("\n------\n")
    print(f"Test path: {test_dir} \n Testing images: {len(test_images)}")

    return True



def caption_model():
    # Load the model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# function to generate captions for images

def genCaption(imagePath, processor, model):
    # Load an image
    image = Image.open(imagePath)

    # Process the image and generate a caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return(caption)



def annotates_images(path, sub_df):
    # Annotates images

    imagesTrainPath = os.path.join(path, "train")
    imagesTestPath = os.path.join(path, "test")

    trainMeta = [["file_name", "label"]]
    testMeta = [["fil_ename", "label"]]

    imagesTrain = os.listdir(imagesTrainPath)

    pattern1 = r'image_\d+\.png'
    pattern2 = r'image_(\d+)'

    regex1 = re.compile(pattern1)
    regex2 = re.compile(pattern2)

    for image in imagesTrain:
        path = os.path.join(imagesTrainPath, image)
        if ".ipynb_checkpoints" == image:
            try:
                shutil.rmtree(path)
                continue
            except:
                pass
        elif "metadata.csv" == image:
            continue
        elif regex1.match(image):
            idx = int(regex2.search(image).group(1))
            trainMeta.append([image, sub_df['prompt'][idx]])
            continue
        caption = genCaption(path)
        trainMeta.append([image, caption])
        with open(os.path.join(imagesTrainPath, "metadata.csv"), "w") as file:
            writer = csv.writer(file)
            writer.writerows(trainMeta)


    imagesTest = os.listdir(imagesTestPath)

    for image in imagesTest:
        path = os.path.join(imagesTestPath, image)
        if ".ipynb_checkpoints" == image:
            try:
                shutil.rmtree(path)
                continue
            except:
                pass
        elif "metadata.csv" == image:
            continue
        elif regex1.match(image):
            idx = int(regex2.search(image).group(1))
            testMeta.append([image, sub_df['prompt'][idx]])
            continue
        caption = genCaption(path)
        testMeta.append([image, caption])
        with open(os.path.join(imagesTestPath, "metadata.csv"), "w") as file:
            writer = csv.writer(file)
            writer.writerows(testMeta)


if __name__ == "__main__": 
    images_train_test_split(args.path)








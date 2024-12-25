import os
import cv2
import numpy as np


# Define a function to preprocess an image
def preprocess_image(image_path, target_size = (224, 224), mean = 0.5, std = 0.5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert to grayscale if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    image = cv2.resize(image, target_size)

    # Normalize and standardize the image in one step
    image = (image / 255.0 - mean) / std

    return image


# Define a function to load images from a directory
def load_images_from_directory(directory, target_size = (224, 224), mean = 0.5, std = 0.5):
    data = []
    labels = []

    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    processed_image = preprocess_image(image_path, target_size, mean, std)
                    data.append(processed_image)
                    labels.append(person_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    return np.array(data), np.array(labels)


# Set the dataset directory and load images
dataset_directory = 'face/lfw-deepfunneled/lfw-deepfunneled'
data, labels = load_images_from_directory(dataset_directory)

print(f"Loaded {len(data)} images with {len(set(labels))} unique labels.")

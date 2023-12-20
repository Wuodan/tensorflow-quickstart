# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf
from PIL import Image

import glob
import re

# Recognize a digit from a variable-sized input image
def recognize_digit(image_path, model):
    # Load and preprocess the input image
    input_image = Image.open(image_path)

    input_image_array = tf.keras.preprocessing.image.img_to_array(input_image)
    input_image_array = input_image_array / 255.0  # Normalize to [0, 1]

    # Make predictions using the pre-trained model
    predictions = model.predict(tf.expand_dims(input_image_array, axis=0), verbose=0)

    # Get the recognized digit
    recognized_digit = tf.argmax(predictions, axis=1).numpy()[0]
    return recognized_digit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize a digit from a variable-sized input image.")
    parser.add_argument("--mnist_folder", type=str, required=True, help="Path to the MNIST dataset folder")
    args = parser.parse_args()

    # Load the pre-trained models
    pattern = re.compile(r'trained_model_(.+)\.h5')
    for file_name in glob.glob('trained_model_*.h5'):
        model_name = pattern.search(file_name).group(1)
        model = tf.keras.models.load_model(file_name)

        # Loop over all images in the MNIST dataset folder
        for image_file in glob.glob(os.path.join(args.mnist_folder, '*.png')):
            # Perform digit recognition
            recognized_digit = recognize_digit(image_file, model)

            # Print the recognized digit and image file name
            print(f"Recognized digit: {recognized_digit}, model-name: {model_name}, image: {image_file}")

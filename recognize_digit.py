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
    input_image = input_image.convert('L')  # Convert to grayscale

    # Resize input image while preserving the original aspect ratio
    aspect_ratio = 28 / max(input_image.width, input_image.height)
    target_size = (
        int(input_image.width * aspect_ratio),
        int(input_image.height * aspect_ratio)
    )
    input_image_resized = input_image.resize(target_size, Image.LANCZOS)

    input_image_resized = tf.keras.preprocessing.image.img_to_array(input_image_resized)
    input_image_resized = input_image_resized / 255.0  # Normalize to [0, 1]

    # Pad the resized image to 28x28
    pad_width = (28 - target_size[0]) // 2, (28 - target_size[1]) // 2
    input_image_resized_padded = tf.pad(input_image_resized, [[pad_width[1], pad_width[1]], [pad_width[0], pad_width[0]], [0, 0]])

    # Resize to (28, 28)
    input_image_resized_padded = tf.image.resize(input_image_resized_padded, (28, 28))

    # Make predictions using the pre-trained model
    predictions = model.predict(tf.expand_dims(input_image_resized_padded, axis=0), verbose=0)

    # Get the recognized digit
    recognized_digit = tf.argmax(predictions, axis=1).numpy()[0]
    return recognized_digit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize a digit from a variable-sized input image.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Load the pre-trained models
    pattern = re.compile(r'trained_model_(.+)\.h5')
    for file_name in glob.glob('trained_model_*.h5'):

        model_name = pattern.search(file_name).group(1)

        model = tf.keras.models.load_model(file_name)

        # Perform digit recognition
        recognized_digit = recognize_digit(args.input_image, model)

        # Print the recognized digit
        print(f"Recognized digit: {recognized_digit}, model-name: {model_name}")

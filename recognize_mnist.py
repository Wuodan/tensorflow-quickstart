# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf
from PIL import Image
import numpy as np

import glob
import re

# Recognize a digit from a variable-sized input image
def recognize_digit(image, model):
    # Load and preprocess the input image
    input_image_array = image / 255.0  # Normalize to [0, 1]

    # Make predictions using the pre-trained model
    predictions = model.predict(tf.expand_dims(input_image_array, axis=0), verbose=0)

    # Get the recognized digit
    recognized_digit = tf.argmax(predictions, axis=1).numpy()[0]
    return recognized_digit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize a digit from a variable-sized input image.")
    parser.add_argument("--mnist_file", type=str, required=True, help="Path to the MNIST dataset file (mnist.npz)")
    args = parser.parse_args()

    # Load the pre-trained models
    pattern = re.compile(r'trained_model_(.+)\.h5')
    for file_name in glob.glob('trained_model_*.h5'):
        model_name = pattern.search(file_name).group(1)
        model = tf.keras.models.load_model(file_name)

        # Load MNIST dataset
        mnist_data = np.load(args.mnist_file)
        x_test, y_test = mnist_data['x_test'], mnist_data['y_test']

        # Loop over all images in the MNIST dataset
        for i in range(len(x_test)):
            image = x_test[i]

            # Perform digit recognition
            recognized_digit = recognize_digit(image, model)

            # Print the recognized digit and true label
            true_label = y_test[i]
            print(f"Recognized digit: {recognized_digit}, True label: {true_label}, model-name: {model_name}")

# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf
import numpy as np

import glob
import re

# Recognize a digit from a variable-sized input image
@tf.function
def recognize_digit(image, model):
    # Load and preprocess the input image
    input_image_array = tf.cast(image, tf.float32) / 255.0  # Explicitly cast to tf.float32 and normalize to [0, 1]

    # Make predictions using the pre-trained model
    predictions = model(tf.expand_dims(input_image_array, axis=0), training=False)

    # Get the recognized digit
    recognized_digit = tf.argmax(predictions, axis=1)[0]
    return recognized_digit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize a digit from a variable-sized input image.")
    parser.add_argument("--mnist_file", type=str, required=True, help="Path to the MNIST dataset file (mnist.npz)")
    args = parser.parse_args()

    # Load the pre-trained models
    pattern = re.compile(r'trained_model_(.+)\.h5')
    models_data = {}

    for file_name in glob.glob('trained_model_*.h5'):
        model_name = pattern.search(file_name).group(1)
        model = tf.keras.models.load_model(file_name)
        models_data[model_name] = model

    # Load MNIST dataset
    mnist_data = np.load(args.mnist_file)
    x_test, y_test = mnist_data['x_test'], mnist_data['y_test']

    # Calculate the width needed for padding based on the total number of files
    total_files = len(x_test)
    padding_width = len(str(total_files))

    # Create a dictionary to count correct predictions for each model and label
    correct_predictions_count = {model_name: {label: 0 for label in range(10)} for model_name in models_data.keys()}

    # Loop over all images in the MNIST dataset
    for i in range(total_files):
        image = x_test[i]

        for model_name, model in models_data.items():
            # Perform digit recognition
            recognized_digit = recognize_digit(image, model)

            # Check if the recognized digit is correct
            if recognized_digit == y_test[i]:
                correct_predictions_count[model_name][y_test[i]] += 1

            # Pad the index with zeros to ensure a consistent string length
            padded_index = str(i).zfill(padding_width)

            # Print the MNIST index, true label, recognized digit, model name
            print(f"MNIST index: {padded_index}, True label: {y_test[i]}, Recognized digit: {recognized_digit}, model-name: {model_name}")

    # Output total images and recognition ratio per model and label
    print("\nTotal Images:", total_files)
    for model_name, model in models_data.items():
        print(f"\nRecognition ratio for model {model_name}:")
        for label in range(10):
            total_label_images = np.sum(y_test == label)
            correct_label_predictions = correct_predictions_count[model_name][label]
            recognition_ratio = correct_label_predictions / total_label_images if total_label_images > 0 else 0
            print(f"Label {label}: {recognition_ratio:.4f} ({correct_label_predictions}/{total_label_images} correct)")

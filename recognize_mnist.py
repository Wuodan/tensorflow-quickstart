# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize to [0, 1]

# Load the pre-trained models
model_basic = load_model('trained_model_basic.h5')
model_improved = load_model('trained_model_improved.h5')

# Initialize counters
total_images = np.zeros(10)
correct_basic = np.zeros(10)
correct_improved = np.zeros(10)

# Loop over the entire MNIST dataset
for i in range(len(x_test)):
    label = y_test[i]
    total_images[label] += 1

    # Preprocess the input image
    input_image = x_test[i]
    input_image = tf.image.resize(input_image, (28, 28))
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.expand_dims(input_image, axis=-1)

    # Make predictions using the pre-trained models
    predictions_basic = model_basic.predict(input_image, verbose=0)
    predictions_improved = model_improved.predict(input_image, verbose=0)

    # Get the recognized digit
    recognized_digit_basic = np.argmax(predictions_basic)
    recognized_digit_improved = np.argmax(predictions_improved)

    # Check correctness and update counters
    if recognized_digit_basic == label:
        correct_basic[label] += 1

    if recognized_digit_improved == label:
        correct_improved[label] += 1

# Calculate recognition rates
recognition_rate_basic = correct_basic / total_images
recognition_rate_improved = correct_improved / total_images

# Print results
print("Label\tTotal Images\tRecognition Rate (Basic)\tRecognition Rate (Improved)")
for label in range(10):
    print(f"{label}\t{total_images[label]}\t\t{recognition_rate_basic[label]:.4f}\t\t\t{recognition_rate_improved[label]:.4f}")

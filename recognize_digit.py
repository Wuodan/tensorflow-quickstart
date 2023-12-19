# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf
from PIL import Image

# Recognize a digit from a variable-sized input image
def recognize_digit(image_path, model):
    # Load and preprocess the input image
    input_image = Image.open(image_path)
    input_image = input_image.convert('L')  # Convert to grayscale
    input_size = (input_image.width, input_image.height)
    
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


    # Generate the output filename in the same folder as the input image
    input_filename, input_extension = os.path.splitext(os.path.basename(image_path))
    output_filename = f"{input_filename}-final.jpg"
    output_path = os.path.join(os.path.dirname(image_path), output_filename)

    # Save the final image
    Image.fromarray((input_image_resized_padded.numpy() * 255).astype('uint8').squeeze()).save(output_path)




    # Make predictions using the pre-trained model
    predictions = model.predict(tf.expand_dims(input_image_resized_padded, axis=0), verbose=0)
    # same with progress bar
    # predictions = model.predict(tf.expand_dims(input_image_resized_padded, axis=0))

    # Get the recognized digit
    recognized_digit = tf.argmax(predictions, axis=1).numpy()[0]
    return recognized_digit, input_size, output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize a digit from a variable-sized input image.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Load the pre-trained model
    model = tf.keras.models.load_model('trained_model.h5')

    # Perform digit recognition
    recognized_digit, input_size, output_filename = recognize_digit(args.input_image, model)

    # Print the recognized digit, input image size, and the final image path
    print(f"Recognized digit: {recognized_digit}, Input image size: {input_size}")


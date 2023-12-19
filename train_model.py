import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Load and preprocess the MNIST dataset
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

# Define the improved model architecture for variable-sized rectangular images
def create_improved_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    return model

if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    x_train, y_train, _, _ = load_and_preprocess_mnist()

    # Set a fixed size for the last dimension of the input shape
    last_dimension_size = 1  # 1 for grayscale, adjust based on your images

    # Create the improved model with dynamic input shape
    improved_model = create_improved_model((28, 28, last_dimension_size))

    # Compile the improved model
    improved_model.compile(optimizer=Adam(),
                           loss=SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    # Train the improved model on the MNIST dataset
    improved_model.fit(x_train, y_train, epochs=5)

    # Save the trained improved model
    improved_model.save('improved_model.h5')

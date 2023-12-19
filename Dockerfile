# Stage 1: Build and train the model
FROM tensorflow/tensorflow:latest AS builder

# Set the working directory in the builder stage
WORKDIR /app

# Copy the Python scripts into the builder stage
COPY train_model.py /app/train_model.py

# Download the MNIST dataset
RUN python -c "from tensorflow.keras.datasets import mnist; mnist.load_data()"

# Run the script to train the model
RUN python train_model.py

# Stage 2: Final runtime stage
FROM tensorflow/tensorflow:latest

# Install Pillow
RUN pip install Pillow

# Set the working directory in the final stage
WORKDIR /app

COPY recognize_digit.py /app/recognize_digit.py

# Copy only the necessary files from the builder stage
COPY --from=builder /app/trained_model.h5 /app/trained_model.h5

# Specify the command to run the Python script
CMD ["python", "recognize_digit.py"]

#!/usr/bin/env bash

docker build -f Dockerfile_full_mnist -t tensorflow_digit_recognition__full_mnist .
winpty docker run -it \
	-v "/$(pwd -W)/error_files":/app/error_files \
	tensorflow_digit_recognition__full_mnist
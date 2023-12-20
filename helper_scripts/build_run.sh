#!/usr/bin/env bash

docker build -f Dockerfile_single_image -t tensorflow_digit_recognition__single_image .
winpty docker run -it \
	-v "/$(pwd -W)/sample_images":/app/sample_images \
	tensorflow_digit_recognition__single_image \
	python recognize_digit.py --input_image sample_images/000_4_whiteBG.png
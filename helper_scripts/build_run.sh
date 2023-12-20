#!/usr/bin/env bash

docker build -t tensorflow_digit_recognition .
winpty docker run -it \
	-v "/$(pwd -W)/sample_images":/app/sample_images \
	tensorflow_digit_recognition \
	python recognize_digit.py --input_image sample_images/000_4_whiteBG.png
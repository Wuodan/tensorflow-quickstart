#!/usr/bin/env bash

for image in $(ls sample_images | grep -v '\-final\.'); do
	echo $image;
	winpty docker run -it \
		-v "/$(pwd -W)/sample_images":/app/sample_images \
		tensorflow_digit_recognition \
		python recognize_digit.py --input_image sample_images/$image
	printf '\n'
done

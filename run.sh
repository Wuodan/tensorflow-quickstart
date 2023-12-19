#!/usr/bin/env bash

for image in $(ls sample_images | grep -v '\-final\.'); do
	echo $image;
	printf 'base recognition: '
	winpty docker run -it -v "//c/development/tensorflow/quickstart/sample_images:/app/sample_images" tensorflow_digit_recognition python recognize_digit.py --input_image sample_images/$image;
	printf 'improved recognition: '
	winpty docker run -it -v "//c/development/tensorflow/quickstart/sample_images:/app/sample_images" tensorflow_digit_recognition_improved python recognize_digit.py --input_image sample_images/$image;
	printf '\n'
done

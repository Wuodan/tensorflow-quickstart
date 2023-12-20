#!/usr/bin/env bash

printf "These are my current scripts:\n\n"
for file_name in Dockerfile train_model.py recognize_digit.py; do
	printf "$file_name:\n\`\`\`"
	cat $file_name
	printf "\`\`\`\n\n"
done
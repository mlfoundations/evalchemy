#!/bin/bash

# Base URL
base_url="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation."​
# Number of files
num_files=9

# Directory to store and unzip the files
dest_dir="/scratch/09534/reinhardh/c4"

# https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz?download=true


# Loop to download and decompress each file
for i in $(seq -f "%05g" 389 $((num_files-1))); do
    # Construct the full URL
    file_url="${base_url}${i}-of-00008.json.gz"
    
    # Destination path
    dest_path="${dest_dir}/c4-train.${i}-of-00008.json.gz"
    
    # Download the file to the specified directory
    wget $file_url -O $dest_path
    
    # Decompress the downloaded file in the directory
    gunzip "$dest_path"
done
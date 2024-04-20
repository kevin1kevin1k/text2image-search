#!/usr/bin/env bash

# Source: https://storage.googleapis.com/openimages/web/download_v7.html
wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv
wget https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv
wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv

python preprocess_dataset.py

img2dataset \
    --url_list=urls.csv \
    --input_format=csv \
    --save_additional_columns="['image_labels','bbox_labels']" \
    --output_folder=open_images \
    --processes_count=4 \
    --image_size=256

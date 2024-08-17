#!/bin/bash

# Install necessary packages
# pip install kaggle

# Ensure the kaggle.json file exists
KAGGLE_JSON_PATH=~/.kaggle/kaggle.json

if [ -f "$KAGGLE_JSON_PATH" ]; then
    echo "kaggle.json found. Setting file permissions."
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "kaggle.json not found. Please place it in the ~/.kaggle directory."
    exit 1
fi

# Download the dataset to the specified folder
# kaggle datasets download -d sabahesaraki/2017-2017 -p /teamspace/studios/this_studio/data/MS_COCO
# kaggle datasets download -d hsankesara/flickr-image-dataset -p /teamspace/studios/this_studio/data/Flickr30

# Unzip the dataset if necessary (uncomment the next line if the dataset is zipped)
# unzip /teamspace/studios/this_studio/data/2017-2017.zip

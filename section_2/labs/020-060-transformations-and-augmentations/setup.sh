#!/bin/bash

# Set hostname
hostname pytorch

# data directory
mkdir -pv pytorch-certification/data

# install Python
apt-get update && \
    apt-get install -y python3 python3-pip python3-venv wget

# Additional packages
apt-get install -y ffmpeg libsm6 libxext6

# Activate python environment
python3 -m venv venv
source venv/bin/activate

# Use a dedicated temporary directory for pip installations
mkdir -p /root/pip-tmp
export TMPDIR=/root/pip-tmp

# Download and install the course requirements
wget -O requirements.txt https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/requirements.txt
python3 -m pip install -r requirements.txt

# Download the lab questions
LAB_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_2/labs/020-060-transformations-and-augmentations
for question_number in {1..8}; do
    wget -O "question_$question_number.py" \
        "$LAB_BASE_URL/question_$question_number.py"
done

# Download the image dataset used by the lab
mkdir -p images/cat images/dog
for class_name in cat dog; do
    for image_number in {1..5}; do
        wget -O "images/$class_name/$class_name-$image_number.jpg" \
            "$LAB_BASE_URL/images/$class_name/$class_name-$image_number.jpg"
    done
done

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

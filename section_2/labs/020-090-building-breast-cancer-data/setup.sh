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

# Download the lab exercises
LAB_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_2/labs/020-090-building-breast-cancer-data
for lab_file in \
    initial_annotations.py \
    initial_dataset.py \
    split_data.py \
    subset_annotations.py \
    create_transformations.py \
    create_datasets.py \
    create_dataloaders.py; do
    wget -O "$lab_file" "$LAB_BASE_URL/$lab_file"
done

# Download and extract the breast cancer image dataset
wget -O data.tar.gz https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/data.tar.gz
tar -xf data.tar.gz
chown -R root:root data
rm -f data.tar.gz

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

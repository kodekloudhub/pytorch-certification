#!/bin/bash

# Set hostname
hostname pytorch

# data directory
mkdir -pv pytorch-certification/data

# install Python
apt-get update && \
    apt-get install -y python3 python3-pip python3-venv wget git

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

# Download the lab exercises and supporting CSV files
LAB_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_3/labs/030-120-transfer-learning
for lab_file in \
    pre.py \
    list_available_models.py \
    list_pytorch_vision_models.py \
    load_mobilenet_v3.py \
    freeze_mobilenet_v3.py \
    mobilenet_v3_scheduler.py \
    train_mobilenet_v3.py \
    load_resnet_model.py \
    modify_resnet_output.py \
    resnet_scheduler.py \
    train_resnet.py \
    training_data.csv \
    val_data.csv; do
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

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

# Download the lab exercises and supporting files
LAB_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_3/labs/030-060-train-and-save-an-image-classi
for lab_file in \
    pre.py \
    breast_cancer_net.py \
    create_model.py \
    loss_function_optimizer.py \
    training_loop.py \
    load_model.py \
    test_inference.py \
    training_data.csv \
    val_data.csv \
    sample-input.jpg; do
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

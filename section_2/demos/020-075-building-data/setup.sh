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

# Download the demo notebook
DEMO_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_2/demos/020-075-building-data
wget -O building-data.ipynb "$DEMO_BASE_URL/building-data.ipynb"

# Download the image dataset used by the notebook
mkdir -p images/cat images/dog
for image_number in {1..5}; do
    wget -O "images/cat/cat-$image_number.jpg" \
        "$DEMO_BASE_URL/images/cat/cat-$image_number.jpg"
    wget -O "images/dog/dog-$image_number.jpg" \
        "$DEMO_BASE_URL/images/dog/dog-$image_number.jpg"
done
wget -O images/cat/frog-1.jpg "$DEMO_BASE_URL/images/cat/frog-1.jpg"
wget -O images/cat/horse-1.jpg "$DEMO_BASE_URL/images/cat/horse-1.jpg"

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

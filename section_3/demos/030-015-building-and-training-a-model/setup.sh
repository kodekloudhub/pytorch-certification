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
DEMO_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_3/demos/030-015-building-and-training-a-model
wget -O building-and-training-a-model.ipynb "$DEMO_BASE_URL/building-and-training-a-model.ipynb"

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

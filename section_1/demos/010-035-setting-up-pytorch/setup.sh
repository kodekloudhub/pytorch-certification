#!/bin/bash

# Set hostname
hostname pytorch-certification

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

# Download and install the course requirements
wget -O requirements.txt https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/requirements.txt
python3 -m pip install -r requirements.txt

# Confirm that PyTorch was installed in the active environment.
python3 -c 'import torch; print(f"PyTorch: {torch.__version__}")'

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

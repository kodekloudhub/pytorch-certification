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

# Keep the repository-relative layout required by the notebook's image paths
COURSE_ROOT=pytorch-certification
DEMO_DIR="$COURSE_ROOT/section_3/demos/030-135-model-evaluation"
mkdir -p "$DEMO_DIR"

# Download the demo notebook and supporting resources
DEMO_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_3/demos/030-135-model-evaluation
for demo_file in \
    model-evaluation.ipynb \
    sample-input.jpg \
    test_data.csv \
    2_checkpoint.tar; do
    wget -O "$DEMO_DIR/$demo_file" "$DEMO_BASE_URL/$demo_file"
done

# Download and extract the breast cancer image dataset at the course root
wget -O "$COURSE_ROOT/data.tar.gz" https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/data.tar.gz
tar -xf "$COURSE_ROOT/data.tar.gz" -C "$COURSE_ROOT"
chown -R root:root "$COURSE_ROOT/data"
rm -f "$COURSE_ROOT/data.tar.gz"

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

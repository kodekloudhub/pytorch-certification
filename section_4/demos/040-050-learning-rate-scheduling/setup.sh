#!/bin/bash

# Set hostname
hostname pytorch

# install Python
apt-get update && \
    apt-get install -y python3 python3-pip python3-venv wget

# Activate python environment
python3 -m venv venv
source venv/bin/activate

# Use a dedicated temporary directory for pip installations
mkdir -p /root/pip-tmp
export TMPDIR=/root/pip-tmp

# Download and install the course requirements
wget -O requirements.txt https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/requirements.txt
python3 -m pip install -r requirements.txt

# Download the demo notebook and standalone reference
DEMO_BASE_URL=https://raw.githubusercontent.com/kodekloudhub/pytorch-certification/main/section_4/demos/040-050-learning-rate-scheduling
wget -O learning-rate-scheduling.ipynb "$DEMO_BASE_URL/learning-rate-scheduling.ipynb"
wget -O learning_rate_scheduling.py "$DEMO_BASE_URL/learning_rate_scheduling.py"

# Install and start code server
curl -fsSL https://code-server.dev/install.sh | sh
cat /root/.config/code-server/config.yaml
code-server --bind-addr 0.0.0.0:9000

#!/usr/bin/env bash
# build.sh

apt-get update
echo "📦 Installing ffmpeg"
apt-get install -y ffmpeg

echo "📦 Installing project dependencies..."
pip install -r requirements.txt

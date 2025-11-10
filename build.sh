#!/usr/bin/env bash
# build.sh

echo "🚀 Upgrading pip and setup tools first..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing project dependencies..."
pip install -r requirements.txt

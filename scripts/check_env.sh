#!/usr/bin/env bash

# Check if running inside WSL or native Linux
if ! grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null && [ "$(uname)" != "Linux" ]; then
    echo "====================================================================="
    echo " [ERROR] Unsupported Environment Detected!"
    echo "====================================================================="
    echo " Emora's 3D pipeline relies on PyTorch3D and CUDA binaries built"
    echo " specifically for Linux/WSL2."
    echo ""
    echo " Please run this project inside:"
    echo "   1. Windows Subsystem for Linux (WSL2 with Ubuntu)"
    echo "   2. A native Linux environment with an NVIDIA GPU and CUDA 11.8+"
    echo ""
    echo " If you are on Windows, please open your terminal inside WSL2"
    echo " (type 'wsl' in PowerShell/CMD) before running setup."
    echo "====================================================================="
    exit 1
fi
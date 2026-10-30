#!/usr/bin/env bash
set -e

# --- PRE-FLIGHT ENVIRONMENT CHECK ---
if ! grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null && [ "$(uname)" != "Linux" ]; then
    echo "[ERROR] This setup script must be run inside WSL2 or native Linux. Exiting."
    exit 1
fi

echo "=== Initializing Emora (Arome) Environment ==="
# Rest of your setup commands...

#!/usr/bin/env bash
set -e

echo "=== Initializing Emora (Arome) Environment ==="

# 1. System Dependencies
sudo apt update && sudo apt install -y git python3.10 python3.10-venv python3.10-dev ffmpeg libgl1 libglib2.0-0 blender unzip

# 2. Virtual Environment Setup (Native Linux storage)
python3.10 -m venv ~/emora-env
source ~/emora-env/bin/activate
pip install --upgrade pip "setuptools<66" wheel

# 3. Core ML & Strict NumPy Pin
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath
pip install "numpy<2.0,>=1.23.5" kornia==0.6.8 face_alignment==1.5.0 "opencv-python<4.9.0" scikit-image yacs "PyYAML>=6.0" "trimesh[easy]" streamlit

# 4. Precompiled PyTorch3D Wheel for Python 3.10 / CUDA 11.8
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

# 5. Clone DECA & Install Chumpy
cd /mnt/d/Emora/emora-stage1
if [ ! -d "third_party/DECA" ]; then
    echo "Cloning DECA..."
    git clone https://github.com/YadiraF/DECA third_party/DECA
fi
pip install chumpy --no-build-isolation || true

# 6. Apply Face Alignment Enum Patch Automatically
python3 -c '
path = "third_party/DECA/decalib/datasets/detectors.py"
with open(path, "r") as f:
    content = f.read()
if "face_alignment.LandmarksType._2D" in content:
    content = content.replace(
        "face_alignment.LandmarksType._2D",
        "face_alignment.LandmarksType.k2D if hasattr(face_alignment.LandmarksType, \"k2D\") else list(face_alignment.LandmarksType)[0]"
    )
    with open(path, "w") as f:
        f.write(content)
    print("Patched DECA detectors.py successfully!")
'

# 7. Fetch FLAME Model Weights directly (Bypassing interactive loop)
mkdir -p third_party/DECA/data
if [ ! -f "third_party/DECA/data/generic_model.pkl" ]; then
    echo "Downloading FLAME model weights..."
    curl -L -o third_party/DECA/data/generic_model.pkl "https://huggingface.co/camenduru/show/resolve/main/data/flame/2019/generic_model.pkl"
fi

echo "====================================================="
echo "Setup complete! Run 'source ~/emora-env/bin/activate'"
echo "to start running your reconstruction pipeline."
echo "====================================================="
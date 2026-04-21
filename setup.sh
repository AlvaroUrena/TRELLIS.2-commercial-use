#!/usr/bin/env bash
# TRELLIS.2 Setup Script for Bash
# Usage: source setup.sh [OPTIONS]
#
# IMPORTANT: Before running, ensure CUDA_HOME is set to CUDA 12.4:
#   export CUDA_HOME="$HOME/.local/cuda-12.4"  # or /usr/local/cuda-12.4
#   export PATH="$CUDA_HOME/bin:$PATH"
#   export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,drtk -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
DRTK=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --drtk) DRTK=true ; shift ;;

        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: source setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --drtk                  Install DRTK (differentiable renderer, MIT license)"

    return 0
fi

# Get system information
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    return 1
fi

# Detect distro for package manager
if [ -f /etc/debian_version ]; then
    DISTRO="debian"
elif [ -f /etc/arch-release ]; then
    DISTRO="arch"
elif [ -f /etc/fedora-release ]; then
    DISTRO="fedora"
else
    DISTRO="unknown"
fi

# Check CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    echo "WARNING: CUDA_HOME is not set. CUDA extensions require CUDA 12.4."
    echo "Set CUDA_HOME before running this script:"
    echo '  export CUDA_HOME="$HOME/.local/cuda-12.4"'
    echo '  export PATH="$CUDA_HOME/bin:$PATH"'
    echo '  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"'
fi

# Initialize git submodules (required for o-voxel's eigen dependency)
echo "[INIT] Initializing git submodules..."
git submodule update --init --recursive

if [ "$NEW_ENV" = true ] ; then
    echo "[NEW_ENV] Creating conda environment 'trellis2' with Python 3.10..."
    conda create -n trellis2 python=3.10 -y
    conda activate trellis2
    if [ "$PLATFORM" = "cuda" ] ; then
        echo "[NEW_ENV] Installing PyTorch 2.6.0 with CUDA 12.4..."
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[NEW_ENV] Installing PyTorch 2.6.0 with ROCm 6.2.4..."
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
fi

if [ "$BASIC" = true ] ; then
    echo "[BASIC] Installing basic Python dependencies..."
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard 'pygltflib>=1.16.0'
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    
    echo "[BASIC] Installing libjpeg development library..."
    case "$DISTRO" in
        debian)
            sudo apt install -y libjpeg-dev
            ;;
        arch)
            sudo pacman -S --noconfirm libjpeg-turbo
            ;;
        fedora)
            sudo dnf install -y libjpeg-turbo-devel
            ;;
        *)
            echo "[BASIC] Warning: Could not detect distro. Install libjpeg manually if needed."
            ;;
    esac
    
    echo "[BASIC] Installing pillow-simd, kornia, timm, psutil..."
    pip install pillow-simd kornia timm psutil
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        echo "[FLASHATTN] Installing flash-attn 2.7.3 (prebuilt wheel for PyTorch 2.6 + CUDA 12.4)..."
        # Prebuilt wheel is faster and more reliable than building from source
        pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install
        cd "$WORKDIR"
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

# Install DRTK (requires patches for CPU kernels and setuptools compatibility)
if [ "$DRTK" = true ] ; then
    echo "[DRTK] Installing DRTK with patches for CPU kernel support..."
    
    ORIG_SETUPTOOLS=$(pip show setuptools 2>/dev/null | grep -oP '^Version: \K.*' || echo "82.0.1")
    echo "[DRTK] Temporarily downgrading setuptools (required for pkg_resources)..."
    pip install setuptools==69.5.1
    
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/DRTK
    git clone https://github.com/facebookresearch/DRTK.git /tmp/extensions/DRTK
    
    # Patch setup.py to include missing CPU kernel sources
    # 1. Add interpolate_kernel_cpu.cpp
    sed -i 's|"src/interpolate/interpolate_kernel.cu",|"src/interpolate/interpolate_kernel.cu",\n                    "src/interpolate/interpolate_kernel_cpu.cpp",|' /tmp/extensions/DRTK/setup.py
    
    # 2. Add rasterize_kernel_cpu.cpp
    sed -i 's|"src/rasterize/rasterize_kernel.cu",|"src/rasterize/rasterize_kernel.cu",\n                    "src/rasterize/rasterize_kernel_cpu.cpp",|' /tmp/extensions/DRTK/setup.py
    
    # 3. Add edge_grad_kernel_cpu.cpp
    sed -i 's|"src/edge_grad/edge_grad_kernel.cu",|"src/edge_grad/edge_grad_kernel.cu",\n                    "src/edge_grad/edge_grad_kernel_cpu.cpp",|' /tmp/extensions/DRTK/setup.py
    
    # 4. Add render_kernel_cpu.cpp
    sed -i 's|"src/render/render_kernel.cu", "src/render/render_module.cpp"|"src/render/render_kernel.cu", "src/render/render_module.cpp", "src/render/render_kernel_cpu.cpp"|' /tmp/extensions/DRTK/setup.py
    
    # 5. Patch cpu_atomic.h for C++17 compatibility (use reference instead of copy)
    sed -i 's/auto target = detail::atomic_ref_at/auto\& target = detail::atomic_ref_at/g' /tmp/extensions/DRTK/src/include/cpu_atomic.h
    
    echo "[DRTK] Building DRTK (this may take a few minutes)..."
    pip install /tmp/extensions/DRTK --no-build-isolation
    
    echo "[DRTK] Restoring setuptools to $ORIG_SETUPTOOLS..."
    pip install setuptools=="$ORIG_SETUPTOOLS"
fi

if [ "$CUMESH" = true ] ; then
    echo "[CUMESH] Installing CuMesh..."
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation
fi

if [ "$FLEXGEMM" = true ] ; then
    echo "[FLEXGEMM] Installing FlexGEMM..."
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
fi

# Install o-voxel (editable mode to find trellis2/utils)
if [ "$OVOXEL" = true ] ; then
    echo "[OVOXEL] Installing o-voxel in editable mode (requires project directory)..."
    pip install -e "$WORKDIR/o-voxel" --no-build-isolation
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  conda activate trellis2"
echo ""
echo "To verify installation:"
echo "  cd $WORKDIR"
echo "  python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')\""
echo "  python -c \"import flash_attn; print(f'flash-attn {flash_attn.__version__}')\""
echo "  python -c \"import drtk; print('drtk OK')\""
echo "  PYTHONPATH=. python -c \"import o_voxel; print('o-voxel OK')\""
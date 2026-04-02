#!/bin/bash
set -e

echo "=== ChACRA Automated Installation ==="

# 1. System checks
if ! command -v conda &> /dev/null && ! command -v mamba &> /dev/null; then
    echo "Error: conda or mamba must be installed."
    exit 1
fi

if ! command -v mpicc &> /dev/null; then
    echo "Error: mpicc not found. Please install the OpenMPI development headers (e.g., sudo apt install libopenmpi-dev openmpi-bin)."
    exit 1
fi

if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install OpenMPI (e.g., sudo apt install openmpi-bin)."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# 2. Extract CUDA version and determine CuPy package
# nvidia-smi outputs something like "CUDA Version: 13.0"
CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version from nvidia-smi. Defaulting to cupy-cuda12x."
    CUPY_PKG="cupy-cuda12x"
else
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$MAJOR" == "11" ]; then
        CUPY_PKG="cupy-cuda11x"
    elif [ "$MAJOR" == "12" ] || [ "$MAJOR" == "13" ]; then
        # CuPy 12x wheels are used for CUDA 13 as well as 12
        CUPY_PKG="cupy-cuda12x"
    else
        echo "Warning: Unrecognized CUDA major version ($MAJOR). Defaulting to cupy-cuda12x."
        CUPY_PKG="cupy-cuda12x"
    fi
fi
echo "Detected CUDA Version: $CUDA_VER. Will install $CUPY_PKG."

# 3. Create conda base environment
echo "Creating conda environment from environment.yaml..."
CONDA_CMD="conda"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
fi

# Read environment name from yaml file
ENV_NAME=$(grep -E "^name:" environment.yaml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    ENV_NAME="chacra-env"
fi

$CONDA_CMD env update -f environment.yaml --prune

# 4. Pip post-install
echo "Installing mpi4py from source, $CUPY_PKG, and femto..."
# Get mpicc path
SYSTEM_MPICC=$(which mpicc)

cat << EOF > post_install_tmp.sh
#!/bin/bash
set -e
export MPICC=$SYSTEM_MPICC

echo "Using MPICC=\$MPICC"

# Install pip packages inside the conda env
pip install --no-cache-dir $CUPY_PKG
pip install --no-binary mpi4py --no-cache-dir mpi4py
pip install --no-cache-dir git+https://github.com/Dan-Burns/femto.git@mps_hremd git+https://github.com/Dan-Burns/ultracontacts.git
EOF

chmod +x post_install_tmp.sh
$CONDA_CMD run -n "$ENV_NAME" ./post_install_tmp.sh
rm post_install_tmp.sh

echo "=== Installation Complete ==="
echo "You can now activate the environment:"
echo "    conda activate $ENV_NAME"

#!/bin/bash
set -e

echo "=== ChACRA Automated Installation ==="

# ── 1. System checks ──────────────────────────────────────────────────────────

if ! command -v conda &> /dev/null && ! command -v mamba &> /dev/null && ! command -v micromamba &> /dev/null; then
    echo "Error: conda, mamba, or micromamba must be installed."
    exit 1
fi

if ! command -v mpicc &> /dev/null; then
    echo "Error: mpicc not found. Please install the OpenMPI development headers."
    echo "  e.g.  sudo apt install libopenmpi-dev openmpi-bin"
    exit 1
fi

if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install OpenMPI."
    echo "  e.g.  sudo apt install openmpi-bin"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# ── 2. Detect CUDA version and select CuPy wheel ─────────────────────────────

CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version from nvidia-smi. Defaulting to cupy-cuda12x."
    CUPY_PKG="cupy-cuda12x"
else
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$MAJOR" == "11" ]; then
        CUPY_PKG="cupy-cuda11x"
    elif [ "$MAJOR" == "12" ] || [ "$MAJOR" == "13" ]; then
        CUPY_PKG="cupy-cuda12x"
    else
        echo "Warning: Unrecognized CUDA major version ($MAJOR). Defaulting to cupy-cuda12x."
        CUPY_PKG="cupy-cuda12x"
    fi
fi
echo "Detected CUDA Version: $CUDA_VER. Will install $CUPY_PKG."

# ── 3. Pick the best available conda frontend ─────────────────────────────────
# Prefer micromamba > mamba > conda (faster solve, same interface)

if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "Using conda frontend: $CONDA_CMD"

# ── 4. Create / update the conda environment ──────────────────────────────────

ENV_NAME=$(grep -E "^name:" environment.yaml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    ENV_NAME="chacra-env"
fi
echo "Target environment: $ENV_NAME"

echo "Creating/updating conda environment from environment.yaml..."
$CONDA_CMD env update -f environment.yaml --prune

# ── 5. Pip post-install (runs *inside* the conda env) ────────────────────────
#
# mpi4py builds against the system MPI (mpicc must be on PATH).
# No conda openmpi is installed — this avoids compiler_compat conflicts
# and ensures mpi4py links against the same libmpi that mpirun uses.

echo "Installing $CUPY_PKG, mpi4py, femto, getcontacts, and ultracontacts..."

cat << EOF > post_install_tmp.sh
#!/bin/bash
set -e

echo "Building mpi4py against system MPI (mpicc=$(which mpicc))"

pip install --no-cache-dir $CUPY_PKG
pip install --no-cache-dir mpi4py
pip install --no-cache-dir "git+https://github.com/Dan-Burns/femto.git"

# getcontacts declares vmd-python (conda-only, not on PyPI) and
# ultracontacts declares cupy (generic, can't build from source).
# Both are already satisfied by the conda env + cupy-cudaXX above.
pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/getcontacts.git"
pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/ultracontacts.git"

# Install ChACRA itself in editable mode so local edits are reflected.
pip install --no-cache-dir -e "$(pwd)"
EOF

chmod +x post_install_tmp.sh
$CONDA_CMD run -n "$ENV_NAME" bash post_install_tmp.sh
rm post_install_tmp.sh

echo ""
echo "=== Installation Complete ==="
echo "Activate the environment with:"
echo "    conda activate $ENV_NAME"

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

# ── 2. Detect CUDA version from the driver ────────────────────────────────────
# nvidia-smi reports the highest CUDA version the *driver* supports.
# We pin cuda-version to this exact value so that the CUDA runtime
# packages (cuda-nvrtc, libcufft, etc.) match the driver and we avoid
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222).

CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version from nvidia-smi."
    echo "         Conda will pick the default CUDA variant — this may cause PTX errors."
    CUDA_MAJOR=""
else
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "Detected driver CUDA version: $CUDA_VER (major=$CUDA_MAJOR)"
fi

# Select the correct CuPy wheel
if [ "$CUDA_MAJOR" == "11" ]; then
    CUPY_PKG="cupy-cuda11x"
elif [ "$CUDA_MAJOR" == "12" ] || [ "$CUDA_MAJOR" == "13" ]; then
    CUPY_PKG="cupy-cuda12x"
else
    echo "Warning: Unrecognized or missing CUDA major version ($CUDA_MAJOR). Defaulting to cupy-cuda12x."
    CUPY_PKG="cupy-cuda12x"
fi
echo "Will install $CUPY_PKG."

# ── 3. Pick the best available conda frontend ─────────────────────────────────

if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "Using conda frontend: $CONDA_CMD"

# ── 4. Create / update the conda environment ──────────────────────────────────
# We append a cuda-version pin to the environment spec so the solver
# installs cuda-nvrtc / libcufft builds that match the driver exactly.
# Without this, the solver may pick a newer CUDA toolkit than the driver
# supports, causing PTX version errors at runtime.

ENV_NAME=$(grep -E "^name:" environment.yaml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    ENV_NAME="chacra-env"
fi
echo "Target environment: $ENV_NAME"

# Build a temporary yaml with the cuda-version pin injected
if [ -n "$CUDA_VER" ]; then
    echo "  Pinning cuda-version=${CUDA_VER} to match driver"
    # Append the pin as the last dependency
    sed "/^  - pre-commit$/a\\  - cuda-version=${CUDA_VER}" environment.yaml > _env_pinned.yaml
    ENV_FILE="_env_pinned.yaml"
else
    ENV_FILE="environment.yaml"
fi

echo "Creating/updating conda environment..."
CONDA_OVERRIDE_CUDA="${CUDA_VER:-13.0}" $CONDA_CMD env update -f "$ENV_FILE" --prune

# Clean up temp file
[ -f _env_pinned.yaml ] && rm _env_pinned.yaml

# ── 5. Pip post-install (runs *inside* the conda env) ────────────────────────
#
# mpi4py builds against the system MPI (mpicc must be on PATH).
# No conda openmpi is installed — this avoids compiler_compat conflicts
# and ensures mpi4py links against the same libmpi that mpirun uses.

echo "Installing $CUPY_PKG, mpi4py, femto, getcontacts, and ultracontacts..."

cat << EOF > post_install_tmp.sh
#!/bin/bash
set -e

echo "Building mpi4py against system MPI (mpicc=\$(which mpicc))"

pip install --no-cache-dir $CUPY_PKG
pip install --no-cache-dir mpi4py
pip install --no-cache-dir "git+https://github.com/Dan-Burns/femto.git"

# getcontacts declares vmd-python (conda-only, not on PyPI) and
# ultracontacts declares cupy (generic, can't build from source).
# Both are already satisfied by the conda env + cupy-cudaXX above.
pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/getcontacts.git"
pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/ultracontacts.git"

# Install ChACRA itself in editable mode so local edits are reflected.
pip install --no-cache-dir -e "\$(pwd)"
EOF

chmod +x post_install_tmp.sh
$CONDA_CMD run -n "$ENV_NAME" bash post_install_tmp.sh
rm post_install_tmp.sh

echo ""
echo "=== Installation Complete ==="
echo "Activate the environment with:"
echo "    conda activate $ENV_NAME"

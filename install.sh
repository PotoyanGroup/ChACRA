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
# nvidia-smi reports the highest CUDA toolkit the *driver* supports.
# We use CONDA_OVERRIDE_CUDA to tell the solver what virtual CUDA
# package to assume.  We do NOT inject a hard cuda-version pin into the
# environment spec because conda-forge may not yet have builds for the
# exact driver version (e.g. 13.0) — that would make the solve impossible.

CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version from nvidia-smi."
    echo "         Conda will pick the default CUDA variant."
    CUDA_MAJOR=""
    CONDA_CUDA_OVERRIDE=""
else
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "Detected driver CUDA version: $CUDA_VER (major=$CUDA_MAJOR)"

    # conda-forge packages are built against specific CUDA versions.
    # If the driver supports 13.x, packages built for 12.x are still
    # compatible (forward-compat).  Map to the latest conda-forge
    # CUDA version that actually has builds.
    if [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
        # CUDA 13+ driver: use 12.x builds (forward-compatible)
        CONDA_CUDA_OVERRIDE="12.8"
        echo "  Driver CUDA >= 13 — using CUDA 12.8 builds (forward-compatible)"
    elif [ "$CUDA_MAJOR" == "12" ]; then
        CONDA_CUDA_OVERRIDE="$CUDA_VER"
    elif [ "$CUDA_MAJOR" == "11" ]; then
        CONDA_CUDA_OVERRIDE="$CUDA_VER"
    else
        CONDA_CUDA_OVERRIDE=""
    fi
fi

# Select the correct CuPy wheel
if [ "$CUDA_MAJOR" == "11" ]; then
    CUPY_PKG="cupy-cuda11x"
elif [ -n "$CUDA_MAJOR" ]; then
    # CUDA 12.x, 13.x, and future versions all use the 12x wheel
    CUPY_PKG="cupy-cuda12x"
else
    echo "Warning: Unrecognized CUDA version. Defaulting to cupy-cuda12x."
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
# CONDA_OVERRIDE_CUDA tells the solver which __cuda virtual package to
# assume, so it picks the right openmm/openff/etc builds.
# We do NOT add cuda-version as an explicit dependency — that causes
# unsolvable environments when the driver's CUDA version is newer than
# what conda-forge has built packages for.

ENV_NAME=$(grep -E "^name:" environment.yaml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    ENV_NAME="chacra-env"
fi
echo "Target environment: $ENV_NAME"

# If using plain conda (not mamba/micromamba), enable the libmamba solver.
# The classic solver is extremely slow and can OOM on complex environments.
if [ "$CONDA_CMD" == "conda" ]; then
    if conda config --show solver 2>/dev/null | grep -q "libmamba"; then
        echo "  conda solver: libmamba (already configured)"
    else
        echo "  Enabling libmamba solver for faster environment resolution..."
        conda install -n base -y conda-libmamba-solver 2>/dev/null || true
        conda config --set solver libmamba 2>/dev/null || true
    fi
fi

# Determine whether to create or update
ENV_EXISTS=false
if conda env list 2>/dev/null | grep -qE "^${ENV_NAME}\s"; then
    ENV_EXISTS=true
elif micromamba env list 2>/dev/null | grep -qE "${ENV_NAME}"; then
    ENV_EXISTS=true
fi

echo "Creating/updating conda environment..."
if [ -n "$CONDA_CUDA_OVERRIDE" ]; then
    echo "  CONDA_OVERRIDE_CUDA=$CONDA_CUDA_OVERRIDE"
    export CONDA_OVERRIDE_CUDA="$CONDA_CUDA_OVERRIDE"
fi

if [ "$ENV_EXISTS" == "true" ]; then
    echo "  Environment exists — updating..."
    $CONDA_CMD env update -f environment.yaml --prune
else
    echo "  Creating fresh environment..."
    $CONDA_CMD env create -f environment.yaml
fi

# ── 5. Pip post-install (runs *inside* the conda env) ────────────────────────
#
# mpi4py builds against the system MPI (mpicc must be on PATH).
# No conda openmpi is installed — this avoids compiler_compat conflicts
# and ensures mpi4py links against the same libmpi that mpirun uses.

echo "Installing $CUPY_PKG, mpi4py, femto, getcontacts, and ultracontacts..."

cat <<EOF > post_install_tmp.sh
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

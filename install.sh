#!/bin/bash
# ============================================================================
# ChACRA Installation Script
#
# Primary path: install from a pre-solved conda lock file (fast, no solving).
# Fallback path: sequential group install if no lock file exists (slower,
#                but works without a pre-generated lock file).
#
# To generate lock files for your CUDA version on a high-RAM machine:
#   bash tools/generate_locks.sh
# ============================================================================
set -e

echo "=== ChACRA Automated Installation ==="

# ── 1. System checks ──────────────────────────────────────────────────────────

if ! command -v conda &>/dev/null && ! command -v mamba &>/dev/null && ! command -v micromamba &>/dev/null; then
    echo "Error: conda, mamba, or micromamba must be installed."
    exit 1
fi

if ! command -v mpicc &>/dev/null; then
    echo "Error: mpicc not found."
    echo "  e.g.  sudo apt install libopenmpi-dev openmpi-bin"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# ── 2. Detect CUDA ────────────────────────────────────────────────────────────

CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version."
    CUDA_MAJOR=""
else
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "Detected CUDA: $CUDA_VER (major=$CUDA_MAJOR)"
fi

# CuPy wheel selection (12.x and 13.x both use the 12x wheel)
if [ "$CUDA_MAJOR" == "11" ]; then
    CUPY_PKG="cupy-cuda11x"
else
    CUPY_PKG="cupy-cuda12x"
fi
echo "Will install: $CUPY_PKG"

# ── 3. Pick conda frontend ────────────────────────────────────────────────────

if command -v micromamba &>/dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
    # Enable libmamba solver — the classic solver OOMs on complex environments
    if ! conda config --show solver 2>/dev/null | grep -q "libmamba"; then
        echo "Enabling libmamba solver..."
        conda install -n base -y conda-libmamba-solver 2>/dev/null || true
        conda config --set solver libmamba 2>/dev/null || true
    fi
fi
echo "Using: $CONDA_CMD"

ENV_NAME=$(grep -E "^name:" environment.yaml | awk '{print $2}')
ENV_NAME="${ENV_NAME:-chacra-env}"
echo "Environment: $ENV_NAME"

# ── 4. Install the conda environment ─────────────────────────────────────────

# Check if environment already exists
ENV_EXISTS=false
if $CONDA_CMD env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
    ENV_EXISTS=true
fi

if [ "$ENV_EXISTS" == "true" ]; then
    # ── Existing env: just update ──────────────────────────────────────────────
    echo ""
    echo "Environment already exists — updating..."
    $CONDA_CMD env update -n "$ENV_NAME" -f environment.yaml

else
    # ── Fresh install ──────────────────────────────────────────────────────────
    # Prefer a pre-solved lock file: zero solving, works on any RAM.
    # Fall back to sequential group installs if no lock file is available.

    LOCK_FILE=""
    if [ -n "$CUDA_MAJOR" ] && [ -f "conda-lock.cuda${CUDA_MAJOR}.yml" ]; then
        LOCK_FILE="conda-lock.cuda${CUDA_MAJOR}.yml"
    elif [ -f "conda-lock.yml" ]; then
        LOCK_FILE="conda-lock.yml"
    fi

    if [ -n "$LOCK_FILE" ]; then
        # ── Lock file path ─────────────────────────────────────────────────────
        echo ""
        echo "Found lock file: $LOCK_FILE"
        echo "Installing from lock file (no solving required)..."

        if ! command -v conda-lock &>/dev/null; then
            echo "Installing conda-lock..."
            $CONDA_CMD install -n base -y -c conda-forge conda-lock
        fi

        conda-lock install -n "$ENV_NAME" "$LOCK_FILE"

    else
        # ── No lock file: sequential group installs ────────────────────────────
        # Solving 30+ CUDA-aware packages simultaneously can exhaust RAM.
        # Installing in small anchored groups keeps each solve small.
        echo ""
        echo "No lock file found. Installing in sequential groups..."
        echo "(To generate a lock file for faster installs on other machines:"
        echo "  bash tools/generate_locks.sh)"
        echo ""

        # CONDA_OVERRIDE_CUDA helps the solver pick the right CUDA builds
        if [ -n "$CUDA_VER" ]; then
            if [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
                export CONDA_OVERRIDE_CUDA="12.8"
            else
                export CONDA_OVERRIDE_CUDA="$CUDA_VER"
            fi
            echo "CONDA_OVERRIDE_CUDA=$CONDA_OVERRIDE_CUDA"
        fi

        C="-c conda-forge -n $ENV_NAME -y"

        echo "  Group 1/5: Python + OpenMM CUDA kernel..."
        $CONDA_CMD create -n "$ENV_NAME" -y -c conda-forge \
            "python>=3.10,<3.13" \
            "openmm>=8.1" \
            "openmmforcefields>=0.15.1" \
            "vmd-python" \
            "pdbfixer"

        echo "  Group 2/5: MD packages..."
        $CONDA_CMD install $C \
            "rdkit" \
            "mdanalysis>=2.9.0" \
            "mdtraj"

        echo "  Group 3/5: femto dependencies..."
        $CONDA_CMD install $C \
            "mdtop" \
            "pydantic-units" \
            "tensorboardx" \
            "pymbar>=4"

        echo "  Group 4/5: Scientific stack..."
        $CONDA_CMD install $C \
            "numpy" "scipy" "pandas" "matplotlib" \
            "scikit-learn" "networkx" \
            "polars" "pyarrow" "sympy"

        echo "  Group 5/5: Utilities..."
        $CONDA_CMD install $C \
            "pydantic>=2" "click" "cloup" "tqdm" \
            "pyyaml" "omegaconf" "gputil" "nglview" "pre-commit"
    fi
fi

# ── 5. Pip post-install ───────────────────────────────────────────────────────

echo ""
echo "Installing $CUPY_PKG, mpi4py, femto, getcontacts, ultracontacts, chacra..."

cat <<EOF > _post_install.sh
#!/bin/bash
set -e
echo "  mpi4py (against system MPI: \$(which mpicc))"
python -m pip install --no-cache-dir $CUPY_PKG
python -m pip install --no-cache-dir mpi4py
python -m pip install --no-cache-dir "git+https://github.com/Dan-Burns/femto.git"
python -m pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/getcontacts.git"
python -m pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/ultracontacts.git"
python -m pip install --no-cache-dir -e "\$(pwd)"
EOF

chmod +x _post_install.sh
$CONDA_CMD run -n "$ENV_NAME" bash _post_install.sh
rm _post_install.sh

echo ""
echo "=== Installation Complete ==="
echo "Activate with:  conda activate $ENV_NAME"
echo ""
echo "Tip: generate a lock file for faster installs on other machines:"
echo "     bash tools/generate_locks.sh"

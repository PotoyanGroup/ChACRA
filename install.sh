#!/bin/bash
# ============================================================================
# ChACRA Installation Script
#
# Usage:
#   bash install.sh               # fresh install or update existing env
#   bash install.sh --reinstall   # remove existing env and reinstall from scratch
#
# Fast path (no solving): uses conda/explicit-cuda{12,13}.txt if present.
# Fallback: sequential group solves (slower, for machines without spec files).
#
# To regenerate spec files after updating environment.yaml:
#   bash tools/generate_locks.sh
# ============================================================================
set -e

# ── 0. Parse flags ────────────────────────────────────────────────────────────
REINSTALL=false
for arg in "$@"; do
    case $arg in
        --reinstall) REINSTALL=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

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

ENV_NAME=$(grep -E "^name:" conda/environment.yaml | awk '{print $2}')
ENV_NAME="${ENV_NAME:-chacra-env}"
echo "Environment: $ENV_NAME"

# ── 4. Install the conda environment ─────────────────────────────────────────

# Check if environment already exists
ENV_EXISTS=false
if $CONDA_CMD env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
    ENV_EXISTS=true
fi

if [ "$ENV_EXISTS" == "true" ] && [ "$REINSTALL" == "true" ]; then
    echo ""
    echo "--reinstall: removing existing '$ENV_NAME' environment..."
    $CONDA_CMD env remove -n "$ENV_NAME" -y
    ENV_EXISTS=false
fi

if [ "$ENV_EXISTS" == "true" ]; then
    # ── Existing env: just update ──────────────────────────────────────────────
    echo ""
    echo "Environment '$ENV_NAME' already exists — updating from environment.yaml..."
    echo "(To do a clean reinstall from the fast explicit spec, use: bash install.sh --reinstall)"
    $CONDA_CMD env update -n "$ENV_NAME" -f conda/environment.yaml

else
    # ── Fresh install ──────────────────────────────────────────────────────
    # Priority order:
    #   1. @EXPLICIT spec file  (fastest — no solving, no extra tools)
    #   2. conda-lock YAML      (no solving, but needs conda-lock installed)
    #   3. Sequential group installs (slowest — full solver, high RAM)

    EXPLICIT_FILE=""
    LOCK_FILE=""

    # 1) Look for @EXPLICIT spec files
    if [ -n "$CUDA_MAJOR" ] && [ -f "conda/explicit-cuda${CUDA_MAJOR}.txt" ]; then
        EXPLICIT_FILE="conda/explicit-cuda${CUDA_MAJOR}.txt"
    elif [ -f "conda/explicit.txt" ]; then
        EXPLICIT_FILE="conda/explicit.txt"
    fi

    # 2) Look for conda-lock YAML files
    if [ -n "$CUDA_MAJOR" ] && [ -f "conda/conda-lock.cuda${CUDA_MAJOR}.yml" ]; then
        LOCK_FILE="conda/conda-lock.cuda${CUDA_MAJOR}.yml"
    elif [ -f "conda/conda-lock.yml" ]; then
        LOCK_FILE="conda/conda-lock.yml"
    fi

    if [ -n "$EXPLICIT_FILE" ]; then
        # ── Explicit spec path (fastest) ───────────────────────────────────
        echo ""
        echo "Found explicit spec: $EXPLICIT_FILE"
        echo "Installing directly (no solving, no extra tools)..."
        $CONDA_CMD create -n "$ENV_NAME" --file "$EXPLICIT_FILE" -y

    elif [ -n "$LOCK_FILE" ]; then
        # ── conda-lock YAML path ──────────────────────────────────────────
        echo ""
        echo "Found lock file: $LOCK_FILE"
        echo "Installing from lock file (no solving required)..."

        if ! command -v conda-lock &>/dev/null; then
            echo "conda-lock not found — installing into base environment..."
            echo "(Tip: generate explicit-*.txt files with 'bash tools/generate_locks.sh'"
            echo " to skip this step on future installs.)"
            $CONDA_CMD install -n base -y -c conda-forge conda-lock
        fi

        conda-lock install -n "$ENV_NAME" "$LOCK_FILE"

    else
        # ── No lock file: sequential group installs ────────────────────────
        # Solving 30+ CUDA-aware packages simultaneously can exhaust RAM.
        # Installing in small anchored groups keeps each solve small.
        echo ""
        echo "No lock or explicit spec files found. Installing in sequential groups..."
        echo "(To generate spec files for faster installs on other machines:"
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

# Ensure pip is available (lock/explicit files generated before pip was added
# to environment.yaml won't have it)
$CONDA_CMD run -n "$ENV_NAME" python -m ensurepip --upgrade 2>/dev/null || \
    $CONDA_CMD install -n "$ENV_NAME" -y pip 2>/dev/null || true

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

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
#
# The environment has many CUDA-aware packages (openmm, openmmforcefields,
# vmd-python, rdkit, mdanalysis, mdtraj) that must all resolve to mutually
# compatible builds.  Solving everything at once creates a huge constraint
# graph that can take 30+ minutes even with libmamba/micromamba.
#
# Strategy: two-pass install.
#   Pass 1 — install only the CUDA-coupled kernel (openmm, openmmforcefields,
#             vmd-python, pdbfixer).  This locks in the CUDA ABI and Python
#             version, giving the solver a tiny constraint space.
#   Pass 2 — install the full environment.  The solver now has openmm's
#             CUDA build as a fixed point and the remaining search space is
#             far smaller.
#
# CONDA_OVERRIDE_CUDA is exported for both passes so the solver selects
# the right CUDA build variant.

ENV_NAME=$(grep -E "^name:" environment.yaml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    ENV_NAME="chacra-env"
fi
echo "Target environment: $ENV_NAME"

# If using plain conda (not mamba/micromamba), enable the libmamba solver.
if [ "$CONDA_CMD" == "conda" ]; then
    if conda config --show solver 2>/dev/null | grep -q "libmamba"; then
        echo "  conda solver: libmamba (already configured)"
    else
        echo "  Enabling libmamba solver for faster environment resolution..."
        conda install -n base -y conda-libmamba-solver 2>/dev/null || true
        conda config --set solver libmamba 2>/dev/null || true
    fi
fi

if [ -n "$CONDA_CUDA_OVERRIDE" ]; then
    echo "  CONDA_OVERRIDE_CUDA=$CONDA_CUDA_OVERRIDE"
    export CONDA_OVERRIDE_CUDA="$CONDA_CUDA_OVERRIDE"
fi

# Determine whether the environment already exists
ENV_EXISTS=false
if $CONDA_CMD env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
    ENV_EXISTS=true
fi

if [ "$ENV_EXISTS" == "true" ]; then
    # ── Update path ────────────────────────────────────────────────────────────
    # For incremental updates we still need env update, but without --prune
    # to avoid re-solving the full removal set.  If you need --prune (to
    # remove deleted packages), run: micromamba env update -f environment.yaml --prune
    echo "  Environment exists — updating..."
    $CONDA_CMD env update -n "$ENV_NAME" -f environment.yaml
else
    # ── Fresh install: sequential small-group strategy ─────────────────────────
    # env update / env create with 30+ CUDA-aware packages simultaneously
    # can exhaust RAM on the solver even with micromamba/libmamba.
    # Installing in small groups keeps each solve's peak memory low.
    # Groups are ordered so heavier constraints are anchored early.

    C="-c conda-forge -n $ENV_NAME -y"

    # Group 1: Python + CUDA kernel — locks in CUDA ABI and Python version
    echo ""
    echo "  Group 1/6: Python + OpenMM CUDA kernel..."
    $CONDA_CMD create -n "$ENV_NAME" -y -c conda-forge \
        "python>=3.10,<3.13" \
        "openmm>=8.1" \
        "openmmforcefields>=0.15.1" \
        "vmd-python" \
        "pdbfixer"

    # Group 2: Bio/MD packages (some CUDA-aware, anchors on openmm's Python)
    echo "  Group 2/6: MD packages (rdkit, mdanalysis, mdtraj)..."
    $CONDA_CMD install $C \
        "rdkit" \
        "mdanalysis>=2.9.0" \
        "mdtraj"

    # Group 3: femto conda-only deps
    echo "  Group 3/6: femto dependencies..."
    $CONDA_CMD install $C \
        "mdtop" \
        "pydantic-units" \
        "tensorboardx" \
        "pymbar>=4"

    # Group 4: Core scientific stack
    echo "  Group 4/6: Scientific stack..."
    $CONDA_CMD install $C \
        "numpy" \
        "scipy" \
        "pandas" \
        "matplotlib" \
        "scikit-learn" \
        "networkx" \
        "polars" \
        "pyarrow" \
        "sympy"

    # Group 5: Utilities (all pure-Python or very lightweight)
    echo "  Group 5/6: Utilities..."
    $CONDA_CMD install $C \
        "pydantic>=2" \
        "click" \
        "cloup" \
        "tqdm" \
        "pyyaml" \
        "omegaconf" \
        "gputil" \
        "nglview" \
        "pre-commit"
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

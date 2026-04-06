#!/bin/bash
# ============================================================================
# ChACRA lock-file generator
#
# Run this ONCE on a machine with plenty of RAM (≥64 GB recommended) to
# generate platform+CUDA-specific lock files.  Commit the resulting
# conda-lock.yml files.  Other machines install from the lock file with
# no solving required (see install.sh).
#
# Usage:
#   bash tools/generate_locks.sh              # auto-detect CUDA from driver
#   bash tools/generate_locks.sh 12.6        # specify CUDA version explicitly
# ============================================================================
set -e

CUDA_VER="${1:-}"

# Auto-detect if not provided
if [ -z "$CUDA_VER" ]; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
    if [ -z "$CUDA_VER" ]; then
        echo "Error: Could not detect CUDA version. Pass it explicitly: bash tools/generate_locks.sh 12.6"
        exit 1
    fi
fi

echo "Generating lock file for CUDA ${CUDA_VER} ..."

# Round down to the latest conda-forge build for that major version
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
if [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
    LOCK_CUDA="12.8"
    echo "  (CUDA ${CUDA_VER} driver is forward-compatible with 12.8 builds)"
else
    LOCK_CUDA="$CUDA_VER"
fi

# Install conda-lock into base if not present
if ! command -v conda-lock &>/dev/null; then
    echo "Installing conda-lock into base environment..."
    conda install -n base -y -c conda-forge conda-lock
fi

# Write a temporary virtual-packages spec for this CUDA version
cat > _vp_tmp.yml <<VPEOF
subdirs:
  linux-64:
    packages:
      __cuda: "${LOCK_CUDA}"
VPEOF

LOCK_FILE="conda-lock.cuda${CUDA_MAJOR}.yml"

conda-lock \
    -f environment.yaml \
    --virtual-package-spec _vp_tmp.yml \
    --lockfile "$LOCK_FILE" \
    -p linux-64

rm _vp_tmp.yml

echo ""
echo "Lock file written: $LOCK_FILE"
echo "Commit this file.  On target machines, install with:"
echo "    conda-lock install -n chacra-env $LOCK_FILE"

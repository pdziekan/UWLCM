#!/bin/bash

# ==============================================================================
# Build Script for UWLCM Fugaku Singularity Image
# ==============================================================================
# This script helps build the Singularity container for Fugaku supercomputer
# ==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/uwlcm_fugaku_arm64.def"
SIF_FILE="${SCRIPT_DIR}/uwlcm_fugaku_arm64.sif"

echo "========================================================================"
echo "UWLCM Fugaku Singularity Image Builder"
echo "========================================================================"
echo ""

# Check if Singularity is installed
if ! command -v singularity &> /dev/null; then
    echo "ERROR: Singularity is not installed or not in PATH"
    echo "Please install Singularity from: https://sylabs.io/singularity/"
    exit 1
fi

echo "Singularity version:"
singularity version
echo ""

# Check architecture
ARCH=$(uname -m)
echo "Current architecture: ${ARCH}"

if [ "${ARCH}" != "aarch64" ] && [ "${ARCH}" != "arm64" ]; then
    echo ""
    echo "WARNING: You are not on ARM64 architecture!"
    echo "Building on ${ARCH} for ARM64 target (aarch64)"
    echo ""
    echo "Options:"
    echo "1. Use QEMU for cross-compilation (slow but works)"
    echo "2. Build on an ARM64 system (recommended)"
    echo "3. Build on Fugaku directly (if permitted)"
    echo ""
    read -p "Continue with cross-compilation using QEMU? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Build cancelled."
        exit 0
    fi
    
    # Check if QEMU is available
    if ! command -v qemu-aarch64-static &> /dev/null; then
        echo ""
        echo "ERROR: QEMU user emulation not found"
        echo "Install it with: sudo apt-get install qemu-user-static"
        exit 1
    fi
    
    BUILD_CMD="sudo singularity build --arch arm64"
else
    echo "Building natively on ARM64 architecture"
    BUILD_CMD="sudo singularity build"
fi

echo ""
echo "Definition file: ${DEF_FILE}"
echo "Output file: ${SIF_FILE}"
echo ""

# Check if .sif file already exists
if [ -f "${SIF_FILE}" ]; then
    echo "WARNING: ${SIF_FILE} already exists"
    read -p "Overwrite? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Build cancelled."
        exit 0
    fi
    rm -f "${SIF_FILE}"
fi

echo "========================================================================"
echo "Starting build..."
echo "Command: ${BUILD_CMD} ${SIF_FILE} ${DEF_FILE}"
echo "========================================================================"
echo ""

# Build the container
${BUILD_CMD} "${SIF_FILE}" "${DEF_FILE}"

BUILD_STATUS=$?

echo ""
echo "========================================================================"
if [ ${BUILD_STATUS} -eq 0 ]; then
    echo "Build completed successfully!"
    echo ""
    echo "Image file: ${SIF_FILE}"
    echo "Size: $(du -h ${SIF_FILE} | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "1. Transfer the image to Fugaku:"
    echo "   scp ${SIF_FILE} username@fugaku.riken.jp:~/"
    echo ""
    echo "2. On Fugaku, test the container:"
    echo "   singularity shell ${SIF_FILE##*/}"
    echo ""
    echo "3. Build libraries and UWLCM (see README_fugaku.md)"
    echo ""
    echo "4. Submit jobs using fugaku_job.sh"
else
    echo "Build failed with exit code ${BUILD_STATUS}"
    echo "Check the error messages above for details"
fi
echo "========================================================================"

exit ${BUILD_STATUS}

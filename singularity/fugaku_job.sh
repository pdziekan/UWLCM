#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM -L "elapse=01:00:00"
#PJM -s
#PJM -j
#PJM -o uwlcm_%j.out
#PJM -e uwlcm_%j.err

# ==============================================================================
# Fugaku Job Script for UWLCM
# ==============================================================================
# This script runs UWLCM inside a Singularity container on Fugaku supercomputer
#
# Modify the following variables according to your setup:
# - SINGULARITY_IMAGE: path to your .sif file
# - UWLCM_EXEC: path to compiled uwlcm binary
# - OUTPUT_DIR: where to store output files
# - Simulation parameters (--nx, --ny, --nz, --nt, etc.)
# ==============================================================================

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${PJM_JOBID}"

# ==============================================================================
# Configuration
# ==============================================================================

# Path to Singularity image
SINGULARITY_IMAGE="${HOME}/containers/uwlcm_fugaku_arm64.sif"

# Path to UWLCM executable (inside container)
UWLCM_EXEC="${HOME}/UWLCM/build/uwlcm"

# Output directory
OUTPUT_DIR="${HOME}/uwlcm_output/run_${PJM_JOBID}"
mkdir -p "${OUTPUT_DIR}"

# ==============================================================================
# OpenMP Configuration for A64FX (48 cores per node)
# ==============================================================================
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DISPLAY_ENV=TRUE

# ==============================================================================
# Simulation Parameters
# ==============================================================================

# Microphysics scheme: blk_1m, blk_2m, lgrngn, none
MICRO="blk_2m"

# Case: moist_thermal, dycoms_rf01, dycoms_rf02, cumulus_congestus_icmw20, etc.
CASE="moist_thermal"

# Grid resolution
NX=256
NY=256  # Set to 0 for 2D simulation
NZ=128

# Time integration
NT=3600      # Number of timesteps
DT=1         # Timestep in seconds

# Output settings
OUTFREQ=300  # Output every N timesteps
OUTSTART=0   # Start output after N timesteps

# Additional options
SPINUP=0
SERIAL=false  # Set to true to force single-threaded execution

# ==============================================================================
# Lagrangian Microphysics Options (if MICRO=lgrngn)
# ==============================================================================
LGRNGN_BACKEND="OpenMP"  # OpenMP or serial (CUDA not available on Fugaku)
SD_CONC=64               # Super-droplets per grid cell

# ==============================================================================
# Run UWLCM
# ==============================================================================

echo "Starting UWLCM simulation..."
echo "Microphysics: ${MICRO}"
echo "Case: ${CASE}"
echo "Grid: ${NX} x ${NY} x ${NZ}"
echo "Timesteps: ${NT} (dt=${DT}s)"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Base command
CMD="singularity exec ${SINGULARITY_IMAGE} ${UWLCM_EXEC}"
CMD="${CMD} --micro=${MICRO}"
CMD="${CMD} --case=${CASE}"
CMD="${CMD} --nx=${NX} --ny=${NY} --nz=${NZ}"
CMD="${CMD} --nt=${NT} --dt=${DT}"
CMD="${CMD} --outdir=${OUTPUT_DIR} --outfreq=${OUTFREQ} --outstart=${OUTSTART}"
CMD="${CMD} --spinup=${SPINUP}"
CMD="${CMD} --serial=${SERIAL}"

# Add Lagrangian-specific options if using lgrngn
if [ "${MICRO}" = "lgrngn" ]; then
    CMD="${CMD} --backend=${LGRNGN_BACKEND}"
    CMD="${CMD} --sd_conc=${SD_CONC}"
fi

# Run the simulation
echo "Executing: ${CMD}"
echo ""
${CMD}

EXIT_CODE=$?

# ==============================================================================
# Post-processing and cleanup
# ==============================================================================

echo ""
echo "Simulation finished with exit code: ${EXIT_CODE}"
echo "Job ended at: $(date)"

# Print output directory size
echo "Output directory size:"
du -sh "${OUTPUT_DIR}"

# Optional: Copy output to a different location
# cp -r "${OUTPUT_DIR}" /path/to/permanent/storage/

exit ${EXIT_CODE}

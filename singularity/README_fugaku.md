# Building UWLCM Singularity Image for Fugaku Supercomputer

## Overview

Fugaku is a supercomputer at RIKEN Center for Computational Science in Japan that uses ARM64 A64FX processors. This guide provides instructions for building and running UWLCM on Fugaku using Singularity containers.

## Important Notes

- **Architecture**: Fugaku uses ARM64 (aarch64) architecture with A64FX processors, not x86_64
- **No CUDA**: Fugaku does not use NVIDIA GPUs, so only CPU-based microphysics backends are available
- **Available backends**: OpenMP, serial (no CUDA or multi_CUDA)
- **Compiler optimization**: The container is configured with A64FX-specific optimizations (`-march=armv8.2-a+sve -mtune=a64fx`)

## Building the Singularity Image

### Option 1: Build on Fugaku (if Singularity is available)

```bash
# On Fugaku login node
cd /path/to/UWLCM/singularity
singularity build uwlcm_fugaku_arm64.sif uwlcm_fugaku_arm64.def
```

### Option 2: Build Remotely on Sylabs Cloud Builder

**Note**: Sylabs Cloud Builder may not support ARM64 architecture. You'll need to build on an ARM64 system.

### Option 3: Build on an ARM64 System

If Fugaku doesn't allow building Singularity images, you can build on another ARM64 system (e.g., AWS Graviton, Raspberry Pi 4+, or Apple Silicon with ARM emulation):

```bash
# On an ARM64 Linux system with Singularity installed
cd /path/to/UWLCM/singularity
sudo singularity build uwlcm_fugaku_arm64.sif uwlcm_fugaku_arm64.def
```

### Option 4: Cross-build with QEMU (slower but works on x86_64)

```bash
# On x86_64 system with QEMU and Singularity
# Install QEMU user emulation
sudo apt-get install qemu-user-static

# Build with --arch flag
sudo singularity build --arch arm64 uwlcm_fugaku_arm64.sif uwlcm_fugaku_arm64.def
```

## Transferring Image to Fugaku

Once built, transfer the `.sif` file to Fugaku:

```bash
# From your local system
scp uwlcm_fugaku_arm64.sif username@fugaku.riken.jp:/path/to/destination/
```

## Using the Container on Fugaku

### Interactive Shell

```bash
singularity shell uwlcm_fugaku_arm64.sif
```

### Building Libraries Inside Container

1. **Clone and build libmpdata++**:

```bash
singularity exec uwlcm_fugaku_arm64.sif bash -c "
git clone --recursive https://github.com/igfuw/libmpdataxx.git
cd libmpdataxx
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local_install -DCMAKE_BUILD_TYPE=Release
make -j 8
make install
"
```

2. **Clone and build libcloudph++**:

```bash
singularity exec uwlcm_fugaku_arm64.sif bash -c "
git clone --recursive https://github.com/igfuw/libcloudphxx.git
cd libcloudphxx
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local_install \
         -DCMAKE_BUILD_TYPE=Release
make -j 8
make install
"
```

3. **Build UWLCM**:

```bash
singularity exec uwlcm_fugaku_arm64.sif bash -c "
cd /path/to/UWLCM
mkdir build && cd build
cmake .. -Dlibmpdata++_DIR=$HOME/local_install/share/libmpdata++ \
         -Dlibcloudph++_DIR=$HOME/local_install/lib/cmake/libcloudph++ \
         -DCMAKE_BUILD_TYPE=Release
make -j 8
"
```

## Running UWLCM on Fugaku

### Interactive Run (for testing)

```bash
singularity exec uwlcm_fugaku_arm64.sif /path/to/UWLCM/build/uwlcm \
    --micro=blk_1m \
    --case=moist_thermal \
    --nx=128 --ny=0 --nz=128 \
    --nt=1800 --dt=1 \
    --outdir=output --outfreq=60
```

### Job Script for Fugaku

Create a job script `run_uwlcm.sh`:

```bash
#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM -L "elapse=01:00:00"
#PJM -s
#PJM -j

# Set OpenMP threads (Fugaku has 48 cores per node)
export OMP_NUM_THREADS=48

# Run UWLCM inside Singularity container
singularity exec uwlcm_fugaku_arm64.sif /path/to/UWLCM/build/uwlcm \
    --micro=blk_2m \
    --case=dycoms_rf01 \
    --nx=256 --ny=256 --nz=128 \
    --nt=7200 --dt=1 \
    --outdir=output_dycoms --outfreq=300 \
    --serial=false
```

Submit the job:

```bash
pjsub run_uwlcm.sh
```

## Performance Optimization for Fugaku

### OpenMP Settings

Fugaku's A64FX has 48 cores per node. Optimize OpenMP:

```bash
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

### Compiler Flags

The container already includes A64FX-optimized flags:
- `-march=armv8.2-a+sve`: Enables ARMv8.2-A with SVE (Scalable Vector Extension)
- `-mtune=a64fx`: Tunes for A64FX processor
- `-O3`: Aggressive optimization

### Memory Considerations

Fugaku nodes have 32 GB of memory. Be mindful of:
- Grid resolution (nx, ny, nz)
- Number of super-droplets (sd_conc for lgrngn)
- Output frequency

## Microphysics Backend Options

Since Fugaku doesn't have GPUs, use:

1. **Bulk microphysics**: `--micro=blk_1m` or `--micro=blk_2m` (fastest)
2. **Lagrangian with OpenMP**: `--micro=lgrngn --backend=OpenMP`
3. **Lagrangian serial**: `--micro=lgrngn --backend=serial`

**Note**: CUDA and multi_CUDA backends are not available.

## Troubleshooting

### Architecture Mismatch Error

If you see "Exec format error", the image was built for the wrong architecture. Rebuild for ARM64.

### Out of Memory

- Reduce grid resolution (nx, ny, nz)
- Reduce super-droplet concentration (sd_conc)
- Increase output frequency (outfreq)
- Use fewer OpenMP threads if memory-bound

### Slow Performance

- Ensure OpenMP is enabled (`--serial=false`)
- Set `OMP_NUM_THREADS=48`
- Use bulk microphysics instead of Lagrangian for faster runs
- Check that A64FX optimizations are enabled (should be by default in container)

## Additional Resources

- Fugaku User Portal: https://www.fugaku.r-ccs.riken.jp/
- Singularity Documentation: https://sylabs.io/docs/
- UWLCM Repository: https://github.com/igfuw/UWLCM
- libmpdata++: https://github.com/igfuw/libmpdataxx
- libcloudph++: https://github.com/igfuw/libcloudphxx

## Contact

For UWLCM-specific questions, consult the main repository.
For Fugaku-specific issues, contact the Fugaku support team.

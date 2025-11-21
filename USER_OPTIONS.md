# UWLCM Model User Options (file created by Claude 4.5)

This document describes the command-line options available for the UWLCM (University of Warsaw Large-Eddy Cloud Model).

## Required Options

### `--micro`
**Type:** string  
**Required:** Yes  
**Description:** Specifies the microphysics scheme to use.  
**Valid values:** `blk_1m`, `blk_2m`, `lgrngn`, `none`

- `blk_1m` - One-moment bulk microphysics
- `blk_2m` - Two-moment bulk microphysics
- `lgrngn` - Lagrangian super-droplet microphysics
- `none` - No microphysics (for dry simulations)

**Note:** Required for dry cases (`dry_thermal`, `dry_pbl`): must be set to `none`

## Simulation Setup Options

### `--case`
**Type:** string  
**Default:** `moist_thermal`  
**Description:** Specifies the simulation case to run.  
**Valid values:** `dry_thermal`, `moist_thermal`, `dycoms_rf01`, `dycoms_rf02`, `cumulus_congestus_icmw20`, `cumulus_congestus_icmw24`, `rico11`, `dry_pbl`

### Domain Size

#### `--X`
**Type:** real  
**Default:** `-1`  
**Unit:** meters  
**Description:** Domain size in X direction. Set negative to use case default.

#### `--Y`
**Type:** real  
**Default:** `-1`  
**Unit:** meters  
**Description:** Domain size in Y direction. Set negative to use case default.

#### `--Z`
**Type:** real  
**Default:** `-1`  
**Unit:** meters  
**Description:** Domain size in Z direction. Set negative to use case default.

### Grid Resolution

#### `--nx`
**Type:** integer  
**Default:** `1`  
**Description:** Number of grid cells in the horizontal X direction.

#### `--ny`
**Type:** integer  
**Default:** `0`  
**Description:** Number of grid cells in the horizontal Y direction. Set to `0` for 2D simulation.

#### `--nz`
**Type:** integer  
**Default:** `1`  
**Description:** Number of grid cells in the vertical Z direction.

### Time Integration

#### `--nt`
**Type:** integer  
**Default:** `0`  
**Description:** Total number of timesteps to simulate.

#### `--dt`
**Type:** real  
**Default:** `1`  
**Unit:** seconds  
**Description:** Length of each timestep.

#### `--spinup`
**Type:** integer  
**Default:** `0`  
**Description:** Number of initial timesteps during which rain formation is turned off.

## Output Options

### `--outdir`
**Type:** string  
**Default:** `""`  
**Required:** Yes (when not using `--help`)  
**Description:** Output directory name for netCDF-compatible HDF5 files.

### `--outfreq`
**Type:** integer  
**Default:** `0`  
**Required:** Yes (when not using `--help`)  
**Description:** Output rate as a timestep interval. Set to `0` to disable output.

### `--outstart`
**Type:** integer  
**Default:** `0`  
**Description:** Output starts after this many timesteps.

### `--outwindow`
**Type:** integer  
**Default:** `1`  
**Description:** Number of consecutive timesteps during which output is performed, starting at `outfreq`. Does not affect output of droplet spectra from Lagrangian microphysics.

## Random Number Generation

### `--rng_seed`
**Type:** integer  
**Default:** `0`  
**Description:** RNG seed for randomness post initialization (currently only affects Lagrangian microphysics). Set to `0` for a random seed.

### `--rng_seed_init`
**Type:** integer  
**Default:** `0`  
**Description:** RNG seed for initial conditions (perturbations of potential temperature and water vapor mixing ratio, and initialization of Lagrangian microphysics). Set to `0` to use `rng_seed` value.

## Performance Options

### `--serial`
**Type:** boolean  
**Default:** `false`  
**Description:** Force CPU components of the model (dynamics and bulk microphysics) to be computed on a single thread.

## Physical Model Options

### `--window`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable moving-window simulation where mean horizontal velocity is subtracted from advectors.

### `--piggy`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable piggybacking from a velocity field stored on disk.

### `--sgs`
**Type:** boolean  
**Default:** `false`  
**Description:** Turn Eulerian subgrid-scale (SGS) turbulence model on/off.

### `--sgs_delta`
**Type:** real  
**Default:** `-1`  
**Unit:** meters  
**Description:** Subgrid-scale turbulence model length scale. If negative, `sgs_delta = dz` (vertical grid spacing).

### `--relax_th_rv`
**Type:** boolean  
**Default:** `false`  
**Description:** Relax per-level mean potential temperature and water vapor mixing ratio to a desired (case-specific) profile.

## Aerosol Distribution Options

These options are only relevant for `lgrngn` and `blk_2m` microphysics schemes.

### Mode 1 Parameters

#### `--mean_rd1`
**Type:** real  
**Default:** `1.0e-6`  
**Unit:** meters  
**Description:** Mean radius for lognormal aerosol distribution mode 1.

#### `--sdev_rd1`
**Type:** real  
**Default:** `1.2`  
**Description:** Geometric standard deviation for lognormal aerosol distribution mode 1.

#### `--n1_stp`
**Type:** real  
**Default:** `-1.0`  
**Unit:** 1/m³  
**Description:** Aerosol concentration at STP for mode 1. If both `n1_stp < 0` and `n2_stp < 0`, case-specific aerosol distribution is used.

#### `--kappa1`
**Type:** real  
**Default:** `0.61`  
**Description:** Hygroscopicity parameter for aerosol distribution mode 1.

### Mode 2 Parameters

#### `--mean_rd2`
**Type:** real  
**Default:** `1.0e-6`  
**Unit:** meters  
**Description:** Mean radius for lognormal aerosol distribution mode 2.

#### `--sdev_rd2`
**Type:** real  
**Default:** `1.2`  
**Description:** Geometric standard deviation for lognormal aerosol distribution mode 2.

#### `--n2_stp`
**Type:** real  
**Default:** `-1.0`  
**Unit:** 1/m³  
**Description:** Aerosol concentration at STP for mode 2. If both `n1_stp < 0` and `n2_stp < 0`, case-specific aerosol distribution is used.

#### `--kappa2`
**Type:** real  
**Default:** `0.61`  
**Description:** Hygroscopicity parameter for aerosol distribution mode 2.

### `--case_n_stp_multiplier`
**Type:** real  
**Default:** `1.0`  
**Description:** If case-specific aerosol distribution is used, multiply the case-default aerosol concentration by this value.

## Microphysics-Specific Options

### Single-Moment Bulk Microphysics (`--micro=blk_1m`)

Based on the Kessler (1969) scheme for warm rain processes.

#### `--cond`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable cloud water condensation (1=on, 0=off).

#### `--cevp`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable cloud water evaporation (1=on, 0=off).

#### `--revp`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable rain water evaporation (1=on, 0=off).

#### `--conv`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable autoconversion of cloud water into rain (1=on, 0=off).

#### `--accr`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable cloud water collection by rain (accretion) (1=on, 0=off).

#### `--sedi`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable rain water sedimentation (1=on, 0=off).

#### `--r_c0`
**Type:** real  
**Default:** scheme-dependent  
**Unit:** kg/kg  
**Description:** Rain autoconversion threshold mixing ratio.

#### `--k_acnv`
**Type:** real  
**Default:** scheme-dependent  
**Description:** Rain autoconversion rate parameter.

#### `--r_eps`
**Type:** real  
**Default:** `1e-6`  
**Description:** Absolute tolerance for water mixing ratios.

**Output variables:** `rc` (cloud water), `rr` (rain water)

### Two-Moment Bulk Microphysics (`--micro=blk_2m`)

Based on the Morrison and Grabowski (2007) scheme.

#### `--acti`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable aerosol activation (on/off).

#### `--cond`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable condensation (on/off).

#### `--accr`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable accretion - collection of cloud water by rain (on/off).

#### `--acnv`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable autoconversion - conversion of cloud water to rain (on/off).

#### `--sedi`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable sedimentation (on/off).

#### `--acnv_A`
**Type:** real  
**Default:** scheme-dependent  
**Description:** Parameter A in autoconversion rate formulae.

#### `--acnv_b`
**Type:** real  
**Default:** scheme-dependent  
**Description:** Parameter b in autoconversion rate formulae.

#### `--acnv_c`
**Type:** real  
**Default:** scheme-dependent  
**Description:** Parameter c in autoconversion rate formulae.

**Output variables:** `rc` (cloud water), `rr` (rain water), `nc` (cloud droplet concentration), `nr` (rain drop concentration)

**Note:** This scheme uses the aerosol distribution parameters (`mean_rd1`, `sdev_rd1`, `n1_stp`, `kappa1`, etc.) for activation.

### Lagrangian Super-Droplet Microphysics (`--micro=lgrngn`)

Lagrangian particle-based microphysics scheme with detailed representation of aerosol activation, condensation, collision-coalescence, and sedimentation.

#### Required Options

##### `--backend`
**Type:** string  
**Required:** Yes  
**Description:** Computational backend to use.  
**Valid values:** `CUDA`, `multi_CUDA`, `OpenMP`, `serial`

- `CUDA` - Single GPU (NVIDIA CUDA)
- `multi_CUDA` - Multiple GPUs
- `OpenMP` - CPU with OpenMP parallelization
- `serial` - Single-threaded CPU execution

##### `--sd_conc`
**Type:** unsigned long long  
**Required:** Yes  
**Description:** Number of super-droplets per grid cell.

#### Physical Processes

##### `--adve`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable particle advection (1=on, 0=off).

##### `--sedi`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable particle sedimentation (1=on, 0=off).

##### `--cond`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable condensational growth (1=on, 0=off).

##### `--coal`
**Type:** boolean  
**Default:** `true`  
**Description:** Enable collisional growth (collision-coalescence) (1=on, 0=off).

##### `--rcyc`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable super-droplet recycling (1=on, 0=off).

##### `--chem_dsl`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable dissolving of trace gases (1=on, 0=off).

##### `--chem_dsc`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable dissociation (1=on, 0=off).

##### `--chem_rct`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable aqueous chemistry (1=on, 0=off).

#### Numerical Parameters

##### `--async`
**Type:** boolean  
**Default:** `true`  
**Description:** Use CPU for advection while GPU processes microphysics (ignored if backend != CUDA).

##### `--sd_const_multi`
**Type:** double  
**Default:** scheme-dependent  
**Description:** Multiplicity in constant multiplicity mode.

##### `--exact_sstp_cond`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Use exact (per-particle) logic for substeps in condensation.

##### `--adaptive_sstp_cond`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Use adaptive number of substeps for condensation.

##### `--sstp_cond_adapt_drw2_eps`
**Type:** real  
**Default:** scheme-dependent  
**Description:** Tolerance for adaptive substepping in condensation: `drw2_err <= eps * rw2`.

##### `--sstp_cond_adapt_drw2_max`
**Type:** real  
**Default:** scheme-dependent  
**Description:** Tolerance for adaptive substepping in condensation: `drw2 < max * rw2`.

##### `--sstp_cond_mix`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Mix potential temperature and water vapor changes after each condensation substep.

##### `--sstp_cond`
**Type:** integer  
**Default:** scheme-dependent  
**Description:** Number of substeps for condensation.

##### `--sstp_cond_act`
**Type:** integer  
**Default:** scheme-dependent  
**Description:** Number of substeps for condensation for droplets that activate.

##### `--sstp_coal`
**Type:** integer  
**Default:** scheme-dependent  
**Description:** Number of substeps for coalescence.

##### `--sstp_chem`
**Type:** integer  
**Default:** scheme-dependent  
**Description:** Number of substeps for chemistry.

#### Turbulence Effects

##### `--turb_cond`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Enable turbulence effects in super-droplet condensation (1=on, 0=off).

##### `--turb_adve`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Enable turbulence effects in super-droplet motion (1=on, 0=off).

##### `--turb_coal`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Enable turbulence effects in super-droplet coalescence (1=on, 0=off).

##### `--ReL`
**Type:** real  
**Default:** `100`  
**Description:** Taylor-microscale Reynolds number (used in Onishi kernel for turbulent collision enhancement).

#### Advection and Terminal Velocity

##### `--adve_scheme`
**Type:** string  
**Default:** `euler`  
**Description:** Advection scheme for super-droplets.  
**Valid values:** `euler`, `implicit`, `pred_corr`

##### `--term_vel`
**Type:** string  
**Default:** `beard77fast`  
**Description:** Terminal velocity formula.  
**Valid values:** `beard76`, `beard77fast`

##### `--coal_kernel`
**Type:** string  
**Default:** `hall_davis`  
**Description:** Collision kernel.  
**Valid values:** `hall`, `hall_davis`

#### Aerosol Initialization

##### `--rd_min`
**Type:** real  
**Default:** scheme-dependent  
**Unit:** meters  
**Description:** Minimum dry radius of initialized droplets. Negative means automatic detection.

##### `--rd_max`
**Type:** real  
**Default:** scheme-dependent  
**Unit:** meters  
**Description:** Maximum dry radius of initialized droplets. Negative means automatic detection. If `sd_conc_large_tail=true`, even larger droplets may be initialized.

##### `--sd_conc_large_tail`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Add super-droplets to better represent the large tail of the size distribution.

##### `--diag_incloud_time`
**Type:** boolean  
**Default:** scheme-dependent  
**Description:** Diagnose in-cloud time of droplets.

#### Giant CCN (GCCN)

##### `--gccn`
**Type:** real  
**Default:** `0`  
**Description:** Concentration of giant cloud condensation nuclei. Value represents a multiplier of VOCALS observations. Set to `0` to disable GCCN.

##### `--relax_ccn`
**Type:** boolean  
**Default:** `false`  
**Description:** Add CCN if per-level mean CCN concentration is lower than the (case-specific) desired concentration.

#### Output Options

##### `--out_dry`
**Type:** string  
**Default:** `""`  
**Description:** Dry radius ranges and moment numbers for output. Format: `r1:r2|n1,n2...;...`

##### `--out_wet`
**Type:** string  
**Default:** `""`  
**Description:** Wet radius ranges and moment numbers for output. Format: `r1:r2|n1,n2...;...`

##### `--out_dry_spec`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable output for plotting dry size spectrum.

##### `--out_wet_spec`
**Type:** boolean  
**Default:** `false`  
**Description:** Enable output for plotting wet size spectrum.

##### `--outfreq_spec`
**Type:** integer  
**Default:** `0`  
**Description:** Frequency (in timesteps) of spectrum output. Set to `0` to use `outfreq` value.

#### GPU/Device Options

##### `--dev_count`
**Type:** integer  
**Default:** `0`  
**Description:** Number of CUDA devices to use (for `multi_CUDA` backend).

##### `--dev_id`
**Type:** integer  
**Default:** `-1`  
**Description:** ID of CUDA device to use (for `CUDA` backend).

**Note:** This scheme uses the aerosol distribution parameters (`mean_rd1`, `sdev_rd1`, `n1_stp`, `kappa1`, etc.) for initialization.

### No Microphysics (`--micro=none`)

No microphysics options are available. This scheme is used for dry simulations (e.g., `dry_thermal`, `dry_pbl` cases).

## Help

### `--help`
**Description:** Display help message showing all available options. Use `--micro X --help` to see microphysics-specific options.

## Example Usage

```bash
# 2D moist thermal simulation with Lagrangian microphysics
./uwlcm --micro=lgrngn --backend=OpenMP --sd_conc=64 \
        --case=moist_thermal --nx=128 --ny=0 --nz=128 \
        --nt=3600 --dt=1 --outdir=output --outfreq=60

# 2D moist thermal with Lagrangian microphysics on GPU with custom options
./uwlcm --micro=lgrngn --backend=CUDA --sd_conc=128 \
        --case=moist_thermal --nx=256 --ny=0 --nz=256 \
        --nt=7200 --dt=1 --outdir=lgrngn_output --outfreq=120 \
        --coal=1 --sedi=1 --cond=1 --adve_scheme=pred_corr \
        --out_wet_spec=1 --outfreq_spec=600

# 3D DYCOMS simulation with two-moment bulk microphysics
./uwlcm --micro=blk_2m --case=dycoms_rf01 --nx=64 --ny=64 --nz=128 \
        --nt=7200 --dt=2 --outdir=dycoms_output --outfreq=300 \
        --acti=1 --cond=1 --accr=1 --acnv=1 --sedi=1

# Single-moment bulk microphysics with custom autoconversion threshold
./uwlcm --micro=blk_1m --case=moist_thermal --nx=128 --ny=0 --nz=128 \
        --nt=3600 --dt=1 --outdir=output --outfreq=60 \
        --r_c0=0.0005 --conv=1 --accr=1

# Dry thermal with no microphysics
./uwlcm --micro=none --case=dry_thermal --nx=128 --ny=0 --nz=128 \
        --nt=1800 --dt=1 --outdir=dry_output --outfreq=60

# Lagrangian microphysics with GCCN and custom aerosol distribution
./uwlcm --micro=lgrngn --backend=CUDA --sd_conc=128 \
        --case=dycoms_rf01 --nx=128 --ny=128 --nz=128 \
        --nt=10800 --dt=1 --outdir=gccn_output --outfreq=300 \
        --gccn=0.1 --mean_rd1=0.04e-6 --sdev_rd1=1.4 \
        --n1_stp=125e6 --kappa1=0.61

# Multi-GPU Lagrangian simulation
./uwlcm --micro=lgrngn --backend=multi_CUDA --dev_count=4 --sd_conc=256 \
        --case=rico11 --nx=256 --ny=256 --nz=256 \
        --nt=14400 --dt=1 --outdir=multi_gpu_output --outfreq=600 \
        --turb_coal=1 --turb_cond=1 --ReL=150
```

## Notes

- Some options may be disabled at compile time. Check compilation flags for availability.
- The `--piggy` option is incompatible with `--sgs`.
- For 2D simulations, set `--ny=0`.
- Output files are in netCDF-compatible HDF5 format.
- Each microphysics scheme outputs different variables. Check the output files to see which variables are available.
- For Lagrangian microphysics (`lgrngn`), both `--backend` and `--sd_conc` are required options.
- The `blk_1m` and `blk_2m` schemes have different process toggles and output variables.
- When using custom aerosol distributions, ensure that if two modes are specified, they have different `kappa` values.
- The `--outfreq_spec` option for Lagrangian microphysics must be a multiple of `--outfreq`.

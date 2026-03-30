# Data Access

All data used in this analysis are publicly available.

## Planck PR3

**Source**: [Planck Legacy Archive](https://pla.esac.esa.int/)

Required maps (Nside=64):
- `COM_CMB_IQU-smica_2048_R3.00_full.fits` → downgrade to Nside=64
- `COM_CMB_IQU-commander_2048_R3.00_full.fits`
- `COM_CMB_IQU-nilc_2048_R3.00_full.fits`

**Reference**: Planck Collaboration (2020), A&A 641, A4

## CosmicFlows-4

**Source**: [NASA/IPAC Extragalactic Database](https://irsa.ipac.caltech.edu/)

Required file:
- CF4 peculiar velocity grid (Cartesian, 3.68 Mpc voxel⁻¹)

**Reference**: Courtois et al. (2023), A&A 670, L15

## No Data Files in Repository

Raw data files are not included in this repository due to size constraints.
All analysis code reads from locally downloaded data files.

See `code/cf4_vorticity_pipeline.py --help` for data format details.

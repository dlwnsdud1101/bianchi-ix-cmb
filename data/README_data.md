# Data Access Instructions

All data used in this analysis are **publicly available**. No proprietary data.

---

## CosmicFlows-4 (CF4) Velocity Field

**File**: `CF4gp_new_64-z008_velocity.fits`  
**Source**: [cosmicflows.iap.fr](https://cosmicflows.iap.fr)  
**Reference**: Courtois et al. (2023), *A&A* 670, L15  
**DOI**: 10.1051/0004-6361/202245331

Format:
- FITS cube, shape `[3, 64, 64, 64]`
- Components: `(vx, vy, vz)` in Supergalactic Cartesian coordinates
- Voxel scale: 3.68 Mpc/voxel
- Velocity unit: raw × 52 = km/s

Download:
```bash
wget https://cosmicflows.iap.fr/CF4gp_new_64-z008_velocity.fits
```

---

## Planck 2018 PR3 CMB Maps

**Source**: [Planck Legacy Archive](https://pla.esac.esa.int)

| Pipeline  | File |
|-----------|------|
| SMICA     | `COM_CMB_IQU-smica_2048_R3.00_full.fits` |
| Commander | `COM_CMB_IQU-commander_2048_R3.00_full.fits` |
| NILC      | `COM_CMB_IQU-nilc_2048_R3.00_full.fits` |

**Mask (recommended)**: Planck UT78  
`COM_Mask_CMB-common-Mask-lowl-field-Int_2048_R3.00.fits`

Download via PLA portal or:
```bash
# Example (SMICA)
wget -O COM_CMB_IQU-smica_2048_R3.00_full.fits \
  "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full"
```

**Reference**: Planck Collaboration XVI (2016), *A&A* 594, A16  
**Reference**: Planck Collaboration VII (2020), *A&A* 641, A7

---

## Data Volume

| Dataset | Size |
|---------|------|
| CF4 velocity FITS | ~4 MB |
| Planck SMICA (Nside=2048) | ~200 MB per pipeline |
| Planck mask | ~50 MB |

For the vorticity and Planck analyses, maps are downgraded to Nside=64 in
memory, so RAM requirement is minimal (< 1 GB).

---

## No Data Required

`cf4_vorticity_pipeline.py --validate-only` runs all unit tests without
any downloaded data. All Monte Carlo scripts (`monte_carlo_alignment.py`,
`isw_sensitivity.py`) are fully self-contained.

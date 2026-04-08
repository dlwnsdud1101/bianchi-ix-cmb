# A Multi-tracer Axis Alignment between the CMB Quadrupole and Large-Scale Velocity Vorticity

**Lee Junyoung** | March 2026

> Independent Convergence on the South Galactic Pole from Planck PR3 and CosmicFlows-4

---

## Summary

This repository contains the analysis code and reproducibility materials for:

**"A Multi-tracer Axis Alignment between the CMB Quadrupole and Large-Scale Velocity Vorticity: Independent Convergence on the South Galactic Pole from Planck PR3 and CosmicFlows-4"**

### Key Result

The CMB quadrupole symmetry axis (Planck PR3) and the large-scale peculiar-velocity vorticity (CosmicFlows-4) independently converge within 5°–9° of the South Galactic Pole (SGP), with a joint Monte Carlo probability of **P = 0.002% (4.1σ)**. The two probes span a redshift factor of ~55 (z ≈ 1100 vs z ≈ 0.02) and share no common calibration, observational method, or physical epoch.

---

## Repository Structure

```
bianchi-ix-cmb/
├── README.md
├── code/
│   ├── cf4_vorticity_pipeline.py     # CF4 vorticity axis analysis (main pipeline)
│   ├── cf4_vorticity_analysis.py     # CF4 vorticity utility functions
│   ├── monte_carlo_alignment.py      # Multi-tracer Monte Carlo significance tests
│   ├── isw_sensitivity.py            # ISW cross-correlation sensitivity sweep
│   ├── bmode_prediction.py           # B-mode upper bound (Eq. 5)
│   ├── tidal_tensor_analysis.py      # GA tidal tensor eigenvector calculation
│   ├── planck_direct_test.py         # Planck PR3 quadrupole axis extraction
│   └── bianchi_ix_camb.py            # CAMB spectrum and transfer function
├── figures/
│   └── isw_sensitivity.pdf           # Figure 1: ISW sensitivity analysis
└── data/
    └── README_data.md                # Data access instructions
```

---

## Reproducibility

All results are fully reproducible from public data.

### Data Sources

- **Planck PR3**: [Planck Legacy Archive](https://pla.esac.esa.int/)
  - SMICA, Commander, NILC maps (Nside=64)
  - Mask: UT78 (`COM_Mask_CMB-common-Mask-lowl-field-Int_2048_R3.00.fits`)
- **CosmicFlows-4**: [cosmicflows.iap.fr](https://cosmicflows.iap.fr)
  - Courtois et al. (2023), A&A 670, L15
  - Velocity grid: 3.68 Mpc voxel⁻¹

### Quick Start

```bash
git clone https://github.com/dlwnsdud1101/bianchi-ix-cmb
cd bianchi-ix-cmb
pip install numpy scipy astropy healpy camb matplotlib

# Run validation suite (no data required)
python code/cf4_vorticity_pipeline.py --validate-only

# Run full CF4 vorticity analysis (requires CF4 velocity grid)
python code/cf4_vorticity_pipeline.py --data CF4_vel_field.fits --output results/

# Run Monte Carlo significance tests
python code/monte_carlo_alignment.py

# Quick sanity check (N=100,000, ~5 s)
python code/monte_carlo_alignment.py --quick

# Run ISW sensitivity sweep
python code/isw_sensitivity.py
```

### Requirements

```
numpy >= 1.24
scipy >= 1.10
astropy >= 5.3
healpy >= 1.16
camb >= 1.5
matplotlib >= 3.7
```

---

## Analysis Code Overview

### `cf4_vorticity_pipeline.py`

Main pipeline for computing the vorticity axis from the CF4 velocity grid.

- `np.gradient`-based ∇×v (no periodic boundary conditions)
- Gaussian pre-smoothing (σ = 1.5 voxels) before differentiation
- Spherical extraction at six pre-specified radii (R = 50–75 Mpc)
- Look-elsewhere Bonferroni correction (×6 scales)
- Built-in validation suite (4 unit tests, verified to < 0.01° accuracy)

### `monte_carlo_alignment.py`

Monte Carlo significance tests for the multi-tracer alignment.

- N = 5,000,000 trials, seed = 2026 (fixed prior to data inspection)
- Two-condition test: P(CMB↔SGP < 6°, CF4↔SGP < 5°) = 0.00196% (4.11σ)
- Bonferroni-corrected (×6 scales): 0.0118% (3.68σ)
- ISW cross-correlation sensitivity: r ∈ [0, 0.41]

### `isw_sensitivity.py`

ISW cross-correlation sensitivity analysis.

- Correlated Monte Carlo across r ∈ {0.00, 0.05, …, 0.41} (N = 2,000,000 per value)
- Conservative model: maximum ISW correlation upper bound
- Generates Figure 1 (ISW sensitivity plot)

### `bmode_prediction.py`

B-mode polarisation upper bound under LRS Bianchi IX.

- Reproduces Eq. 5: |a^B_{2,0}|² ≤ 0.034 μK² (≤ 183 nK)
- Ωω cancellation demonstrated explicitly
- CMB-S4 detection probability estimate (20–40% standard; 50–70% axis-constrained)

### `tidal_tensor_analysis.py`

Great Attractor tidal tensor analysis.

- Eigendecomposition of T_ij = −GM/r³(δ_ij − 3n̂_in̂_j)
- Doubly degenerate eigenspace identification
- SGP proximity of equilibrium eigenvector: 11.2°
- Robustness: N = 100,000 random GA positions (~5% achieve < 11.2°)

### `planck_direct_test.py`

Planck PR3 quadrupole axis extraction.

- Supports SMICA, Commander, NILC pipelines
- Planck UT78 mask (falls back to |b| > 20° with warning if unavailable)
- C₂^ΛCDM computed via CAMB (no hardcoded values)

### `bianchi_ix_camb.py`

CAMB-based spectrum computation and forward prediction.

- Forward: Ωω → predicted |a₂₀| (no circular SNR_obs input)
- Even-ℓ (ℓ = 2, 4) spectrum modification under LRS axial symmetry
- Four-method T'₂ convergence verification (< 2% spread)

---

## Key Numbers (for verification)

| Quantity | Value |
|----------|-------|
| P(CMB↔SGP < 6°, CF4↔SGP < 5°) | 0.00196% (4.11σ) |
| P after Bonferroni (×6 scales) | 0.0118% (3.68σ) |
| ISW sensitivity range | 3.6σ–4.1σ (r ∈ [0, 0.41]) |
| CF4 vorticity ↔ SGP (R=70 Mpc) | 4.5° |
| CMB quadrupole ↔ SGP (SMICA) | 6°–9° |
| B-mode prediction (Bianchi IX) | ≤ 0.034 μK² (≤ 183 nK) |
| SGP ↔ thermal gradient | 62.0° |
| GA tidal equilibrium ↔ SGP | 11.2° |
| Collins–Hawking range percentile | 80th (log scale) |

---

## Citation

If you use this code or analysis, please cite:

```
Lee, J. (2026). A Multi-tracer Axis Alignment between the CMB Quadrupole
and Large-Scale Velocity Vorticity: Independent Convergence on the South
Galactic Pole from Planck PR3 and CosmicFlows-4.
Submitted to Astronomy & Astrophysics.
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Lee Junyoung — dlwnsdud1101@naver.com

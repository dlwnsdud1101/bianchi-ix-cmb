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
│   ├── monte_carlo_alignment.py      # Multi-tracer Monte Carlo significance tests
│   ├── isw_sensitivity.py            # ISW cross-correlation sensitivity sweep
│   ├── bmode_prediction.py           # B-mode transfer function and prediction
│   └── tidal_tensor_analysis.py      # GA tidal tensor eigenvector calculation
├── paper/
│   └── bianchi_ix_v15.tex            # LaTeX source (submitted version)
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
- **CosmicFlows-4**: [IRSA/NED](https://irsa.ipac.caltech.edu/)
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
- Central finite-difference ∇×v computation
- Spherical extraction at six pre-specified radii (R = 50–75 Mpc)
- Look-elsewhere Bonferroni correction (N=2000 bootstrap)
- Built-in validation suite (4 unit tests, verified to <0.01° accuracy)

### `monte_carlo_alignment.py`
Monte Carlo significance tests for the multi-tracer alignment.
- N = 5,000,000 trials, seed = 2026 (fixed for reproducibility)
- Two-condition test: P(CMB↔SGP < 6°, CF4↔SGP < 5°) = 0.00196% (4.11σ)
- ISW cross-correlation r sweep: r ∈ [0, 0.41]
- Three-condition fallback (geometric equivalence verification)

### `isw_sensitivity.py`
ISW cross-correlation sensitivity analysis.
- Correlated Monte Carlo across r ∈ {0.00, 0.05, ..., 0.41}
- Generates Figure 1 (ISW sensitivity plot)

### `bmode_prediction.py`
B-mode polarisation prediction under LRS Bianchi IX.
- Derivation of |a^B_{2,0}|² from first principles
- Ωω cancellation demonstration
- CAMB tensor transfer function computation

### `tidal_tensor_analysis.py`
Great Attractor tidal tensor analysis.
- Eigendecomposition of T_ij = −GM/r³(δ_ij − 3n̂_in̂_j)
- Doubly degenerate eigenspace identification
- SGP proximity of equilibrium eigenvector (11.2°)

---

## Key Numbers (for verification)

| Quantity | Value |
|---|---|
| P(CMB↔SGP < 6°, CF4↔SGP < 5°) | 0.00196% (4.11σ) |
| P after Bonferroni (×6 scales) | 0.0118% (3.68σ) |
| ISW sensitivity range | 3.6σ–4.1σ (r ∈ [0, 0.41]) |
| CF4 vorticity ↔ SGP (R=70 Mpc) | 4.5° |
| CMB quadrupole ↔ SGP (SMICA) | 6°–9° |
| B-mode prediction (Bianchi IX) | ≤ 0.034 μK² (≤183 nK) |
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

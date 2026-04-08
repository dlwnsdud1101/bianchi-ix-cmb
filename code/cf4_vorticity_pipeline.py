"""
CosmicFlows-4 Vorticity Pipeline
==================================
Lee Junyoung | dlwnsdud1101@naver.com

Main pipeline for CF4 vorticity axis analysis.
Includes --validate-only mode (no data required) with 4 unit tests.

Usage:
    python cf4_vorticity_pipeline.py --validate-only
    python cf4_vorticity_pipeline.py --data CF4_vel_field.fits --output results/

Requirements:
    pip install astropy numpy scipy matplotlib
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import os
import sys

# ── Coordinate utilities ──────────────────────────────────────────────────────
def g2c(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])

def c2g(v):
    v = v / np.linalg.norm(v)
    b = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    l = np.degrees(np.arctan2(v[1], v[0])) % 360
    return l, b

def sg_to_gal(v_sg):
    e_sgz = g2c(47.37,  6.32)
    e_sgx = g2c(137.37, 0.0)
    e_sgy = np.cross(e_sgz, e_sgx)
    e_sgy /= np.linalg.norm(e_sgy)
    return np.column_stack([e_sgx, e_sgy, e_sgz]) @ v_sg

def angle_between(v1, v2):
    c = abs(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))

n_sgp  = g2c(0,   -90)
n_zrot = g2c(0,   -84)

# ── Vorticity computation ─────────────────────────────────────────────────────
def compute_vorticity(vx, vy, vz, dx=1.0, smooth_sigma=1.5):
    """curl ω = ∇×v via np.gradient (no periodic wrap)."""
    if smooth_sigma > 0:
        vx = gaussian_filter(vx, smooth_sigma, mode='nearest')
        vy = gaussian_filter(vy, smooth_sigma, mode='nearest')
        vz = gaussian_filter(vz, smooth_sigma, mode='nearest')
    dvz_dy = np.gradient(vz, dx, axis=1)
    dvy_dz = np.gradient(vy, dx, axis=0)
    dvx_dz = np.gradient(vx, dx, axis=0)
    dvz_dx = np.gradient(vz, dx, axis=2)
    dvy_dx = np.gradient(vy, dx, axis=2)
    dvx_dy = np.gradient(vx, dx, axis=1)
    return (dvz_dy - dvy_dz,
            dvx_dz - dvz_dx,
            dvy_dx - dvx_dy)

def analyze_scale(ox, oy, oz, R_voxels, cx=32, cy=32, cz=32):
    """Mean vorticity direction within sphere of radius R_voxels."""
    N = ox.shape[0]
    d = np.arange(-R_voxels, R_voxels + 1)
    dz, dy, dx = np.meshgrid(d, d, d, indexing='ij')
    sphere    = (dz**2 + dy**2 + dx**2) <= R_voxels**2
    pz, py, px = cz+dz, cy+dy, cx+dx
    in_bounds = ((pz>=0)&(pz<N)&(py>=0)&(py<N)&(px>=0)&(px<N))
    valid     = sphere & in_bounds
    if not valid.any():
        return None, None
    pz_v, py_v, px_v = pz[valid], py[valid], px[valid]
    w = np.array([ox[pz_v,py_v,px_v].sum(),
                  oy[pz_v,py_v,px_v].sum(),
                  oz[pz_v,py_v,px_v].sum()])
    if np.linalg.norm(w) < 1e-10:
        return None, None
    w_gal = sg_to_gal(w / np.linalg.norm(w))
    if angle_between(w_gal, n_sgp) > angle_between(-w_gal, n_sgp):
        w_gal = -w_gal
    return angle_between(w_gal, n_sgp), angle_between(w_gal, n_zrot)

# ── Unit tests (--validate-only) ──────────────────────────────────────────────
def run_validation():
    """4 unit tests verifiable without CF4 data."""
    print("Running validation suite (4 unit tests)...")
    passed = 0

    # Test 1: coordinate round-trip accuracy < 0.01°
    for l, b in [(0,-90),(137.37,0),(47.37,6.32),(307,9)]:
        v = g2c(l, b)
        l2, b2 = c2g(v)
        err = max(abs(l-l2)%360, abs(b-b2))
        assert err < 0.01, f"Round-trip error {err:.4f}° at (l={l},b={b})"
    print("  [1/4] Coordinate round-trip accuracy < 0.01°  ✓")
    passed += 1

    # Test 2: SGP is correctly identified
    sgp_check = g2c(0, -90)
    assert abs(sgp_check[2] - (-1.0)) < 1e-10, "SGP z-component should be -1"
    assert angle_between(sgp_check, n_sgp) < 1e-8
    print("  [2/4] SGP direction correct  ✓")
    passed += 1

    # Test 3: vorticity of a pure-rotation field = 2ω (analytical check)
    # v = ω × r  for solid-body rotation about z-axis → curl = (0,0,2)
    N = 32
    xs = np.linspace(-1, 1, N)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    omega_z = 1.0
    vx = -omega_z * Y
    vy =  omega_z * X
    vz = np.zeros_like(X)
    dx = xs[1] - xs[0]
    ox, oy, oz = compute_vorticity(vx, vy, vz, dx=dx, smooth_sigma=0)
    # Interior voxels should give curl_z ≈ 2*omega_z
    interior = slice(2, -2)
    err = abs(oz[interior,interior,interior].mean() - 2*omega_z)
    assert err < 0.05, f"Curl error: {err:.4f}  (expected 0)"
    print(f"  [3/4] Solid-body rotation curl error: {err:.5f}  ✓")
    passed += 1

    # Test 4: Monte Carlo baseline (isotropic → P(angle<5°) ≈ 1-cos5° ≈ 0.38%)
    rng = np.random.default_rng(42)
    v = rng.standard_normal((200_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.abs(v @ n_sgp), 0, 1)))
    p_emp = (angles < 5.0).mean()
    p_theory = 1 - np.cos(np.radians(5.0))
    err = abs(p_emp - p_theory)
    assert err < 0.001, f"MC baseline error: {err:.4f}"
    print(f"  [4/4] MC isotropic baseline  P(θ<5°) = {p_emp*100:.3f}%  "
          f"(theory {p_theory*100:.3f}%)  ✓")
    passed += 1

    print(f"\nAll {passed}/4 unit tests passed ✓")
    return True

# ── Data pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(data_path, output_dir='.'):
    from astropy.io import fits

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading: {data_path}")
    hdul = fits.open(data_path)
    data = hdul[0].data
    vx = data[0] * 52.0
    vy = data[1] * 52.0
    vz = data[2] * 52.0
    hdul.close()
    print(f"  Grid: {vx.shape}  (3.68 Mpc/voxel)")

    print("Computing vorticity (σ=1.5 vox Gaussian, np.gradient)...")
    ox, oy, oz = compute_vorticity(vx, vy, vz)

    scales = [(10,50),(11,55),(12,60),(13,65),(14,70),(15,75)]
    print(f"\n{'R (vox)':>8} {'R (Mpc)':>8} {'ω↔SGP':>8} {'ω↔ẑ_rot':>9}")
    print("-" * 38)

    results = {}
    for R, R_Mpc in scales:
        sep_sgp, sep_zrot = analyze_scale(ox, oy, oz, R)
        if sep_sgp is not None:
            results[(R, R_Mpc)] = (sep_sgp, sep_zrot)
            flag = ' ← paper result' if R == 14 else ''
            print(f"  R={R:2d}   {R_Mpc:5d}    {sep_sgp:6.1f}°   {sep_zrot:6.1f}°{flag}")

    # Save
    out_file = os.path.join(output_dir, 'vorticity_results.txt')
    with open(out_file, 'w') as f:
        f.write("R_vox  R_Mpc  omega_SGP_deg  omega_zrot_deg\n")
        for (R, Rm), (s, z) in results.items():
            f.write(f"{R}  {Rm}  {s:.4f}  {z:.4f}\n")
    print(f"\nResults saved to {out_file}")

    print("\nPaper result at R=14 (70 Mpc):")
    print("  ω ↔ SGP   = 4.5°")
    print("  ω ↔ ẑ_rot = 1.7°")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CF4 vorticity pipeline')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run unit tests only (no data required)')
    parser.add_argument('--data',   type=str, default=None,
                        help='Path to CF4 velocity FITS file')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    args = parser.parse_args()

    if args.validate_only:
        run_validation()
    elif args.data:
        run_validation()
        print()
        run_pipeline(args.data, args.output)
    else:
        parser.print_help()
        print("\nTip: run --validate-only first, then --data <file>")

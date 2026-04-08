"""
Direct Planck PR3 Test
=======================
Lee Junyoung | dlwnsdud1101@naver.com

Rotates Planck 2018 PR3 CMB maps to ẑ_rot = (0°, -84°) and
measures |a_{2,0}|² and SNR_obs across three pipelines.

Fix log
-------
v1: C₂ CAMB 계산, UNSEEN→0 마스크 수정
v2: Planck UT78 공식 마스크 사용 (mode coupling 감소)
    세 파이프라인 (SMICA, Commander, NILC) 모두 실행
    inpainting 미적용 주의사항 명시

Data required
-------------
  Maps  : COM_CMB_IQU-{smica,commander,nilc}_2048_R3.00_full.fits
  Mask  : COM_Mask_CMB-common-Mask-lowl-field-Int_2048_R3.00.fits  (UT78)
  All available at https://pla.esac.esa.int

Requirements:
    pip install healpy numpy camb
"""

import numpy as np
import healpy as hp
import camb
import os

# ── Planck 2018 best-fit C₂^ΛCDM via CAMB ────────────────────────────────────
def get_C2_lcdm():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200,
                       mnu=0.06, omk=0, tau=0.0544)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars.set_for_lmax(4, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers  = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    D2 = powers['total'][2, 0]
    return D2 * 2 * np.pi / (2 * 3)   # D_ℓ → C_ℓ

C2_lcdm = get_C2_lcdm()
print(f"C₂^ΛCDM (CAMB) = {C2_lcdm:.2f} μK²")

# ── Pipeline file names ───────────────────────────────────────────────────────
PIPELINES = {
    'SMICA'    : 'COM_CMB_IQU-smica_2048_R3.00_full.fits',
    'Commander': 'COM_CMB_IQU-commander_2048_R3.00_full.fits',
    'NILC'     : 'COM_CMB_IQU-nilc_2048_R3.00_full.fits',
}
MASK_FILE = 'COM_Mask_CMB-common-Mask-lowl-field-Int_2048_R3.00.fits'

# ── Rotation ──────────────────────────────────────────────────────────────────
def rotate_to_zrot(alm, l_zrot=0, b_zrot=-84):
    """Rotate alm so ẑ_rot = (l=0°, b=-84°) maps to north pole."""
    rotator = hp.Rotator(rot=[np.radians(l_zrot),
                               np.radians(90 - b_zrot), 0.0], deg=False)
    return rotator.rotate_alm(alm)

# ── Mask loading ──────────────────────────────────────────────────────────────
def load_mask(mask_file, nside):
    """Load Planck UT78 mask (preferred) or fall back to |b|>20° cut.

    The UT78 mask removes the galactic plane and point sources with
    ~78% sky fraction, minimising mode coupling at low ℓ. A simple
    |b|>20° cut retains ~66% sky but couples ℓ modes and can bias a₂₀.
    """
    if os.path.exists(mask_file):
        print(f"  Using Planck UT78 mask: {mask_file}")
        m = hp.read_map(mask_file, verbose=False)
        return hp.ud_grade(m, nside) > 0.5   # binary after degrading
    else:
        print(f"  WARNING: UT78 mask not found. Falling back to |b|>20° cut.")
        print(f"           This introduces mode coupling and may bias a₂₀.")
        print(f"           Download from: https://pla.esac.esa.int")
        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        b_gal = np.degrees(np.pi / 2 - theta)
        return np.abs(b_gal) > 20

# ── Analysis ──────────────────────────────────────────────────────────────────
def compute_a20_power(map_file, mask, nside=64, lmax=4):
    """
    Load map, apply mask, rotate to ẑ_rot frame, extract |a_{2,0}|².

    Note on inpainting: we zero-fill masked pixels (no inpainting).
    This causes power leakage into low-ℓ modes. For a fully rigorous
    analysis, use Wiener-filter inpainting (e.g. healpy.smoothing +
    constrained realisation). The effect on a₂₀ direction is small
    but can shift |a₂₀|² by ~5–10%.
    """
    m_full = hp.read_map(map_file, field=0, verbose=False)
    m = hp.ud_grade(m_full, nside)

    # Apply mask: zero-fill (not UNSEEN) so map2alm gets no spurious power
    m_masked = m.copy()
    m_masked[~mask] = 0.0

    alm     = hp.map2alm(m_masked, lmax=lmax)
    alm_rot = rotate_to_zrot(alm)

    idx_20 = hp.Alm.getidx(lmax, 2, 0)
    a20    = alm_rot[idx_20].real   # μK
    a20_sq = a20 ** 2               # μK²

    noise  = np.sqrt(C2_lcdm / 5)
    snr    = abs(a20) / noise

    return a20_sq, snr

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    nside = 64
    mask  = load_mask(MASK_FILE, nside)
    print(f"  Sky fraction: {mask.mean():.3f}  ({mask.mean()*100:.1f}%)")
    print()

    # Override with command-line map files if provided
    if len(sys.argv) > 1:
        # Single map mode: python planck_direct_test.py SMICA.fits [Commander.fits NILC.fits]
        names = ['SMICA', 'Commander', 'NILC']
        files = sys.argv[1:]
        run_list = list(zip(names[:len(files)], files))
    else:
        # Check which default files exist
        run_list = [(name, f) for name, f in PIPELINES.items() if os.path.exists(f)]
        if not run_list:
            print("No Planck map files found. Expected filenames:")
            for name, f in PIPELINES.items():
                print(f"  {name}: {f}")
            print("\nExpected results (from paper):")
            print(f"  {'Pipeline':<12} {'|a₂₀|² (μK²)':>14} {'SNR_obs':>10}")
            print(f"  {'SMICA':<12} {'431.8':>14} {'0.92':>10}")
            print(f"  {'Commander':<12} {'407.5':>14} {'0.90':>10}")
            print(f"  {'NILC':<12} {'407.5':>14} {'0.90':>10}")
            raise SystemExit(0)

    print(f"  {'Pipeline':<12} {'|a₂₀|² (μK²)':>14} {'SNR_obs':>10} "
          f"{'∈[312,1122]':>12} {'SNR∈[0.64,2]':>13}")
    print("  " + "-" * 65)

    results_all = {}
    for name, fpath in run_list:
        print(f"  Processing {name}: {fpath}")
        a20_sq, snr = compute_a20_power(fpath, mask, nside)
        in_range    = 312 <= a20_sq <= 1122
        snr_ok      = 0.64 <= snr <= 2.0
        results_all[name] = (a20_sq, snr)
        print(f"  {name:<12} {a20_sq:>14.1f} {snr:>10.3f} "
              f"{'✓' if in_range else '✗':>12} {'✓' if snr_ok else '✗':>13}")

    if len(results_all) >= 2:
        vals = [v[0] for v in results_all.values()]
        print(f"\n  Pipeline spread: {max(vals)-min(vals):.1f} μK²  "
              f"(paper: [{min(vals):.0f}, {max(vals):.0f}])")

    print()
    print("Note: inpainting not applied. Masked-pixel zeroing may shift")
    print("|a₂₀|² by ~5–10%; axis direction robust to ~1°. (Sect. 2.2)")

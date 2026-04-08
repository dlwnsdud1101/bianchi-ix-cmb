"""
ISW Cross-Correlation Sensitivity Analysis
==========================================
Lee Junyoung | dlwnsdud1101@naver.com

Reproduces Figure 1: joint probability and significance vs ISW
cross-correlation coefficient r ∈ [0, 0.41].

Francis & Peacock (2010) upper bound: r ≤ 0.41.
All variants remain > 3.6σ.

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ────────────────────────────────────────────────────────────────
N_MC        = 2_000_000   # per r value (matches paper caption)
SEED_BASE   = 2026
BONFERRONI  = 6
R_VALUES    = np.arange(0.0, 0.42, 0.05)   # r ∈ [0, 0.41]
R_UPPER     = 0.41        # Francis & Peacock (2010) bound

THETA_CMB   = 6.0         # degrees
THETA_CF4   = 5.0         # degrees

def g2c(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])

n_sgp = g2c(0, -90)

def p_to_sigma(p):
    return -stats.norm.ppf(max(p, 1e-12))

# ── ISW-correlated MC ────────────────────────────────────────────────────────
def mc_isw(r, N=N_MC, seed=SEED_BASE):
    """
    ISW-correlated Monte Carlo.

    Model: both CMB and CF4 axes share a common "ISW direction" component
    with weight r, plus independent random components with weight √(1-r²).

        cmb_axis = √(1-r²) · u_cmb + r · u_isw
        cf4_axis = √(1-r²) · u_cf4 + r · u_isw

    This is the most conservative (maximum correlation) model.
    r=0 recovers the fully independent case.
    """
    rng = np.random.default_rng(seed)

    def rand_unit(n):
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    u_isw = rand_unit(N)
    u_cmb = rand_unit(N)
    u_cf4 = rand_unit(N)

    w  = np.sqrt(max(1 - r**2, 0))
    v_cmb = w * u_cmb + r * u_isw
    v_cf4 = w * u_cf4 + r * u_isw
    v_cmb /= np.linalg.norm(v_cmb, axis=1, keepdims=True)
    v_cf4 /= np.linalg.norm(v_cf4, axis=1, keepdims=True)

    ang_cmb = np.degrees(np.arccos(np.clip(np.abs(v_cmb @ n_sgp), 0, 1)))
    ang_cf4 = np.degrees(np.arccos(np.clip(np.abs(v_cf4 @ n_sgp), 0, 1)))

    joint = ((ang_cmb < THETA_CMB) & (ang_cf4 < THETA_CF4)).mean()
    return joint

# ── Run sweep ────────────────────────────────────────────────────────────────
print(f"ISW sensitivity sweep  (N={N_MC:,} per value)")
print(f"r range: [0, {R_UPPER}]  (Francis & Peacock 2010 bound)")
print("─" * 52)

p_raw_list  = []
p_bonf_list = []
sig_raw     = []
sig_bonf    = []

for r in R_VALUES:
    p = mc_isw(r)
    pb = min(p * BONFERRONI, 1.0)
    p_raw_list.append(p * 100)
    p_bonf_list.append(pb * 100)
    sig_raw.append(p_to_sigma(p))
    sig_bonf.append(p_to_sigma(pb))
    print(f"  r={r:.2f}  P_raw={p*100:.5f}%  P_Bonf={pb*100:.5f}%  "
          f"σ_raw={p_to_sigma(p):.2f}  σ_Bonf={p_to_sigma(pb):.2f}")

# Paper reference values
paper_raw  = 0.00196
paper_bonf = 0.01176

# ── Plot (Figure 1) ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Left: probability vs r
ax1 = fig.add_subplot(gs[0])
ax1.plot(R_VALUES, p_raw_list,  'b-o', ms=5, label='Raw P (4-cond)')
ax1.plot(R_VALUES, p_bonf_list, 'r-s', ms=5, label='Bonferroni-corrected (×6)')
ax1.axhline(paper_raw,  ls='--', color='blue',  alpha=0.6,
            label=f'Paper value ({paper_raw:.4f}%)')
ax1.axhline(paper_bonf, ls='--', color='red',   alpha=0.6,
            label=f'ISW-corrected ({paper_bonf:.4f}%)')
ax1.axvline(R_UPPER, ls=':', color='gray', label=f'r upper bound ({R_UPPER})')
ax1.set_xlabel('ISW cross-correlation coefficient r')
ax1.set_ylabel('Joint probability P (%)')
ax1.set_title('Sensitivity to ISW cross-correlation')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# Right: significance vs r
ax2 = fig.add_subplot(gs[1])
ax2.plot(R_VALUES, sig_raw,   'b-o', ms=5, label='Raw')
ax2.plot(R_VALUES, sig_bonf,  'r-s', ms=5, label='Bonferroni-corrected')
ax2.axhline(3.0, ls='--', color='orange', alpha=0.7, label='3σ threshold')
ax2.axhline(2.0, ls='--', color='green',  alpha=0.7, label='2σ threshold')
ax2.axvline(R_UPPER, ls=':', color='gray')
ax2.fill_between(R_VALUES, 3.6, max(sig_raw)+0.2,
                 alpha=0.08, color='blue', label='> 3.6σ zone')
ax2.set_xlabel('ISW cross-correlation coefficient r')
ax2.set_ylabel('Significance (σ)')
ax2.set_title('Statistical significance vs r')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.suptitle(f'ISW Sensitivity Analysis  (N={N_MC:,} per value)',
             fontsize=11, y=1.01)
plt.savefig('isw_sensitivity.pdf', bbox_inches='tight', dpi=150)
plt.savefig('isw_sensitivity.png', bbox_inches='tight', dpi=150)
plt.show()
print("\nSaved: isw_sensitivity.pdf, isw_sensitivity.png")
print(f"\nMin σ_Bonf across r ∈ [0, {R_UPPER}]: {min(sig_bonf):.2f}σ  "
      f"({'> 3.6σ ✓' if min(sig_bonf) > 3.6 else '< 3.6σ ✗'})")

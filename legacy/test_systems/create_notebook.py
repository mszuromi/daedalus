import json

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": []
}

# ── Cell 0: title markdown ─────────────────────────────────────────────────────
nb["cells"].append({
    "cell_type": "markdown",
    "id": "cell-md-title",
    "metadata": {},
    "source": [
        "# 2-Neuron Nonlinear Hawkes: Parameter Search for 3–15% Rate Mismatch\n",
        "\n",
        "Finds parameter regimes where the mean firing rates differ from the mean-field\n",
        "prediction by 3–15%, driven by the curvature of the softplus nonlinearity.\n",
        "\n",
        "**Model**: current-based, discrete-time Euler, hard clamp `v >= 0`.\n",
        "**Nonlinearity**: softplus `phi(v) = r0 * log(1 + exp(beta*(v - v_th))) / log(2)`."
    ]
})

# ── Cell 1: Imports and model definitions ──────────────────────────────────────
cell1 = r"""
import numpy as np
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')

# ── Softplus nonlinearity and its derivatives ────────────────────────────────
LOG2 = np.log(2.0)

def phi(v, r0, beta, v_th):
    """Softplus transfer function. Always positive, smooth."""
    x = beta * (v - v_th)
    # Numerically stable: log(1+exp(x)) = x + log(1+exp(-x)) for large x
    return r0 * np.where(x > 30, x / LOG2, np.log1p(np.exp(np.clip(x, -500, 30))) / LOG2)

def phi_prime(v, r0, beta, v_th):
    """First derivative of softplus."""
    x = beta * (v - v_th)
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return r0 * beta * sig / LOG2

def phi_double_prime(v, r0, beta, v_th):
    """Second derivative of softplus."""
    x = beta * (v - v_th)
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return r0 * beta**2 * sig * (1.0 - sig) / LOG2

# ── Mean-field fixed-point equations (symmetric: tau1=tau2=tau, mu1=mu2=mu) ──
def mf_rhs(v, mu, tau, alpha_self, alpha_cross, r0, beta, v_th):
    """Returns dv/dt for [v1, v2] (symmetric 2-neuron system)."""
    v1, v2 = v
    phi1 = phi(v1, r0, beta, v_th)
    phi2 = phi(v2, r0, beta, v_th)
    dv1 = (-(v1 - mu) + alpha_self * phi1 + alpha_cross * phi2) / tau
    dv2 = (-(v2 - mu) + alpha_self * phi2 + alpha_cross * phi1) / tau
    return [dv1, dv2]

def find_fixed_points(mu, tau, alpha_self, alpha_cross, r0, beta, v_th,
                      v_grid=None, tol=1e-8):
    """
    Find fixed points of the mean-field system by trying many starting points.
    Returns list of (v1*, v2*) tuples that satisfy the FP equations and v1,v2 > 0.
    """
    if v_grid is None:
        v_grid = np.linspace(0.1, 20.0, 8)
    fps = []
    for v1_0 in v_grid:
        for v2_0 in v_grid:
            try:
                sol = opt.fsolve(mf_rhs, [v1_0, v2_0],
                                 args=(mu, tau, alpha_self, alpha_cross, r0, beta, v_th),
                                 full_output=True, xtol=tol, ftol=tol)
                x, info, ier, msg = sol
                if ier == 1 and x[0] > 0 and x[1] > 0:
                    # Deduplicate
                    duplicate = False
                    for prev in fps:
                        if np.max(np.abs(x - prev)) < 1e-4:
                            duplicate = True
                            break
                    if not duplicate:
                        # Verify residual
                        res = mf_rhs(x, mu, tau, alpha_self, alpha_cross, r0, beta, v_th)
                        if np.max(np.abs(res)) < 1e-6:
                            fps.append(x)
            except Exception:
                pass
    return fps

def compute_jacobian(vbar, mu, tau, alpha_self, alpha_cross, r0, beta, v_th):
    """Jacobian of the mean-field system at fixed point vbar = [v1*, v2*]."""
    v1, v2 = vbar
    pp1 = phi_prime(v1, r0, beta, v_th)
    pp2 = phi_prime(v2, r0, beta, v_th)
    J = np.array([
        [(-1.0 + alpha_self * pp1) / tau,   alpha_cross * pp2 / tau],
        [alpha_cross * pp1 / tau,            (-1.0 + alpha_self * pp2) / tau]
    ])
    return J

def is_stable(J):
    """Returns True if all eigenvalues of J have negative real part."""
    eigvals = np.linalg.eigvals(J)
    return np.all(np.real(eigvals) < 0), eigvals

def predict_mismatch(vbar, J, mu, tau, alpha_self, alpha_cross, r0, beta, v_th):
    """
    Predict the fractional rate mismatch Delta_i = phi''(vbar_i)/(2*phi(vbar_i)) * Var(v_i)
    using the diagonal Lyapunov approximation.
    Returns (Delta1, Delta2, Var1, Var2).
    """
    v1, v2 = vbar
    eigvals = np.linalg.eigvals(J)
    # Stability margin: most negative real part (least stable eigenvalue)
    lambda_max_real = np.max(np.real(eigvals))  # should be negative for stable

    # Diagonal noise approximation
    phi1 = phi(v1, r0, beta, v_th)
    phi2 = phi(v2, r0, beta, v_th)
    D0_1 = (alpha_self**2 + alpha_cross**2) * phi1
    D0_2 = (alpha_cross**2 + alpha_self**2) * phi2  # symmetric

    # Variance from diagonal Lyapunov: Var_i ≈ D0_i / (2 * |J_ii| * tau_i)
    # J_ii = (-1 + alpha_self * phi'(v_i)) / tau
    J11 = J[0, 0]
    J22 = J[1, 1]
    if J11 >= 0 or J22 >= 0:
        return None  # diagonally unstable

    Var1 = D0_1 / (2.0 * abs(J11) * tau)
    Var2 = D0_2 / (2.0 * abs(J22) * tau)

    # Jensen correction
    pp2_1 = phi_double_prime(v1, r0, beta, v_th)
    pp2_2 = phi_double_prime(v2, r0, beta, v_th)
    if phi1 <= 0 or phi2 <= 0:
        return None

    Delta1 = pp2_1 / (2.0 * phi1) * Var1
    Delta2 = pp2_2 / (2.0 * phi2) * Var2

    return Delta1, Delta2, Var1, Var2

print("Cell 1: Model definitions loaded.")
print(f"  phi(1.0, r0=10, beta=1.0, v_th=1.0) = {phi(1.0, 10.0, 1.0, 1.0):.4f} Hz")
print(f"  phi'(1.0, ...) = {phi_prime(1.0, 10.0, 1.0, 1.0):.4f}")
print(f"  phi''(1.0, ...) = {phi_double_prime(1.0, 10.0, 1.0, 1.0):.4f}")
""".strip()

nb["cells"].append({
    "cell_type": "code",
    "id": "cell-1-model",
    "metadata": {},
    "source": cell1,
    "outputs": [],
    "execution_count": None
})

# ── Cell 2: Latin Hypercube parameter scan ─────────────────────────────────────
cell2 = r"""
from scipy.stats import qmc

# ── Sampling bounds ──────────────────────────────────────────────────────────
# Parameters: r0, beta, v_th, mu, tau, alpha_self, alpha_cross
BOUNDS_LOW  = np.array([2.0,  0.1,  0.5,  1.0,  0.05,  0.0,   0.01])
BOUNDS_HIGH = np.array([50.0, 3.0,  5.0,  8.0,  2.0,   1.5,   2.0 ])
PARAM_NAMES = ['r0', 'beta', 'v_th', 'mu', 'tau', 'alpha_self', 'alpha_cross']

N_SAMPLES = 80_000

print(f"Generating {N_SAMPLES} Latin Hypercube samples...")
sampler = qmc.LatinHypercube(d=7, seed=42)
raw = sampler.random(N_SAMPLES)
samples = qmc.scale(raw, BOUNDS_LOW, BOUNDS_HIGH)
print("Done. Scanning parameters...")

# ── Filters ─────────────────────────────────────────────────────────────────
RATE_MIN       = 5.0    # Hz — minimum mean firing rate for both neurons
DELTA_MIN      = 0.03   # minimum predicted mismatch fraction
DELTA_MAX      = 0.15   # maximum predicted mismatch fraction
STABILITY_MIN  = 0.2    # |Re(lambda_max)| must exceed this
SNR_MIN        = 3.0    # vbar / sqrt(Var) must exceed this
JUMP_FRACTION  = 0.3    # max(alpha) / tau < JUMP_FRACTION * vbar

good_params = []

for i, s in enumerate(samples):
    r0, beta, v_th, mu, tau, a_self, a_cross = s

    # Find symmetric fixed point (v1=v2=v*) first (fast check)
    # For symmetric system, v1=v2=v* satisfies:
    # -(v* - mu) + (a_self + a_cross) * phi(v*) = 0
    def fp_sym(v_scalar):
        return -(v_scalar - mu) + (a_self + a_cross) * phi(float(v_scalar), r0, beta, v_th)

    # Quick scalar solve
    fp_found = False
    vstar = None
    for v0 in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]:
        try:
            vs = opt.brentq(fp_sym, 0.01, 50.0, xtol=1e-8) if fp_sym(0.01) * fp_sym(50.0) < 0 else None
            if vs is None:
                sol = opt.fsolve(fp_sym, v0, full_output=True)
                vs_candidate = float(sol[0])
                if abs(fp_sym(vs_candidate)) < 1e-6 and vs_candidate > 0:
                    vs = vs_candidate
            if vs is not None and vs > 0.05:
                vstar = vs
                fp_found = True
                break
        except Exception:
            pass

    if not fp_found or vstar is None:
        continue

    vbar = np.array([vstar, vstar])

    # Compute Jacobian and check stability
    J = compute_jacobian(vbar, mu, tau, a_self, a_cross, r0, beta, v_th)
    stable, eigvals = is_stable(J)
    if not stable:
        continue

    lambda_max_real = np.max(np.real(eigvals))
    stab_margin = abs(lambda_max_real)
    if stab_margin < STABILITY_MIN:
        continue

    # Predict mismatch
    result = predict_mismatch(vbar, J, mu, tau, a_self, a_cross, r0, beta, v_th)
    if result is None:
        continue
    Delta1, Delta2, Var1, Var2 = result

    # Delta must be in target range
    Delta_mean = 0.5 * (Delta1 + Delta2)
    if not (DELTA_MIN <= Delta_mean <= DELTA_MAX):
        continue

    # Firing rate check
    phi1 = phi(vstar, r0, beta, v_th)
    if phi1 < RATE_MIN:
        continue

    # SNR check: v well above 0
    sdv = np.sqrt(max(Var1, 1e-12))
    if vstar / sdv < SNR_MIN:
        continue
    # Also check that phi at vstar - 2*sigma is still > 1 Hz
    if phi(max(vstar - 2.0 * sdv, 0.01), r0, beta, v_th) < 1.0:
        continue

    # Jump size check
    alpha_max = max(a_self, a_cross)
    if alpha_max / tau > JUMP_FRACTION * vstar:
        continue

    good_params.append({
        'r0': r0, 'beta': beta, 'v_th': v_th, 'mu': mu, 'tau': tau,
        'alpha_self': a_self, 'alpha_cross': a_cross,
        'vstar': vstar,
        'phi_star': phi1,
        'Delta1': Delta1, 'Delta2': Delta2, 'Delta_mean': Delta_mean,
        'Var1': Var1, 'Var2': Var2,
        'stab_margin': stab_margin,
        'eigvals': eigvals,
        'J': J
    })

print(f"\nFound {len(good_params)} candidates passing all filters.")

# Sort by how close Delta_mean is to 7% (middle of target range)
good_params.sort(key=lambda d: abs(d['Delta_mean'] - 0.07))

# Print top 20
print("\n{'':=<120}")
header = f"{'#':>3}  {'r0':>6} {'beta':>6} {'v_th':>6} {'mu':>5} {'tau':>6} {'a_self':>7} {'a_cross':>8} | "
header += f"{'v*':>6} {'phi*':>7} {'Delta1%':>8} {'Delta2%':>8} {'stab':>6}"
print(header)
print("-" * len(header))
for idx, p in enumerate(good_params[:20]):
    print(f"{idx+1:>3}  "
          f"{p['r0']:>6.2f} {p['beta']:>6.3f} {p['v_th']:>6.3f} "
          f"{p['mu']:>5.2f} {p['tau']:>6.4f} {p['alpha_self']:>7.4f} {p['alpha_cross']:>8.4f} | "
          f"{p['vstar']:>6.3f} {p['phi_star']:>7.2f} "
          f"{100*p['Delta1']:>8.2f} {100*p['Delta2']:>8.2f} {p['stab_margin']:>6.3f}")
""".strip()

nb["cells"].append({
    "cell_type": "code",
    "id": "cell-2-scan",
    "metadata": {},
    "source": cell2,
    "outputs": [],
    "execution_count": None
})

# ── Cell 3: Stochastic simulation ──────────────────────────────────────────────
cell3 = r"""
# ── Simulation parameters ───────────────────────────────────────────────────
N_CANDIDATES = min(12, len(good_params))
N_TRIALS     = 8
T_TOTAL      = 3000.0   # seconds
T_BURNIN     = 500.0    # seconds
DT           = 5e-3     # seconds
BLOWUP_THRESH = 50.0    # if v > this, abort trial
CLAMP_FRAC_MAX = 0.005  # fraction of time spent at v=0 must be < this

N_STEPS  = int(T_TOTAL / DT)
N_BURNIN = int(T_BURNIN / DT)

rng = np.random.default_rng(2024)

sim_results = []

for cand_idx in range(N_CANDIDATES):
    p = good_params[cand_idx]
    r0 = p['r0']; beta = p['beta']; v_th = p['v_th']
    mu = p['mu']; tau = p['tau']
    a11 = p['alpha_self']; a12 = p['alpha_cross']
    a21 = p['alpha_cross']; a22 = p['alpha_self']

    trial_rates1 = []
    trial_rates2 = []
    trial_meanv1 = []
    trial_meanv2 = []
    trial_clamp1 = []
    trial_clamp2 = []
    blowup = False

    for trial in range(N_TRIALS):
        v1 = float(p['vstar'])
        v2 = float(p['vstar'])
        spk1_count = 0
        spk2_count = 0
        clamp1_count = 0
        clamp2_count = 0
        sum_v1 = 0.0
        sum_v2 = 0.0
        sum_phi1 = 0.0
        sum_phi2 = 0.0
        n_post = 0
        blown = False

        for j in range(N_STEPS):
            # Spike generation
            rate1 = phi(v1, r0, beta, v_th)
            rate2 = phi(v2, r0, beta, v_th)
            s1 = float(rng.random() < rate1 * DT)
            s2 = float(rng.random() < rate2 * DT)

            # Voltage update (current-based, Euler)
            v1_new = v1 + (DT * (-v1 + mu) + a11 * s1 + a12 * s2) / tau
            v2_new = v2 + (DT * (-v2 + mu) + a21 * s1 + a22 * s2) / tau

            # Hard clamp v >= 0
            if v1_new < 0.0:
                clamp1_count += 1
                v1_new = 0.0
            if v2_new < 0.0:
                clamp2_count += 1
                v2_new = 0.0

            v1 = v1_new
            v2 = v2_new

            # Blowup check
            if v1 > BLOWUP_THRESH or v2 > BLOWUP_THRESH or np.isnan(v1) or np.isnan(v2):
                blown = True
                break

            # Accumulate post-burnin
            if j >= N_BURNIN:
                sum_phi1 += rate1
                sum_phi2 += rate2
                sum_v1   += v1
                sum_v2   += v2
                n_post   += 1

        if blown:
            blowup = True
            break

        if n_post > 0:
            trial_rates1.append(sum_phi1 / n_post)
            trial_rates2.append(sum_phi2 / n_post)
            trial_meanv1.append(sum_v1 / n_post)
            trial_meanv2.append(sum_v2 / n_post)
            n_post_steps = N_STEPS - N_BURNIN
            trial_clamp1.append(clamp1_count / n_post_steps)
            trial_clamp2.append(clamp2_count / n_post_steps)

    if blowup or len(trial_rates1) == 0:
        sim_results.append({
            'cand_idx': cand_idx, 'status': 'UNSTABLE', **p
        })
        print(f"Candidate {cand_idx+1:>2}: UNSTABLE (blowup detected)")
        continue

    mean_rate1 = np.mean(trial_rates1)
    mean_rate2 = np.mean(trial_rates2)
    mf_rate1   = phi(p['vstar'], r0, beta, v_th)
    mf_rate2   = mf_rate1  # symmetric

    mismatch1 = (mean_rate1 - mf_rate1) / mf_rate1
    mismatch2 = (mean_rate2 - mf_rate2) / mf_rate2

    frac_clamp1 = np.mean(trial_clamp1)
    frac_clamp2 = np.mean(trial_clamp2)
    clamp_ok = (frac_clamp1 < CLAMP_FRAC_MAX) and (frac_clamp2 < CLAMP_FRAC_MAX)

    valid = (abs(mismatch1) >= 0.03 and abs(mismatch1) <= 0.15 and
             abs(mismatch2) >= 0.03 and abs(mismatch2) <= 0.15 and
             clamp_ok)

    sim_results.append({
        'cand_idx': cand_idx, 'status': 'OK' if clamp_ok else 'CLAMP',
        'valid': valid,
        'mean_rate1': mean_rate1, 'mean_rate2': mean_rate2,
        'mf_rate1': mf_rate1, 'mf_rate2': mf_rate2,
        'mismatch1': mismatch1, 'mismatch2': mismatch2,
        'frac_clamp1': frac_clamp1, 'frac_clamp2': frac_clamp2,
        'trial_rates1': trial_rates1, 'trial_rates2': trial_rates2,
        **p
    })

    star = ' ★' if valid else ''
    print(f"Candidate {cand_idx+1:>2}: rate1={mean_rate1:6.2f} Hz (MF={mf_rate1:.2f}) "
          f"mismatch1={100*mismatch1:+.2f}%  "
          f"rate2={mean_rate2:6.2f} Hz mismatch2={100*mismatch2:+.2f}%  "
          f"clamp={frac_clamp1:.4f}{star}")
""".strip()

nb["cells"].append({
    "cell_type": "code",
    "id": "cell-3-sim",
    "metadata": {},
    "source": cell3,
    "outputs": [],
    "execution_count": None
})

# ── Cell 4: Results table ──────────────────────────────────────────────────────
cell4 = r"""
import warnings
warnings.filterwarnings('ignore')

# Filter to OK results and sort by mean absolute mismatch
ok_results = [r for r in sim_results if r.get('status') not in ('UNSTABLE',) and 'mismatch1' in r]
ok_results.sort(key=lambda r: 0.5 * (abs(r['mismatch1']) + abs(r['mismatch2'])))

print("=" * 130)
print(f"{'#':>3}  {'r0':>6} {'beta':>5} {'v_th':>5} {'mu':>5} {'tau':>6} {'a_self':>7} {'a_cross':>8} |"
      f" {'phi*(Hz)':>8} {'rate1(Hz)':>9} {'mis1%':>7} {'rate2(Hz)':>9} {'mis2%':>7} {'clamp%':>7} {'valid':>6}")
print("-" * 130)

for rank, r in enumerate(ok_results):
    valid_str = ' ★' if r.get('valid') else ''
    print(f"{rank+1:>3}  "
          f"{r['r0']:>6.2f} {r['beta']:>5.3f} {r['v_th']:>5.3f} "
          f"{r['mu']:>5.2f} {r['tau']:>6.4f} {r['alpha_self']:>7.4f} {r['alpha_cross']:>8.4f} | "
          f"{r['mf_rate1']:>8.2f} {r['mean_rate1']:>9.2f} {100*r['mismatch1']:>+7.2f} "
          f"{r['mean_rate2']:>9.2f} {100*r['mismatch2']:>+7.2f} "
          f"{100*r['frac_clamp1']:>7.4f} {valid_str:>6}")

n_valid = sum(1 for r in ok_results if r.get('valid'))
print(f"\nTotal OK candidates: {len(ok_results)} | Valid (3-15% mismatch, both neurons): {n_valid}")
""".strip()

nb["cells"].append({
    "cell_type": "code",
    "id": "cell-4-table",
    "metadata": {},
    "source": cell4,
    "outputs": [],
    "execution_count": None
})

# ── Cell 5: Detailed plots for best valid candidate ───────────────────────────
cell5 = r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Pick the best valid candidate (or best OK if none valid)
valid_results = [r for r in ok_results if r.get('valid')]
best = valid_results[0] if valid_results else (ok_results[0] if ok_results else None)

if best is None:
    print("No valid candidates to plot.")
else:
    r0 = best['r0']; beta = best['beta']; v_th = best['v_th']
    mu = best['mu']; tau = best['tau']
    a_self = best['alpha_self']; a_cross = best['alpha_cross']
    vstar = best['vstar']

    print(f"Best candidate: r0={r0:.3f}, beta={beta:.3f}, v_th={v_th:.3f}, "
          f"mu={mu:.3f}, tau={tau:.4f}, alpha_self={a_self:.4f}, alpha_cross={a_cross:.4f}")
    print(f"  Fixed point: v* = {vstar:.4f}")
    print(f"  MF rate: {best['mf_rate1']:.3f} Hz")
    print(f"  Simulated rate1: {best['mean_rate1']:.3f} Hz  (mismatch {100*best['mismatch1']:+.2f}%)")
    print(f"  Simulated rate2: {best['mean_rate2']:.3f} Hz  (mismatch {100*best['mismatch2']:+.2f}%)")
    print(f"  Eigenvalues: {best['eigvals']}")
    print(f"  Predicted Delta1: {100*best['Delta1']:.2f}%  Delta2: {100*best['Delta2']:.2f}%")

    # ── Re-run one long trial to get traces ─────────────────────────────────
    rng_plot = np.random.default_rng(999)
    T_PLOT  = 3300.0
    DT_PLOT = DT
    N_PLOT  = int(T_PLOT / DT_PLOT)
    N_BURN  = int(500.0 / DT_PLOT)
    PLOT_WIN = int(300.0 / DT_PLOT)  # 300 s window

    v1_trace = np.zeros(N_PLOT)
    v2_trace = np.zeros(N_PLOT)
    phi1_trace = np.zeros(N_PLOT)
    phi2_trace = np.zeros(N_PLOT)

    v1 = vstar; v2 = vstar
    for j in range(N_PLOT):
        r1 = phi(v1, r0, beta, v_th)
        r2 = phi(v2, r0, beta, v_th)
        s1 = float(rng_plot.random() < r1 * DT_PLOT)
        s2 = float(rng_plot.random() < r2 * DT_PLOT)
        v1_new = v1 + (DT_PLOT * (-v1 + mu) + a_self * s1 + a_cross * s2) / tau
        v2_new = v2 + (DT_PLOT * (-v2 + mu) + a_cross * s1 + a_self * s2) / tau
        v1 = max(0.0, v1_new)
        v2 = max(0.0, v2_new)
        v1_trace[j] = v1
        v2_trace[j] = v2
        phi1_trace[j] = r1
        phi2_trace[j] = r2

    t_axis = np.arange(N_PLOT) * DT_PLOT
    # Plot post-burnin window
    t_plot   = t_axis[N_BURN: N_BURN + PLOT_WIN]
    v1_win   = v1_trace[N_BURN: N_BURN + PLOT_WIN]
    v2_win   = v2_trace[N_BURN: N_BURN + PLOT_WIN]
    phi1_win = phi1_trace[N_BURN: N_BURN + PLOT_WIN]
    phi2_win = phi2_trace[N_BURN: N_BURN + PLOT_WIN]

    # Smoothed firing rates (sigma = 0.5 s)
    sigma_pts = int(0.5 / DT_PLOT)
    phi1_smooth = gaussian_filter1d(phi1_win, sigma_pts)
    phi2_smooth = gaussian_filter1d(phi2_win, sigma_pts)
    mf_rate = phi(vstar, r0, beta, v_th)

    # ── Nullclines and phase portrait ────────────────────────────────────────
    v_range = np.linspace(max(0.01, vstar * 0.1), vstar * 2.5, 300)
    # Nullcline v1: dv1/dt=0 => v1 = mu + a_self * phi(v1) + a_cross * phi(v2)
    # Solve numerically on a grid
    V1g, V2g = np.meshgrid(v_range, v_range)
    DV1g = (-(V1g - mu) + a_self * phi(V1g, r0, beta, v_th) + a_cross * phi(V2g, r0, beta, v_th)) / tau
    DV2g = (-(V2g - mu) + a_cross * phi(V1g, r0, beta, v_th) + a_self * phi(V2g, r0, beta, v_th)) / tau

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Best Valid Candidate: r0={r0:.1f}, β={beta:.2f}, v_th={v_th:.2f}, '
        f'μ={mu:.2f}, τ={tau:.3f}, α_self={a_self:.3f}, α_cross={a_cross:.3f}\n'
        f'MF rate={mf_rate:.2f} Hz  |  Sim rate1={best["mean_rate1"]:.2f} Hz '
        f'({100*best["mismatch1"]:+.1f}%)  Sim rate2={best["mean_rate2"]:.2f} Hz '
        f'({100*best["mismatch2"]:+.1f}%)',
        fontsize=11
    )

    # Panel 1: voltage traces
    ax = axes[0, 0]
    ax.plot(t_plot - t_plot[0], v1_win, lw=0.5, alpha=0.7, color='steelblue', label='v₁(t)')
    ax.plot(t_plot - t_plot[0], v2_win, lw=0.5, alpha=0.7, color='tomato', label='v₂(t)')
    ax.axhline(vstar, ls='--', color='navy', lw=1.2, label=f'v* = {vstar:.3f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage v')
    ax.set_title('Stochastic voltage traces (300 s window)')
    ax.legend(fontsize=8)

    # Panel 2: smoothed firing rates
    ax = axes[0, 1]
    ax.plot(t_plot - t_plot[0], phi1_smooth, lw=1.0, color='steelblue', label='rate₁(t) smoothed')
    ax.plot(t_plot - t_plot[0], phi2_smooth, lw=1.0, color='tomato', label='rate₂(t) smoothed')
    ax.axhline(mf_rate, ls='--', color='navy', lw=1.5, label=f'MF rate = {mf_rate:.2f} Hz')
    ax.axhline(best['mean_rate1'], ls=':', color='steelblue', lw=1.5,
               label=f'sim mean₁ = {best["mean_rate1"]:.2f} Hz')
    ax.axhline(best['mean_rate2'], ls=':', color='tomato', lw=1.5,
               label=f'sim mean₂ = {best["mean_rate2"]:.2f} Hz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Smoothed firing rates (σ=0.5 s)')
    ax.legend(fontsize=7)

    # Panel 3: joint histogram
    ax = axes[1, 0]
    v1_post = v1_trace[N_BURN:]
    v2_post = v2_trace[N_BURN:]
    h, xedge, yedge = np.histogram2d(v1_post, v2_post, bins=60)
    ax.imshow(h.T, origin='lower', aspect='auto',
              extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
              cmap='Blues')
    ax.plot(vstar, vstar, 'r*', ms=14, label=f'MF FP ({vstar:.3f}, {vstar:.3f})')
    ax.set_xlabel('v₁')
    ax.set_ylabel('v₂')
    ax.set_title('Joint histogram of (v₁, v₂)')
    ax.legend(fontsize=8)

    # Panel 4: phase portrait + nullclines
    ax = axes[1, 1]
    skip = 12
    ax.quiver(V1g[::skip, ::skip], V2g[::skip, ::skip],
              DV1g[::skip, ::skip], DV2g[::skip, ::skip],
              alpha=0.5, scale=None, width=0.003, color='gray')
    ax.contour(V1g, V2g, DV1g, levels=[0], colors=['steelblue'], linewidths=2)
    ax.contour(V1g, V2g, DV2g, levels=[0], colors=['tomato'], linewidths=2)
    ax.plot(vstar, vstar, 'k*', ms=14, zorder=5, label=f'FP ({vstar:.3f}, {vstar:.3f})')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=2, label='dv₁/dt = 0 nullcline'),
        Line2D([0], [0], color='tomato',    lw=2, label='dv₂/dt = 0 nullcline'),
        Line2D([0], [0], marker='*', color='k', ms=10, lw=0, label='Fixed point')
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_xlabel('v₁')
    ax.set_ylabel('v₂')
    ax.set_title('Phase portrait + nullclines')

    plt.tight_layout()
    plt.savefig('hawkes_2neuron_best_candidate.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Figure saved to hawkes_2neuron_best_candidate.png")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL SUMMARY — BEST VALID CANDIDATE")
    print("="*70)
    print(f"  r0          = {r0:.4f} Hz")
    print(f"  beta        = {beta:.4f}")
    print(f"  v_th        = {v_th:.4f}")
    print(f"  mu          = {mu:.4f}")
    print(f"  tau         = {tau:.4f} s")
    print(f"  alpha_self  = {a_self:.4f}")
    print(f"  alpha_cross = {a_cross:.4f}")
    print(f"  Fixed point: v1* = v2* = {vstar:.5f}")
    print(f"  MF firing rate: phi(v*) = {mf_rate:.4f} Hz")
    print(f"  Jacobian eigenvalues: {best['eigvals']}")
    print(f"  Stability margin |Re(lam_max)|: {best['stab_margin']:.4f}")
    print(f"  Predicted Delta1: {100*best['Delta1']:.3f}%  Delta2: {100*best['Delta2']:.3f}%")
    print(f"  Simulated mean rate1: {best['mean_rate1']:.4f} Hz  (mismatch {100*best['mismatch1']:+.3f}%)")
    print(f"  Simulated mean rate2: {best['mean_rate2']:.4f} Hz  (mismatch {100*best['mismatch2']:+.3f}%)")
    print(f"  Clamp fraction: {100*best['frac_clamp1']:.4f}%")
""".strip()

nb["cells"].append({
    "cell_type": "code",
    "id": "cell-5-plots",
    "metadata": {},
    "source": cell5,
    "outputs": [],
    "execution_count": None
})

# Write notebook
out_path = "/Users/matthewszuromi/Documents/Education/BU PhD/Ocker Lab/Automated Feynman Calculations/Test System Code/Hawkes 2 Neuron Parameter Search.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written to:\n  {out_path}")

"""Data for the delayed Hawkes figure: pipeline rates and spike cross-covariances, and independent simulations.

    sage -python compute.py rate <D> [ell]     -> data/pipe_rate_D<D>_ell<ell>.json   (default ell = 2)
    sage -python compute.py xcov <D>           -> data/pipe_xcov_D<D>.json            (one loop, 25 lags)
    sage -python compute.py xtree <D>          -> data/pipe_xcov_D<D>_tree.json       (tree, 241 lags)
    sage -python compute.py sim <D> <dt>       -> data/sim_D<D>_dt<dt>.json
    sage -python compute.py sims               -> every simulation the figure uses (dt = 0.005, 0.0025, 0.00125)

Costs (v0.4.0, n_workers = 2, at the pinned Pade orders below): two-loop rate 10 s (D = 0) to about 3 min (D = 2); the
delay's Pade states drive it. One-loop cross-covariance with delay, 25 lags: about 25 min. Each simulation takes
seconds (numba)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
sys.path.insert(0, os.path.dirname(HERE))
import style
style.repo_root(HERE)

D_RATE = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
D_XCOV = (0.0, 1.0, 2.0)
TAU_XCOV = np.round(np.linspace(-6, 6, 25), 6)
TAU_TREE = np.round(np.linspace(-6, 6, 241), 6)
# Pade order of the delay, pinned: the automatic order escalates 4 -> 8 -> 16 only while the predicted wall time of
# the next order stays under wall_max (900 s), so it depends on the machine's speed and load. Pinned orders make
# the results reproducible. Estimated relative errors: rates at n = 8 up to 3e-6 (D = 2); cross-covariance at
# n = 16 up to 1e-4 (D = 2). Both are far below the simulation errors.
ORDER_RATE, ORDER_XCOV = 8, 16
DTS = (0.005, 0.0025, 0.00125)
T_RUN, N_REP, T_BURN, TAU_M, TAUG, A_RATE = 4000.0, 1024, 200.0, 2.0, 1.0, 1.0


def _model(D):
    import model as hk
    return hk.build(delay=D > 0).build(), hk.params(D)


def run_rate(D, ell=2, n_workers=2):
    import daedalus as dd
    m, p = _model(D); t0 = time.perf_counter()
    r = dd.run(m, dd.Config(k=1, max_ell=ell, external_fields=[('dn', 1)], parameters=p, tau_grid=np.array([0.0]),
                            parallel=True, n_workers=n_workers, verbose=False,
                            delay={'order': ORDER_RATE} if D > 0 else None))
    inc = [float(np.real(np.atleast_1d(r['C_tau_by_ell'][l]))[0]) for l in sorted(r['C_tau_by_ell'])]
    nstar = float(np.atleast_1d(r['mf_values']['nstar'])[0])
    out = dict(D=D, ell=ell, pade_order=ORDER_RATE if D > 0 else None, nstar=nstar, increments=inc, cum=(nstar + np.cumsum(inc)).tolist(),
               ndiag=len(r['diagrams']), wall_s=time.perf_counter() - t0, version=dd.__version__)
    json.dump(out, open(os.path.join(DATA, f'pipe_rate_D{D:g}_ell{ell}.json'), 'w'))
    print('rate', D, out['cum'], f'{out["wall_s"]:.0f}s', flush=True)
    return out


def run_xcov(D, ell=1, n_workers=2):
    """<n_1(t) n_2(t + tau)>: one loop on the 25-lag grid; ell = 0 gives tree level on a dense 241-lag grid."""
    import daedalus as dd
    m, p = _model(D); t0 = time.perf_counter(); taus = TAU_XCOV if ell else TAU_TREE
    r = dd.run(m, dd.Config(k=2, max_ell=ell, external_fields=[('dn', 1), ('dn', 2)], parameters=p, tau_grid=taus,
                            parallel=True, n_workers=n_workers, verbose=False,
                            delay={'order': ORDER_XCOV} if D > 0 else None))
    out = dict(D=D, tau=taus.tolist(), pade_order=ORDER_XCOV if D > 0 else None, by_ell={str(l): np.real(np.asarray(v)).tolist() for l, v in r['C_tau_by_ell'].items()},
               ndiag=len(r['diagrams']), wall_s=time.perf_counter() - t0, version=dd.__version__)
    json.dump(out, open(os.path.join(DATA, f'pipe_xcov_D{D:g}' + ('' if ell else '_tree') + '.json'), 'w'))
    print('xcov', D, f'{out["wall_s"]:.0f}s', flush=True)
    return out


def run_sim(D, dt):
    """Independent simulation (no diagrams): Poisson spike counts per step at intensity a exp(v) from the step start,
    exact exponential updates of the voltage and alpha filters, spikes held in a delay line of D/dt steps.
    1024 replicas x T = 4000 after a burn-in of 200; replica r uses seed 12345 + D/dt + r."""
    from simulator import run
    import model as hk
    w = np.array(hk.W); binsz = max(1, int(round(0.1 / dt))); bdt = binsz * dt; nD = int(round(D / dt)); t0 = time.perf_counter()
    c, vv = run(N_REP, int(T_RUN / dt), int(T_BURN / dt), dt, hk.EM, TAU_M, TAUG, A_RATE, w, binsz, 12345 + nD, nD)
    # The exponential network is metastable: a replica can escape to runaway firing (the expansion describes the
    # metastable state). Escaped replicas (any bin above 50 spikes) are dropped and counted.
    keep = np.isfinite(c).all((1, 2)) & (c.max((1, 2)) < 50)
    c = c[keep]; n_ok = int(keep.sum())
    rate = c.sum(2) / T_RUN
    L = 60
    Xc = c / bdt; Xc = Xc - Xc.mean(2, keepdims=True); res = np.zeros((n_ok, 2, 2, L + 1))
    for k in range(L + 1):                       # res[r, i, j, k] = <x_i(t + k) x_j(t)>
        res[:, :, :, k] = np.einsum('rit,rjt->rij', Xc[:, :, k:], Xc[:, :, :Xc.shape[2] - k]) / (Xc.shape[2] - k)
    out = dict(D=D, dt=dt, T=T_RUN, n_rep=N_REP, n_escaped=N_REP - n_ok, bin=bdt, rate_mean=rate.mean(0).tolist(),
               rate_se=(rate.std(0, ddof=1) / np.sqrt(n_ok)).tolist(), lags=(np.arange(L + 1) * bdt).tolist(),
               Cnn=res.mean(0).tolist(), Cnn_se=(res.std(0, ddof=1) / np.sqrt(n_ok)).tolist(),
               convention='Cnn[i][j][k] = <x_i(t + lag_k) x_j(t)>, x = binned spike rate', wall_s=time.perf_counter() - t0)
    json.dump(out, open(os.path.join(DATA, f'sim_D{D:g}_dt{dt:g}.json'), 'w'))
    print('sim', D, dt, np.round(out['rate_mean'], 5), f'escaped {N_REP - n_ok}', f'{out["wall_s"]:.0f}s', flush=True)
    return out


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def extrapolate(vals, ses, dts=DTS):
    """Weighted least-squares fit v(dt) = v0 + c dt; returns v0 and its standard error (arrays allowed)."""
    vals, ses = np.asarray(vals, float), np.asarray(ses, float)
    x = np.asarray(dts, float).reshape((-1,) + (1,) * (vals.ndim - 1)); w = 1 / ses ** 2
    S0, S1, S2 = w.sum(0), (w * x).sum(0), (w * x * x).sum(0); Sy, Sxy = (w * vals).sum(0), (w * x * vals).sum(0)
    det = S0 * S2 - S1 ** 2
    return (S2 * Sy - S1 * Sxy) / det, np.sqrt(S2 / det)


def sim_rate(D):
    """Population-1 rate extrapolated linearly to dt -> 0 from the three time steps, with its standard error."""
    runs = [load(f'sim_D{D:g}_dt{dt:g}.json') for dt in DTS]
    return extrapolate([r['rate_mean'][0] for r in runs], [r['rate_se'][0] for r in runs])


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    if sys.argv[1] == 'rate':
        run_rate(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif sys.argv[1] == 'xcov':
        run_xcov(float(sys.argv[2]))
    elif sys.argv[1] == 'xtree':
        run_xcov(float(sys.argv[2]), ell=0)
    elif sys.argv[1] == 'sims':
        for D in D_RATE:
            for dt in DTS:
                run_sim(D, dt)
    else:
        run_sim(float(sys.argv[2]), float(sys.argv[3]))

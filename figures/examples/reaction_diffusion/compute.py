"""Data for the two-species reaction-diffusion figure: pipeline cumulants and lattice simulations.

    sage -python compute.py pipeline eq1|eq2|tau1|tau2     -> data/pipe_<job>.json
    sage -python compute.py sim <n_x> <dt> <lam> [taus]    -> data/sim_nx<n_x>_lam<lam>[_tau].json
    sage -python compute.py sims                           -> every simulation the figure uses

Two loops need SPATIAL_MEM_BUDGET_GB=20 (the chambers take 9-13 GB; the default 6 GB guard refuses).
Costs on an Apple M-series laptop (v0.4.0): eq1 and tau1 about 10 s; eq2 30-60 min per pair; tau2 about 6 h.
The simulations take about 1 h in total (most of it the three n_x = 512 runs)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
sys.path.insert(0, os.path.dirname(HERE))
import style
style.repo_root(HERE)

LAM = 0.1                                   # the loop increments scale exactly as lam^ell, so one lam gives all
CHI = np.arange(0.0, 6.01, 0.5)
TAU1 = np.round(np.arange(-4.0, 4.001, 0.2), 10)
TAU2 = np.array([-3.0, -2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0, 3.0])
JOBS = {'eq1': (1, ('aa', 'bb', 'ab'), CHI, np.array([0.0])),
        'eq2': (2, ('aa', 'bb', 'ab'), CHI, np.array([0.0])),
        'tau1': (1, ('ab',), np.array([0.0]), TAU1),
        'tau2': (2, ('ab',), np.array([0.0]), TAU2)}
# simulations: L = 32 so dx = 0.25, 0.125, 0.0625 land exactly on the chi grid
L_BOX, NCHUNK, NREP, TRUN, TBURN = 32.0, 12, 8, 250.0, 30.0
SIMS = [(128, 0.02, lam) for lam in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)] + \
       [(256, 0.005, lam) for lam in (0.05, 0.1, 0.15, 0.2, 0.3)] + \
       [(512, 0.00125, lam) for lam in (0.05, 0.1, 0.2)]
SIM_TAU = (256, 0.005, 0.1, (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0))
M_LIN = np.array([[1.0, 1.0], [-1.0, 1.0]]); NOISE = np.diag([2.0, 1.4])   # M = [[mua, g], [-h, mub]], N = diag(2Ta, 2Tb)
IJ = {'aa': (0, 0), 'bb': (1, 1), 'ab': (0, 1)}


def run_pipeline(job, n_workers=4):
    import daedalus as dd
    import model as rd
    ell, pairs, chi, taus = JOBS[job]
    m = rd.build().build()
    fn = os.path.join(DATA, f'pipe_{job}.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for pr in pairs:
        t0 = time.perf_counter()
        cfg = dd.Config(k=2, max_ell=ell, external_fields=[('d' + pr[0], 1), ('d' + pr[1], 1)], parameters=rd.params(LAM),
                        chi_grid=chi, tau_grid=taus, parallel=True, n_workers=n_workers, verbose=False)
        byo = dd.run(m, cfg)['C_tau_x_by_order']
        out[pr] = dict(chi=chi.tolist(), taus=taus.tolist(), params=rd.params(LAM), wall_s=time.perf_counter() - t0,
                       version=dd.__version__, cum={str(o): np.real(np.asarray(byo[o])).tolist() for o in sorted(byo)})
        json.dump(out, open(fn, 'w'))
        print(job, pr, f'{out[pr]["wall_s"]:.0f}s', flush=True)
    return out


def run_sim(n_x, dt, lam, taus=(0.0,)):
    """Semi-implicit Crank-Nicolson lattice simulation; NCHUNK chunks x NREP replicas, chunk seeds 1000*chunk + 7."""
    from daedalus.simulations.coupled_rd_1d_sim import simulate_coupled_rd_1d
    reps, t0 = [], time.perf_counter()
    for ch in range(NCHUNK):
        r = simulate_coupled_rd_1d(M_LIN, 1.0, NOISE, g=np.array([lam, lam]), L=L_BOX, n_x=n_x, dt=dt, t_burn=TBURN,
                                   t_run=TRUN, n_rep=NREP, seed=1000 * ch + 7, lags=taus, x_lags=CHI)
        reps.append(r['C_rep'])                                     # (n_rep, n_tau, 2, 2, n_chi)
    A = np.concatenate(reps, 0)
    out = dict(n_x=n_x, dt=dt, dx=L_BOX / n_x, lam=lam, L=L_BOX, taus=list(map(float, r['taus'])),
               x=list(map(float, r['x_grid'])), C=A.mean(0).tolist(), C_se=(A.std(0, ddof=1) / np.sqrt(len(A))).tolist(),
               n_samples=len(A), wall_s=time.perf_counter() - t0, convention='C[t, i, j, x] = <phi_i(0, 0) phi_j(x, tau_t)>')
    tag = f'nx{n_x}_lam{lam}' + ('_tau' if len(taus) > 1 else '')
    json.dump(out, open(os.path.join(DATA, f'sim_{tag}.json'), 'w'))
    print('sim', tag, f'{out["wall_s"]:.0f}s', flush=True)
    return out


# ---- analysis used by the notebook ------------------------------------------------------------------------------
def load(name):
    return json.load(open(os.path.join(DATA, name)))


def pipe(lam):
    """{pair: {ell: C_ij(chi, 0) up to ell loops}} at any lam, by the exact lam^ell scaling of the lam = 0.1 runs."""
    p1, p2 = load('pipe_eq1.json'), load('pipe_eq2.json')
    out = {}
    for pr in IJ:
        t = np.ravel(p1[pr]['cum']['0']); i1 = np.ravel(p1[pr]['cum']['1']) - t
        i2 = np.ravel(p2[pr]['cum']['2']) - np.ravel(p2[pr]['cum']['1'])
        out[pr] = {0: t, 1: t + i1 * (lam / LAM), 2: t + i1 * (lam / LAM) + i2 * (lam / LAM) ** 2}
    return out


def lattice_shift(S, taus=(0.0,)):
    """Exact linear lattice-minus-continuum difference for a simulation file, (n_tau, 2, 2, n_chi)."""
    from daedalus.simulations.coupled_rd_1d_sim import coupled_box_correlator
    lat = coupled_box_correlator(M_LIN, 1.0, NOISE, L=S['L'], n_x=S['n_x'], taus=list(taus), dispersion='lattice')
    idx = np.round(CHI / S['dx']).astype(int)
    return np.asarray(lat['C'])[..., idx]


def sim_pooled(lam, tree):
    """Lattice-corrected equal-time simulations at every dx, pooled by inverse variance: {pair: (value, se)}."""
    import glob
    fns = sorted(f for f in glob.glob(os.path.join(DATA, f'sim_nx*_lam{lam}.json')) if not f.endswith('_tau.json'))
    vals = {pr: [] for pr in IJ}
    for f_ in fns:
        S = json.load(open(f_)); lat = lattice_shift(S)[0]; C = np.asarray(S['C'])[0]; E = np.asarray(S['C_se'])[0]
        for pr, (i, j) in IJ.items():
            vals[pr].append((C[i, j] - (lat[i, j] - tree[pr]), E[i, j]))
    out = {}
    for pr, v in vals.items():
        V = np.array([a for a, _ in v]); W = 1 / np.array([e for _, e in v]) ** 2
        out[pr] = ((W * V).sum(0) / W.sum(0), 1 / np.sqrt(W.sum(0)))
    return out, len(fns)


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    if sys.argv[1] == 'pipeline':
        run_pipeline(sys.argv[2])
    elif sys.argv[1] == 'sims':
        for a in SIMS:
            run_sim(*a)
        run_sim(*SIM_TAU)
    else:
        run_sim(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
                tuple(float(t) for t in sys.argv[5].split(',')) if len(sys.argv) > 5 else (0.0,))

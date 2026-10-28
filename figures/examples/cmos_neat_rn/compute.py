"""Data for the NEAT-RN capacitance-scaling figure.

At fixed gate voltages the dynamics (in units of t_u) depend on the node capacitance C only through s = C_T/C: the
tree covariance scales as s, the one-loop correction as s^2, the mean shift as s. So ONE pipeline run per gate point
(at C_REF = 300 aF) gives tree and one loop at every C, while the exact master equation is re-solved at each C.

    sage -python compute.py loop <name>        one loop at C_REF            -> data/loop_<name>.json
    sage -python compute.py me <name>          exact master equation at CS  -> data/me_<name>.json
    sage -python compute.py comp <C> <vp1> <vp2>   one-loop gate pre-compensation, checked on the exact solve
                                                                            -> data/comp_C<C>_vp<vp1>_<vp2>.json
    sage -python compute.py all                everything below (about 1.5 h)

Gate points <name>: bal1 / bal2 = inverter 1 / 2 at balance (V_p^1 = 120 / 64 mV); gen = a generic point (90 mV);
chip1 / chip2 = the balance points with the chip's measured amplification (T_eff/T = 4.4 / 5.4).
Costs (v0.4.0): loop 1-2 min per point; me 2-60 s per C; comp 2-5 min per point."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import style
style.repo_root(HERE)
import neat

NAMES = ['bal1', 'bal2', 'gen', 'chip1', 'chip2']
CS = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 1770.0, 3000.0, 5000.0]
C_REF = 300.0
COMP = [(C, vp1, vp2) for vp1, vp2 in ((120.0, 80.0), (64.0, 136.0))
        for C in (100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 1770.0, 3000.0, 5000.0)]
VT_mV = neat.VT * 1e3
VDD_CHIP = [2 * np.log(2 * 4.4 - 1) * VT_mV, 2 * np.log(2 * 5.4 - 1) * VT_mV]   # T_eff/T = 4.4, 5.4 at balance
SEED_BOX = {'u': (-0.05, 0.05), 'w': (-0.05, 0.05)}
TRI = [(0, 0), (0, 1), (1, 1)]


def gates(name):
    """(vp1, vp2, vdd) in mV. chip*: each inverter's V_p shifted with its effective V_dd, so c_p/c_n (balance) is kept
    while the balance amplification (1 + e^{V_dd/2V_T})/2 drops to the chip's measured 4.4 / 5.4."""
    d1, d2 = VDD_CHIP[0] - 137.0, VDD_CHIP[1] - 142.0
    return {'bal1': (120.0, 80.0, None), 'bal2': (64.0, 136.0, None), 'gen': (90.0, 110.0, None),
            'chip1': (120.0 + d1, 80.0 + d2, VDD_CHIP), 'chip2': (64.0 + d1, 136.0 + d2, VDD_CHIP)}[name]


def P_at(name, C):
    vp1, vp2, vdd = gates(name)
    return neat.point(vp1, C, vp2_mV=vp2, vdd_mV=vdd)


def t_I(C):
    """The chip's 200 ns gyration window in units of t_u."""
    return 200e-9 / (C * 1e-18 * neat.VT / neat.I0P)


_MODEL = None
def model():
    global _MODEL
    if _MODEL is None:
        import model as cm
        _MODEL = cm.build().build()
    return _MODEL


def run(P, ext, grid, n_workers=4):
    import daedalus as dd
    import model as cm
    r = dd.run(model(), dd.Config(k=len(ext), max_ell=1, external_fields=ext, tau_grid=grid, parameters=cm.params_of(P),
                                  parallel=True, n_workers=n_workers, verbose=False, mf_dae_seed_box=SEED_BOX))
    return {str(l): np.real(np.atleast_1d(r['C_tau_by_ell'][l])).tolist() for l in sorted(r['C_tau_by_ell'])}


def gl_grid_multi(P, extra, nodes=10):
    """Gauss-Legendre lag grid with a breakpoint at every t_I(C); weights of int_0^{t_I(C)} per C."""
    lam = np.sort(np.abs(np.linalg.eigvals(P['A'])))
    t_end = 14.0 / lam[0]
    br = [0.0, 1.0 / lam[1], 6.0 / lam[1], 0.5 / lam[0], 2.0 / lam[0], 6.0 / lam[0], t_end] + list(extra)
    br = np.unique(np.round([b for b in br if 0 <= b <= t_end], 12))
    x, w = np.polynomial.legendre.leggauss(nodes)
    taus, ws, pan = [0.0], [0.0], [(-1.0, -1.0)]
    for a, b in zip(br[:-1], br[1:]):
        taus += list(0.5 * (b - a) * x + 0.5 * (a + b)); ws += list(0.5 * (b - a) * w); pan += [(a, b)] * nodes
    taus, ws = np.array(taus), np.array(ws)
    hi = np.array([p[1] for p in pan])
    weights = {float(e): (ws * (hi <= min(e, t_end) + 1e-12)).tolist() for e in list(extra) + [t_end]}
    return taus, weights, t_end


def run_loop(name):
    import daedalus as dd
    P = P_at(name, C_REF)
    taus, weights, t_end = gl_grid_multi(P, [t_I(C) for C in CS])
    sym = np.concatenate([-taus[:0:-1], taus])
    out = dict(name=name, C_ref=C_REF, s_ref=P['s'], gates=gates(name), taus=taus.tolist(), weights=weights,
               t_end=t_end, wall={}, version=dd.__version__)
    for nm, ext, grid in (('m_u', [('du', 1)], np.array([0.0])), ('m_w', [('dw', 1)], np.array([0.0])),
                          ('X_uu', [('du', 1), ('du', 1)], taus), ('X_ww', [('dw', 1), ('dw', 1)], taus),
                          ('X_uw', [('du', 1), ('dw', 1)], sym)):
        t0 = time.perf_counter()
        out[nm] = run(P, ext, grid)
        out['wall'][nm] = time.perf_counter() - t0
        print(f'{name} {nm}: {out["wall"][nm]:.0f}s ({len(grid)} lags)', flush=True)
    json.dump(out, open(os.path.join(DATA, f'loop_{name}.json'), 'w'))


def run_me(name, Cs=CS):
    import me_exact
    out = []
    for C in Cs:
        t0 = time.perf_counter()
        P = P_at(name, C)
        rec = dict(C=C, s=P['s'], tI=t_I(C))
        try:
            for K in (9.5, 13.0):
                L = me_exact.build(P, K=K)
                p, resid = me_exact.stationary(L)
                mo = me_exact.moments(L, p)
                if mo['edge_mass'] < 1e-12:
                    break
            lam = np.sort(np.abs(np.linalg.eigvals(P['A'])))
            lag = me_exact.lagged(L, p, mo['mu'], np.array([0.0, 1.0 / lam[0], 3.0 / lam[0]]), t_I=t_I(C), m=80, P=P)
            rec.update(ok=True, K=K, mu=mo['mu'].tolist(), Sigma=mo['Sigma'].tolist(), skew=mo['skew'].tolist(),
                       exkurt=mo['exkurt'].tolist(), edge=mo['edge_mass'], resid=resid, nstates=int(L['Q'].shape[0]),
                       Xbar=lag['Xbar'].tolist(), Xinf=lag['Xinf'].tolist(), conv=max(lag['conv'].values()))
        except Exception as e:                       # where the exact solve stops working (very small C)
            rec.update(ok=False, err=f'{type(e).__name__}: {e}'[:300])
        rec['wall'] = time.perf_counter() - t0
        out.append(rec)
        print(f'me {name} C={C:g}: ok={rec["ok"]} ({rec["wall"]:.1f}s)', flush=True)
    json.dump(out, open(os.path.join(DATA, f'me_{name}.json'), 'w'))


# ---- one-loop pre-compensation of the gate voltages --------------------------------------------------------------
def P_of(g, C):
    """g = (V_p^1, V_n^1, V_p^2, V_n^2, V_dd^1, V_dd^2) in mV."""
    return neat.point(g[0], C, vp2_mV=g[2], vn_mV=[g[1], g[3]], vdd_mV=[g[4], g[5]])


def sig3(S):
    return np.array([S[i, j] for i, j in TRI])


def tree3(g, C):
    return sig3(P_of(g, C)['Sigma'])


def loop3(g, C):
    P = P_of(g, C)
    return np.array([run(P, ext, np.array([0.0]))['1'][0]
                     for ext in ([('du', 1), ('du', 1)], [('du', 1), ('dw', 1)], [('dw', 1), ('dw', 1)])])


def solve_tree(target, g0, C, shift, tol=1e-12, it=400):
    """Minimum-norm Gauss-Newton for tree3(g) + shift = target (relative residuals), 3 mV trust region."""
    g = np.array(g0, float)

    def resid(g):
        try:
            return (tree3(g, C) + shift - target) / target
        except ValueError:
            return np.full(3, np.inf)
    r = resid(g)
    for _ in range(it):
        if np.max(np.abs(r)) < tol:
            break
        n = len(g); Jm = np.zeros((3, n)); h = 1e-3
        for k in range(n):
            e = np.zeros(n); e[k] = h
            Jm[:, k] = (tree3(g + e, C) - tree3(g - e, C)) / (2 * h) / target
        step = -np.linalg.pinv(Jm) @ r
        if np.linalg.norm(step) > 3.0:
            step *= 3.0 / np.linalg.norm(step)
        lam = 1.0
        while lam > 1e-4:
            r_new = resid(g + lam * step)
            if np.linalg.norm(r_new) < np.linalg.norm(r):
                break
            lam /= 2
        g, r = g + lam * step, r_new
    return g, float(np.max(np.abs(r)))


def me3(g, C):
    import me_exact
    L = me_exact.build(P_of(g, C), K=9.0)
    p, _ = me_exact.stationary(L)
    return sig3(me_exact.moments(L, p)['Sigma'])


def run_comp(C, vp1, vp2, outer=4):
    """Target: the covariance the paper's Lyapunov (tree) model gives at the default gates g0. Knobs: V_p, V_n and
    V_dd of both inverters (gates alone cannot raise T_eff at balance). The one-loop design solves
    tree(g) + one loop(g) = target, refreshing the one-loop term each outer iteration; both designs are then checked
    against the exact master equation."""
    g0 = np.array([vp1, 76.0, vp2, 76.0, 137.0, 142.0])
    target = tree3(g0, C)
    log = dict(C_aF=C, g0=g0.tolist(), target=target.tolist(), iters=[], me_naive=me3(g0, C).tolist())
    g = g0.copy(); d1 = loop3(g, C)
    for _ in range(outer):
        g_new, res = solve_tree(target, g, C, d1)
        d1_new = loop3(g_new, C)
        log['iters'].append(dict(g=g_new.tolist(), d1=d1_new.tolist(), tree_res=res, pred=(tree3(g_new, C) + d1_new).tolist()))
        done = np.max(np.abs(g_new - g)) < 1e-4
        g, d1 = g_new, d1_new
        if done:
            break
    log.update(g_comp=g.tolist(), me_comp=me3(g, C).tolist())
    json.dump(log, open(os.path.join(DATA, f'comp_C{C:g}_vp{vp1:g}_{vp2:g}.json'), 'w'), indent=1)
    print(f'comp C={C:g} vp1={vp1:g}: naive {np.round(np.array(log["me_naive"]) / target - 1, 5)}  '
          f'compensated {np.round(np.array(log["me_comp"]) / target - 1, 5)}', flush=True)


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    mode = sys.argv[1]
    if mode == 'loop':
        run_loop(sys.argv[2])
    elif mode == 'me':
        run_me(sys.argv[2])
    elif mode == 'comp':
        run_comp(*map(float, sys.argv[2:5]))
    elif mode == 'all':
        for n in NAMES:
            run_loop(n); run_me(n)
        for c in COMP:
            run_comp(*c)

"""Observables, tables and fits for the NEAT-RN capacitance-scaling study (data written by compute.py)."""
import os, json, glob
import numpy as np
import neat
from compute import DATA, NAMES, P_at, t_I

BAL_NODE = {'bal1': 0, 'bal2': 1, 'chip1': 0, 'chip2': 1, 'gen': 0}
C_CHIP = 1700.0 / (1 - neat.R_DEFAULT ** 2)        # Maxwell diagonal C for C_tot = 1.7 fF


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def theory_X(d):
    """Lagged correlation matrices X_jk(tau) = <x_j(0) x_k(tau)> by loop order, on the tau >= 0 grid."""
    taus = np.array(d['taus']); n = len(taus)
    X = {}
    for ell in ('0', '1'):
        A = np.zeros((n, 2, 2))
        A[:, 0, 0] = d['X_uu'][ell]; A[:, 1, 1] = d['X_ww'][ell]
        sym = np.array(d['X_uw'][ell]); A[:, 0, 1] = sym[n - 1:]; A[:, 1, 0] = sym[n - 1::-1]
        X[ell] = A
    return taus, X


def weights_for(d, tI):
    keys = np.array([float(k) for k in d['weights']])
    k = float(d['t_end']) if tI >= d['t_end'] - 1e-9 else keys[np.argmin(np.abs(keys - tI))]
    return np.array(d['weights'][[kk for kk in d['weights'] if abs(float(kk) - k) < 1e-9][0]])


def observables(mu, S, Xbar, Xinf):
    """Means, covariance, correlation times, and the circulation l over the window t_I (and t_I -> infinity)."""
    return dict(mu1=mu[0], mu2=mu[1], var1=S[0, 0], var2=S[1, 1], cov=S[0, 1],
                tauc1=Xinf[0, 0] / S[0, 0], tauc2=Xinf[1, 1] / S[1, 1],
                l=(Xbar[0, 1] - Xbar[1, 0]) / np.sqrt(S[0, 0] * S[1, 1]),
                linf=(Xinf[0, 1] - Xinf[1, 0]) / np.sqrt(S[0, 0] * S[1, 1]))


def rows(name):
    """Tree, tree + one loop and exact observables at every C of the master-equation sweep."""
    d, me = load(f'loop_{name}.json'), load(f'me_{name}.json')
    taus, X = theory_X(d)
    wT = np.array(d['weights'][[k for k in d['weights'] if abs(float(k) - d['t_end']) < 1e-9][0]])
    out = []
    for e in me:
        if not e.get('ok'):
            continue
        C = e['C']; f = e['s'] / d['s_ref']; tI = t_I(C); w = weights_for(d, tI); P = P_at(name, C)
        Xt, X1 = X['0'] * f, X['1'] * f * f                        # tree ~ s, one loop ~ s^2
        shift = np.array([d['m_u']['1'][0], d['m_w']['1'][0]]) * f
        ob = {}
        for src, Xs, mu in (('tree', Xt, np.zeros(2)), ('loop', Xt + X1, shift)):
            ob[src] = observables(mu, Xs[0], np.tensordot(w, Xs, 1) / tI, np.tensordot(wT, Xs, 1))
        ob['exact'] = observables(np.array(e['mu']), np.array(e['Sigma']), np.array(e['Xbar']), np.array(e['Xinf']))
        ob['tree_check'] = float(np.max(np.abs(Xt[0] - P['Sigma'])) / np.max(np.abs(P['Sigma'])))
        out.append(dict(C=C, s=e['s'], eps=np.diag(P['Sigma']).tolist(), ob=ob))
    return out


def rel(o, k, src):
    t = o['tree'][k]
    return o[src][k] - t if k.startswith('mu') else (o[src][k] - t) / abs(t)


def crossing(Cs, vals, level):
    """Largest C at which |vals| >= level (log-log interpolation); None if never."""
    Cs, v = np.array(Cs), np.abs(np.array(vals))
    order = np.argsort(Cs)[::-1]; Cs, v = Cs[order], v[order]
    for i in range(1, len(Cs)):
        if v[i - 1] < level <= v[i]:
            x0, x1 = np.log(Cs[i - 1]), np.log(Cs[i]); y0, y1 = np.log(v[i - 1]), np.log(v[i])
            return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0)))
    return None


def fit_law(rs, k, nd):
    """delta Sigma / Sigma = c1 eps + c2 eps^2: c1 from one loop, c2 from the residual R = exact - tree - one loop
    fitted as a2 s^2 + a3 s^3 on C >= 300 aF."""
    pts = [(r['s'], rel(r['ob'], k, 'exact') - rel(r['ob'], k, 'loop'), r['eps'][nd]) for r in rs if r['C'] >= 300]
    s = np.array([p[0] for p in pts]); R = np.array([p[1] for p in pts]); eps = np.array([p[2] for p in pts])
    a2, a3 = np.linalg.lstsq(np.vstack([s ** 2, s ** 3]).T, R, rcond=None)[0]
    b = np.mean(eps / s)
    a1 = np.mean([rel(r['ob'], k, 'loop') / r['s'] for r in rs])
    return dict(c1=a1 / b, c2=a2 / b ** 2, eps_over_s=b)


def summary():
    out = {}
    for name in NAMES:
        rs = rows(name); nd = BAL_NODE[name]; k = f'var{nd + 1}'
        Cs = [r['C'] for r in rs]
        ex = [rel(r['ob'], k, 'exact') for r in rs]
        res = [rel(r['ob'], k, 'exact') - rel(r['ob'], k, 'loop') for r in rs]
        out[name] = dict(rows=rs, node=nd, C=Cs, exact=ex, resid=res, eps=[r['eps'][nd] for r in rs],
                         fit=fit_law(rs, k, nd), cross={lv: crossing(Cs, ex, lv) for lv in (0.01, 0.05, 0.10)},
                         chip_exact=float(np.interp(np.log(C_CHIP), np.log(Cs), ex)),
                         chip_resid=float(np.interp(np.log(C_CHIP), np.log(Cs), res)),
                         tree_check=max(r['ob']['tree_check'] for r in rs))
    return out


def compensation():
    """Exact error of the naive (tree) and one-loop-compensated designs at the two balance points, against C."""
    out = []
    for fn in sorted(glob.glob(os.path.join(DATA, 'comp_C*_vp*.json'))):
        d = json.load(open(fn)); t = np.array(d['target']); nd = 1 if d['g0'][0] == 120.0 else 2; ix = 0 if nd == 1 else 2
        pred = np.array(d['iters'][-1]['pred']) / t - 1
        out.append(dict(C=d['C_aF'], node=nd, naive=float(np.array(d['me_naive'])[ix] / t[ix] - 1),
                        comp=float(np.array(d['me_comp'])[ix] / t[ix] - 1), converged=bool(np.max(np.abs(pred)) < 1e-3),
                        dvdd=float(np.array(d['g_comp'])[4 + nd - 1] - d['g0'][4 + nd - 1])))
    return sorted(out, key=lambda r: (r['node'], -r['C']))

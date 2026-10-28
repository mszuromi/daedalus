"""Exact (to truncation) numerical solution of the NEAT-RN master equation (Eq. B1) on the charge lattice.

State n in Z^2 (net charge on each free node, n = 0 <-> v = v*); x(n) = s B n (units of V_T).
Generator on functions:  (Q f)(n) = sum_i W+_i(n) [f(n + h e_i) - f(n)] + W-_i(n) [f(n - h e_i) - f(n)].
h = 1 is the exact jump process.  For h < 1 the same construction with rates
    r+-_i = S_i/(2 s h^2) +- J_i/(2 s h)
is the central-difference discretisation of the Kramers-Moyal (Ito SDE, Eq. B2) Fokker-Planck operator; at
h = 1 it coincides with the jump process, and every observable is M(h) = M_KM + c2 h^2 + O(h^4), so
M_KM = (4 M(1/2) - M(1))/3 + O(h^4).

Stationary law: sparse LU of Q^T with one row replaced by a normalisation pin.  Lagged correlations
X_jk(tau) = sum_n p(n) f_j(n) [e^{tau Q} f_k](n), f = x - <x>, by shift-and-invert Arnoldi on (I - gam Q)^{-1}
(one LU, all tau at once).  Reflecting box of half-widths K * sigma_n (tree level).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm


def build(P, K=9.0, h=1.0, box=None):
    s, B = P['s'], P['B']
    Bi = np.linalg.inv(B)
    Sn = Bi @ P['Sigma'] @ Bi / s ** 2                     # tree covariance of n
    if box is None:
        N = np.ceil(K * np.sqrt(np.diag(Sn)) / h).astype(int)
    else:
        N = np.asarray(box, int)
    g1 = np.arange(-N[0], N[0] + 1) * h
    g2 = np.arange(-N[1], N[1] + 1) * h
    n1, n2 = np.meshgrid(g1, g2, indexing='ij')
    n1, n2 = n1.ravel(), n2.ravel()
    x1 = s * (B[0, 0] * n1 + B[0, 1] * n2)
    x2 = s * (B[1, 0] * n1 + B[1, 1] * n2)
    M1, M2 = len(g1), len(g2)
    idx = np.arange(M1 * M2).reshape(M1, M2)
    rows, cols, vals = [], [], []
    diag = np.zeros(M1 * M2)
    for i, x in ((0, x1), (1, x2)):
        cp, cn, al, be = P['cp'][i], P['cn'][i], P['alpha'][i], P['beta'][i]
        J = al * (1 - np.exp(x)) - be * (1 - np.exp(-x))
        S = cp + cn + al * np.exp(x) + be * np.exp(-x)
        up = S / (2 * s * h * h) + J / (2 * s * h)
        dn = S / (2 * s * h * h) - J / (2 * s * h)
        if dn.min() < 0:
            raise ValueError('negative rate: h too large for the KM discretisation')
        for sgn, rate in ((+1, up), (-1, dn)):
            I = idx.copy()
            if i == 0:
                src = idx[:-1, :] if sgn > 0 else idx[1:, :]
                dst = idx[1:, :] if sgn > 0 else idx[:-1, :]
            else:
                src = idx[:, :-1] if sgn > 0 else idx[:, 1:]
                dst = idx[:, 1:] if sgn > 0 else idx[:, :-1]
            src, dst = src.ravel(), dst.ravel()
            rows.append(src); cols.append(dst); vals.append(rate[src])
            np.add.at(diag, src, -rate[src])
    rows.append(np.arange(M1 * M2)); cols.append(np.arange(M1 * M2)); vals.append(diag)
    Q = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(M1 * M2, M1 * M2))
    return dict(Q=Q, x=np.stack([x1, x2], 1), N=N, shape=(M1, M2), h=h, center=int(idx[N[0], N[1]]))


def stationary(L):
    Q, c = L['Q'], L['center']
    n = Q.shape[0]
    M = Q.T.tocsr()
    mask = np.ones(n); mask[c] = 0.0
    M = sp.diags(mask) @ M + sp.csr_matrix(([1.0], ([c], [c])), shape=(n, n))
    rhs = np.zeros(n); rhs[c] = 1.0
    lu = spla.splu(M.tocsc(), permc_spec='COLAMD')
    p = lu.solve(rhs)
    p = np.maximum(p, 0.0)
    p /= p.sum()
    resid = np.abs(Q.T @ p).sum() / np.abs(Q.diagonal() * p).sum()
    return p, resid


def moments(L, p):
    x = L['x']
    mu = p @ x
    f = x - mu
    Sig = (f * p[:, None]).T @ f
    # mass on the box edge (truncation diagnostic)
    M1, M2 = L['shape']
    P2 = p.reshape(M1, M2)
    edge = P2[0, :].sum() + P2[-1, :].sum() + P2[:, 0].sum() + P2[:, -1].sum()
    # third/fourth standardized cumulants (non-Gaussianity)
    k3 = np.array([p @ f[:, i] ** 3 for i in range(2)])
    k4 = np.array([p @ f[:, i] ** 4 - 3 * Sig[i, i] ** 2 for i in range(2)])
    return dict(mu=mu, Sigma=Sig, edge_mass=edge, skew=k3 / np.diag(Sig) ** 1.5, exkurt=k4 / np.diag(Sig) ** 2)


def lagged(L, p, mu, taus, t_I=None, m=80, gam=None, P=None):
    """X_jk(tau) for tau >= 0 (array (ntau, 2, 2)); time averages Xbar_jk over [0, t_I]; integrals
    int_0^inf X_jk.  Shift-and-invert Arnoldi with pole 1/gam, Krylov dimension m."""
    Q = L['Q']
    n = Q.shape[0]
    if gam is None:
        lam = np.sort(np.abs(np.linalg.eigvals(P['A'])))
        gam = 1.0 / np.sqrt(lam[0] * lam[1])
    lu = spla.splu((sp.identity(n, format='csc') - gam * Q.tocsc()), permc_spec='COLAMD')
    f = L['x'] - mu
    taus = np.asarray(taus, float)
    X = np.zeros((len(taus), 2, 2)); Xbar = np.zeros((2, 2)); Xinf = np.zeros((2, 2))
    conv = {}
    for k in range(2):
        v0 = f[:, k].copy()
        beta = np.sqrt(v0 @ v0)
        V = np.zeros((n, m + 1)); H = np.zeros((m + 1, m))
        V[:, 0] = v0 / beta
        for j in range(m):
            wv = lu.solve(V[:, j])
            for _ in range(2):                                   # full re-orthogonalisation (CGS2)
                hcol = V[:, :j + 1].T @ wv
                wv -= V[:, :j + 1] @ hcol
                H[:j + 1, j] += hcol
            H[j + 1, j] = np.sqrt(wv @ wv)
            V[:, j + 1] = wv / H[j + 1, j]
        res = {}
        for mm in (m - 20, m):
            Hm = H[:mm, :mm]
            Qm = (np.eye(mm) - np.linalg.inv(Hm)) / gam
            e1 = np.zeros(mm); e1[0] = beta
            W = (p[:, None] * f).T @ V[:, :mm]                   # (2, mm): <p f_j, v_l>
            Xk = np.array([W @ (expm(Qm * t) @ e1) for t in taus])   # (ntau, 2): X_{j k}(tau)
            Xinf_k = W @ (-np.linalg.solve(Qm, e1))
            Xbar_k = None
            if t_I is not None:
                Aug = np.zeros((mm + 1, mm + 1)); Aug[:mm, :mm] = Qm; Aug[:mm, mm] = e1
                Xbar_k = W @ expm(Aug * t_I)[:mm, mm] / t_I
            res[mm] = (Xk, Xinf_k, Xbar_k)
        Xk, Xinf_k, Xbar_k = res[m]
        X[:, :, k] = Xk; Xinf[:, k] = Xinf_k
        if t_I is not None:
            Xbar[:, k] = Xbar_k
        conv[k] = float(np.max(np.abs(res[m][0] - res[m - 20][0])) / np.max(np.abs(Xk)))
    return dict(X=X, Xbar=Xbar if t_I is not None else None, Xinf=Xinf, conv=conv, gam=gam)


def solve_point(P, taus, t_I=None, K=9.0, h=1.0, m=80, verbose=True):
    t0 = time.perf_counter()
    L = build(P, K=K, h=h)
    p, resid = stationary(L)
    mo = moments(L, p)
    t1 = time.perf_counter()
    lag = lagged(L, p, mo['mu'], taus, t_I=t_I, m=m, P=P)
    t2 = time.perf_counter()
    if verbose:
        print(f'  ME h={h} N={L["N"]} states={L["Q"].shape[0]} resid={resid:.1e} edge={mo["edge_mass"]:.1e} '
              f'krylov_conv={lag["conv"]}  stat {t1-t0:.1f}s lag {t2-t1:.1f}s', flush=True)
    return dict(mu=mo['mu'], Sigma=mo['Sigma'], edge_mass=mo['edge_mass'], skew=mo['skew'], exkurt=mo['exkurt'],
                X=lag['X'], Xbar=lag['Xbar'], Xinf=lag['Xinf'], krylov_conv=lag['conv'], resid=resid,
                nstates=L['Q'].shape[0], N=L['N'].tolist(), h=h)

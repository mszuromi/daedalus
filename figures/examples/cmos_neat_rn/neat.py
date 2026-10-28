"""NEAT-RN (Freitas et al., arXiv 2605.26267): two subthreshold CMOS inverters joined by a capacitance matrix.

Exact model = Markov jump process on the node charges n in Z^2 (App. B), EKV subthreshold currents with
slope factors n = 1 and no DIBL (App. C).  Everything here is dimensionless:
    voltages in V_T = k_B T / q_e,   currents in I0p,   time in t_u = C V_T / I0p,   s = C_T / C,
where C is the diagonal entry of the Maxwell matrix, C_Max = C [[1, -r], [-r, 1]] (r = C_m / C) and
C_T = q_e^2 / (k_B T) = q_e / V_T.

Node i (output of inverter i) gains one charge (n_i -> n_i + 1) at rate W+_i = (cp_i + cn_i e^{-v_i}) / s
(PMOS forward + NMOS backward) and loses one at rate W-_i = (cp_i e^{v_i - Vdd_i} + cn_i) / s; these are the
thermodynamically consistent rates of Eq. (B3) with I_p = cp (1 - e^{v - Vdd}), I_n = cn (1 - e^{-v}).
A jump of node i moves the voltages by +/- s b_i, b_i = column i of B = C C_Max^{-1}.
Jump cumulants:  kappa_m(v) = s^{m-1} sum_i T_m,i(v_i) b_i^{(x) m},
    T_odd = J_i = cp (1 - e^{v-Vdd}) - cn (1 - e^{-v}),   T_even = S_i = cp (1 + e^{v-Vdd}) + cn (1 + e^{-v}).
About the fixed point (J_i(v_i*) = 0), with x = v - v*, alpha = cp e^{v*-Vdd}, beta = cn e^{-v*}:
    J_i = alpha (1 - e^x) - beta (1 - e^{-x}),   S_i = cp + cn + alpha e^x + beta e^{-x}.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm

KB = 1.380649e-23
QE = 1.602176634e-19
T_K = 295.15                       # 22 C, Fig. 5(c) of the paper
VT = KB * T_K / QE                 # 25.43 mV
CT = QE / VT                       # 6.30 aF
I0P = 1e-9                         # A
VDD = np.array([137e-3, 142e-3])   # V
VN = np.array([76e-3, 76e-3])      # V
VP_SUM = 200e-3                    # V_p^1 + V_p^2
# Balance (c_n = c_p) points read off Fig. 5(c) (T_eff peaks / f0 minima): inverter 1 at V_p^1 = 120 mV,
# inverter 2 at V_p^1 = 64 mV (V_p^2 = 136 mV).  This fixes I0n_i (threshold absorbed): cn_i = e^{(Vdd_i - Vp_bal,i)/V_T}.
VP_BAL = np.array([120e-3, 136e-3])
R_DEFAULT = 0.2                    # C_m / C; equilibrium Pearson correlation = r (Fig. 7(b) range 0.11-0.23)


def point(vp1_mV, C_aF, r=R_DEFAULT, vp2_mV=None, vn_mV=None, vdd_mV=None):
    """All dimensionless quantities at one operating point (V_p^2 = 200 mV - V_p^1 unless given)."""
    vdd = VDD.copy() if vdd_mV is None else np.asarray(vdd_mV, float) * 1e-3
    vn = VN.copy() if vn_mV is None else np.asarray(vn_mV, float) * 1e-3
    vp = np.array([vp1_mV * 1e-3, (VP_SUM - vp1_mV * 1e-3) if vp2_mV is None else vp2_mV * 1e-3])
    Vdd = vdd / VT
    cp = np.exp((vdd - vp) / VT)
    cn = np.exp((VDD - VP_BAL) / VT) * np.exp((vn - VN) / VT)   # I0n fixed by the default-point balance
    # fixed point, Eq. (C4), in a cancellation-free form: e^{v*} = e^{Vdd}/2 [(1-q) + sqrt((1-q)^2 + 4 q e^{-Vdd})]
    q = cn / cp
    disc = np.sqrt((1 - q) ** 2 + 4 * q * np.exp(-Vdd))
    ev = np.where(q < 1, np.exp(Vdd) / 2 * ((1 - q) + disc),
                  np.exp(Vdd) / 2 * (4 * q * np.exp(-Vdd)) / (disc - (1 - q)))
    vstar = np.log(ev)
    for _ in range(3):                                   # polish: J(v*) = 0
        J = cp * (1 - np.exp(vstar - Vdd)) - cn * (1 - np.exp(-vstar))
        dJ = -cp * np.exp(vstar - Vdd) - cn * np.exp(-vstar)
        vstar = vstar - J / dJ
    alpha = cp * np.exp(vstar - Vdd)
    beta = cn * np.exp(-vstar)
    s = CT / (C_aF * 1e-18)
    B = np.array([[1.0, r], [r, 1.0]]) / (1 - r * r)
    g = alpha + beta                                     # G_i V_T / I0p
    S0 = cp + cn + alpha + beta
    A = -B @ np.diag(g)                                  # drift Jacobian dx/dt = A x
    D = s * B @ np.diag(S0) @ B                          # kappa_2(v*)
    Sigma = solve_continuous_lyapunov(A, -D)
    t_u = C_aF * 1e-18 * VT / I0P                        # seconds per time unit
    return dict(vp1_mV=vp1_mV, vp_mV=vp * 1e3, C_aF=C_aF, r=r, Vdd=Vdd, cp=cp, cn=cn, vstar=vstar,
                alpha=alpha, beta=beta, g=g, S0=S0, s=s, B=B, A=A, D=D, Sigma=Sigma, t_u=t_u,
                Teff=S0 / (2 * g))                       # k_B T_eff / (k_B T) per inverter


def X_tree(P, taus):
    """Tree-level lagged correlations X_jk(tau) = <x_j(0) x_k(tau)> = [Sigma e^{A^T tau}]_jk, tau >= 0 (Eq. F1)."""
    return np.array([P['Sigma'] @ expm(P['A'].T * t) for t in np.atleast_1d(taus)])


def X_tree_any(P, taus):
    """X_jk(tau) for tau of either sign (X_jk(-tau) = X_kj(tau))."""
    out = []
    for t in np.atleast_1d(taus):
        M = P['Sigma'] @ expm(P['A'].T * abs(t))
        out.append(M if t >= 0 else M.T)
    return np.array(out)


def paper_lyapunov_SI(P):
    """Eq. (2) built from the paper's own Appendix-C formulas (SI units), returned in V_T^2.
    G = V_T^{-1} sqrt((c_p - c_n)^2 + 4 c_p c_n e^{-Vdd});  k_B T = q_e I*/(2G) [coth((Vdd - v*)/2) + coth(v*/2)],
    I* = (c_p + c_n - V_T G)/2;  G Sigma C + C Sigma G = 2 k_B G^{1/2} T G^{1/2}."""
    cpA, cnA = P['cp'] * I0P, P['cn'] * I0P
    Vdd, v = P['Vdd'], P['vstar']
    G = np.sqrt((cpA - cnA) ** 2 + 4 * cpA * cnA * np.exp(-Vdd)) / VT
    Istar = (cpA + cnA - VT * G) / 2
    kT = QE * Istar / (2 * G) * (1 / np.tanh((Vdd - v) / 2) + 1 / np.tanh(v / 2))
    C = P['C_aF'] * 1e-18 * np.array([[1, -P['r']], [-P['r'], 1]])
    Gm = np.diag(G)
    # G S C + C S G = Q  <=>  (C^{-1} G) S + S (G C^{-1}) = C^{-1} Q C^{-1}
    Ci = np.linalg.inv(C)
    Q = 2 * np.diag(np.sqrt(G)) @ np.diag(kT) @ np.diag(np.sqrt(G))
    Sig = solve_continuous_lyapunov(Ci @ Gm, Ci @ Q @ Ci)
    return Sig / VT ** 2, G, kT


def taylor_coeffs(P, deg_F=3, deg_K2=2, deg_K3=1, deg_K4=0):
    """Per-inverter Taylor coefficients in x = v_i - v_i*:
    J_i: [(-alpha + (-1)^m beta)/m!]_m,  S_i: [S0, (alpha + (-1)^m beta)/m!]_m."""
    from math import factorial
    out = []
    for i in range(2):
        a, b = P['alpha'][i], P['beta'][i]
        J = [0.0] + [(-a + (-1) ** m * b) / factorial(m) for m in range(1, 5)]
        S = [P['S0'][i]] + [(a + (-1) ** m * b) / factorial(m) for m in range(1, 5)]
        out.append(dict(J=J, S=S))
    return out


def observables(Sig, X12_bar=None, X21_bar=None):
    """Paper's covariance descriptors: aspect ratio alpha = sqrt(l0/l1), angle theta = arctan(q0_1/q0_2),
    Pearson rho; with time-averaged X's also the normalized gyration l."""
    w, V = np.linalg.eigh(Sig)
    l0, l1 = w[1], w[0]
    q0 = V[:, 1]
    theta = np.arctan(q0[0] / q0[1])
    rho = Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])
    out = dict(aspect=np.sqrt(l0 / l1), theta=theta, rho=rho, var1=Sig[0, 0], var2=Sig[1, 1], cov12=Sig[0, 1])
    if X12_bar is not None:
        out['l'] = (X12_bar - X21_bar) / np.sqrt(Sig[0, 0] * Sig[1, 1])
    return out


SWEEP_MV = np.arange(30.0, 160.1, 5.0)        # V_p^1 sweep (Fig. 1 / App. E range ~28-158 mV)


if __name__ == '__main__':
    print(f'V_T = {VT*1e3:.3f} mV, C_T = {CT*1e18:.3f} aF')
    for C in (300, 1000):
        print(f'--- C = {C} aF:  s = C_T/C = {CT/(C*1e-18):.5f},  t_u = {C*1e-18*VT/I0P*1e9:.2f} ns')
        for vp1 in (30, 64, 90, 120, 160):
            P = point(vp1, C)
            Sp, G, kT = paper_lyapunov_SI(P)
            err = np.max(np.abs(Sp - P['Sigma'])) / np.max(np.abs(P['Sigma']))
            o = observables(P['Sigma'])
            f0 = G / (2 * np.pi * C * 1e-18)
            print(f'Vp1={vp1:5.1f}: v*={np.round(P["vstar"],3)}  Teff/T={np.round(P["Teff"],2)} '
                  f'(paper kT/kBT {np.round(kT/(KB*T_K),2)})  f0(MHz, C_i=C)={np.round(f0/1e6,2)}  '
                  f'var={np.round(np.diag(P["Sigma"]),4)} sd={np.round(np.sqrt(np.diag(P["Sigma"])),3)} '
                  f'rho={o["rho"]:.3f}  |Sigma - Eq.2|/|Sigma| = {err:.1e}')

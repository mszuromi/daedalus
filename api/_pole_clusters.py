"""
Near-degenerate and repeated poles of the propagator.

Phase J writes every propagator entry as a pole-residue sum
``G(t) = sum_a C_a exp(lambda_a t)`` (``lambda_a = i p_a``) and adds up pole
tuples, one closed-form integral per tuple.  When poles come close (a gap
``g``), the residues of the entries that see the cluster grow like ``1/g``
and the tuple sum cancels terms of size ``g**-E`` (``E`` cluster-carrying
lines) down to ``O(1)``: about ``E log10(1/g)`` digits are lost in double
precision, whatever the accuracy of the residues.  At ``g = 0`` (a Jordan
block) the representation is not merely ill-conditioned but incomplete (no
``t exp(lambda t)`` term).

The fix used here does not touch the integrators.  The result ``F`` of
Phase J (any k, loop order and time) depends on the propagator only through
the rational matrix ``G(lambda)``.  For each ill-conditioned cluster ``c``
(cluster polynomial ``r_c(lambda) = prod_k (lambda - lambda_k)^{m_k}``,
degree ``m_c``) every entry that has poles in ``c`` is deformed to::

    G_ij,w = P_ij (r_c / r_c,ij) / (Q_ij,rest (r_c - w h_c^{m_c}))

(``r_c,ij``: the part of the entry's own denominator in ``c``; entries
without poles in ``c`` are left alone).  ``G_0 = G`` and ``F(w)`` is
analytic for ``|w| < R_w = (d_lim / h_c)^{m_c}`` (``d_lim``: distance from
the cluster to the other poles and to the stability boundary).  On ``|w| =
1`` the cluster's poles sit on a circle of radius ``~h_c``, as well separated
as in a generic model, so every node run is well conditioned.  Cauchy's
mean-value theorem gives ``F(0)`` as the average of ``F`` over the circle;
the N-point trapezoid rule converges geometrically (error ``~R_w^-N``).
For a real system ``F(conj w) = conj F(w)``, so only ``N/2 + 1`` node runs
are needed.  A repeated pole is the case ``r_c = (lambda - c)^m``: no special
case.  The error is estimated from the discrete Fourier tail of the node
values.

Screening is cheap (an O(n^2) look at the poles the exact tier already has);
only a close pair (``CLUSTER_RTOL``) or a repeated root triggers the exact
high-precision analysis, and only a cluster whose cancellation factor
``kappa`` exceeds ``KAPPA_MAX`` (or a repeated root) is deformed.  Models
without such a cluster take exactly the old path.  ``DAEDALUS_POLE_CLUSTERS=0``
switches the deformation off (old numbers, plus a warning when a cluster
needs it).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field

import mpmath as mp
import numpy as np


CLUSTER_RTOL = 5e-2        # single-linkage |p_i - p_j| / max(|p_i|, |p_j|)
KAPPA_MAX = 10.0           # deform a cluster whose cancellation factor exceeds this
H_FRACTION = 0.25          # deformation radius h = H_FRACTION * d_lim ...
H_TAU = 3.0                # ... and h * tau_max <= H_TAU (pointwise accuracy at large tau)
SEPARABILITY = 2.0         # h >= SEPARABILITY * cluster radius, else unresolved
ALIAS_TARGET = 1e-14       # a-priori trapezoid error target
N_NODES_MIN = 8
N_NODES_MAX = 48
ACCURACY_WARN = 1e-6       # warn above this estimated relative error
LOST_DIGITS_WARN = 10.0    # an unresolved cluster warns above this loss (1e-6)
TAU_MAX_DEFAULT = 50.0
ENV_SWITCH = 'DAEDALUS_POLE_CLUSTERS'
_DPS = 50                  # working precision of the root / residue arithmetic


class PoleClusterWarning(UserWarning):
    """A cluster of near-degenerate or repeated poles could not be resolved
    (or the resolution is switched off, or its accuracy estimate is poor):
    the numbers may have lost the stated number of digits."""


def enabled():
    """The deformation is on unless ``DAEDALUS_POLE_CLUSTERS`` is 0/off."""
    return os.environ.get(ENV_SWITCH, '1').strip().lower() not in (
        '0', 'off', 'false', 'no')


def has_close_pair(poles, rtol=None):
    """True when two of ``poles`` are closer than ``rtol`` (default
    ``CLUSTER_RTOL``) relative."""
    rtol = CLUSTER_RTOL if rtol is None else rtol
    ps = [complex(p) for p in poles]
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = ps[i], ps[j]
            if abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300):
                return True
    return False


# ── exact data (Frac(QQ[i][omega])) -> lambda = i omega, mpmath ─────────────

def _mpc(c):
    """A CyclotomicField(4) element ``a + b i`` as an mpmath complex."""
    v = list(c.list())
    a = v[0]
    b = v[1] if len(v) > 1 else 0
    return mp.mpc(mp.mpf(int(a.numerator())) / int(a.denominator()),
                  mp.mpf(int(b.numerator())) / int(b.denominator()))


def _to_lambda(poly, PR_lam):
    """``p(omega)`` -> ``p(-i lambda)`` (exact, same base field)."""
    CF = PR_lam.base_ring()
    m_i = -CF.gen()
    return PR_lam([c * m_i ** j for j, c in enumerate(poly.list())])


def _coeffs(poly):
    return [_mpc(c) for c in poly.list()]          # ascending


def _horner(coeffs, x):
    v = mp.mpc(0)
    for c in reversed(coeffs):
        v = v * x + c
    return v


def _is_real_poly(poly):
    return all(len(c.list()) < 2 or c.list()[1] == 0 for c in poly.list())


def _roots(coeffs_asc):
    """Roots of a polynomial (ascending mpmath coefficients), polished."""
    deg = len(coeffs_asc) - 1
    if deg <= 0:
        return []
    if deg == 1:
        return [-coeffs_asc[0] / coeffs_asc[1]]
    desc = list(reversed(coeffs_asc))
    rts = mp.polyroots(desc, maxsteps=400, extraprec=4 * mp.mp.prec)
    out = []
    dcoef = [j * coeffs_asc[j] for j in range(1, deg + 1)]
    for r in rts:
        r = mp.mpc(r)
        for _ in range(6):                          # Newton polish
            d = _horner(dcoef, r)
            if d == 0:
                break
            step = _horner(coeffs_asc, r) / d
            r -= step
            if abs(step) <= mp.mpf(10) ** (-mp.mp.dps) * max(1, abs(r)):
                break
        out.append(r)
    return out


@dataclass
class _Entry:
    i: int
    j: int
    num: list                     # P_hat ascending (lambda)
    lc: object                    # leading coefficient of Q_hat_ij
    mult: dict                    # root index -> multiplicity in Q_hat_ij
    real: bool


@dataclass
class _Cluster:
    members: list                 # root indices
    mult: dict                    # root index -> multiplicity in Q_hat
    m: int                        # degree of r_c
    centroid: object
    radius: float
    kappa: float
    active: bool = False
    d_lim: float = None
    h: float = None
    r_coeffs: list = None         # r_c ascending (mpmath)
    R_w: float = None


@dataclass
class PoleDeformation:
    """The deformation of the active clusters; ``propagator_at(w)`` gives the
    ``(pole_vals, C_mats)`` of ``G_w`` in the pipeline's convention."""
    nf: int
    roots: list                   # distinct roots (lambda), mpmath
    retarded: list                # bool per root
    entries: list                 # [_Entry]
    clusters: list                # active [_Cluster]
    n_nodes: int
    real_system: bool
    a_priori_error: float

    def nodes(self, symmetric):
        """``[(w, weight, paired)]``: ``F(0) ~ sum weight * F(w)`` (paired
        nodes contribute ``weight * Re F(w)``, valid for a real result)."""
        N = self.n_nodes
        if symmetric:
            out = []
            for n in range(N // 2 + 1):
                w = complex(mp.expjpi(mp.mpf(2 * n) / N))
                if n in (0, N // 2):
                    w = complex(round(w.real), 0.0)
                    out.append((w, 1.0 / N, False))
                else:
                    out.append((w, 2.0 / N, True))
            return out
        return [(complex(mp.expjpi(mp.mpf(2 * n) / N)), 1.0 / N, False)
                for n in range(N)]

    def propagator_at(self, w):
        from sage.all import matrix, CDF
        with mp.workdps(_DPS):
            w = mp.mpc(w)
            cl_of = {}
            for ci, c in enumerate(self.clusters):
                for k in c.members:
                    cl_of[k] = ci
            # poles: retarded roots outside the active clusters + the
            # deformed roots of every active cluster
            poles = []                     # (lambda, ('root', k) | ('cl', ci))
            for k, rho in enumerate(self.roots):
                if k not in cl_of and self.retarded[k]:
                    poles.append((rho, ('root', k)))
            for ci, c in enumerate(self.clusters):
                kap = mp.mpf(c.h) ** c.m
                co = list(c.r_coeffs)
                co[0] = co[0] - w * kap
                for s in _roots(co):
                    poles.append((s, ('cl', ci)))
            rows = []
            for lam, tag in poles:
                C = [[0j] * self.nf for _ in range(self.nf)]
                for e in self.entries:
                    C[e.i][e.j] = complex(self._residue(e, lam, tag, w, cl_of))
                rows.append((complex(-1j * complex(lam)), C))
            rows.sort(key=lambda r: (r[0].imag, r[0].real))
            return ([p for p, _ in rows],
                    [matrix(CDF, C) for _, C in rows])

    def _residue(self, e, lam, tag, w, cl_of):
        kind, idx = tag
        touched = [ci for ci, c in enumerate(self.clusters)
                   if any(e.mult.get(k, 0) for k in c.members)]
        if kind == 'root':
            if e.mult.get(idx, 0) == 0:
                return 0
        elif idx not in touched:
            return 0
        num = _horner(e.num, lam)
        for ci in touched:
            c = self.clusters[ci]
            for k in c.members:
                num *= (lam - self.roots[k]) ** (c.mult[k] - e.mult.get(k, 0))
        den = e.lc
        for k, v in e.mult.items():
            if k in cl_of or v == 0:
                continue
            if kind == 'root' and k == idx:
                continue                       # simple root: derivative
            den *= (lam - self.roots[k]) ** v
        for ci in touched:
            c = self.clusters[ci]
            if kind == 'cl' and ci == idx:
                dco = [j * c.r_coeffs[j] for j in range(1, len(c.r_coeffs))]
                den *= _horner(dco, lam)
            else:
                den *= (_horner(c.r_coeffs, lam) - w * mp.mpf(c.h) ** c.m)
        return num / den


@dataclass
class ClusterAnalysis:
    """What the screen found.  ``status``: 'harmless' (close poles, small
    cancellation factor: no deformation, the 50-digit residues), 'resolved'
    (deformation built), 'unresolved' (needed but not possible: the 50-digit
    residues, or the old path for a repeated pole; a warning above
    ``LOST_DIGITS_WARN``), 'disabled' (needed, switched off: the old path).
    ``nominal``: (pole_vals, C_mats) at w = 0, residues to 50 digits (at a
    repeated root the Laurent 1/(lambda - rho) coefficient only)."""
    status: str
    clusters: list = field(default_factory=list)   # plain dicts (report)
    reason: str = ''
    deformation: PoleDeformation = None
    nominal: tuple = None          # (pole_vals, C_mats) at w = 0, 50 digits
    kappa_max: float = 0.0
    jordan: bool = False

    def report(self):
        d = dict(status=self.status, reason=self.reason,
                 clusters=self.clusters, kappa_max=self.kappa_max,
                 repeated_pole=self.jordan)
        if self.deformation is not None:
            d.update(n_nodes=self.deformation.n_nodes,
                     real_system=self.deformation.real_system,
                     a_priori_error=self.deformation.a_priori_error)
        return d


def analyze(G_ft_F, Q_poly, nf, tau_max=None):
    """Exact high-precision cluster analysis of the Tier-1 inverse
    ``G_ft_F`` (``Frac(QQ[i][omega])``, common denominator ``Q_poly``)."""
    from sage.all import PolynomialRing
    tau_max = float(tau_max) if tau_max else TAU_MAX_DEFAULT
    CF = Q_poly.base_ring()
    PR_lam = PolynomialRing(CF, 'lam_pc')
    with mp.workdps(_DPS):
        Qh = _to_lambda(Q_poly, PR_lam)
        dens = {}
        for i in range(nf):
            for j in range(nf):
                if G_ft_F[i, j] != 0:
                    dens[(i, j)] = _to_lambda(G_ft_F[i, j].denominator(),
                                              PR_lam)
        # an exact coprime base of Q's roots: squarefree, pairwise coprime
        # factors, each with ONE multiplicity in Q and in every entry's
        # denominator (gcd refinement in QQ[i][lambda]); root membership is
        # then exact, however close two roots are
        base = _coprime_base(Qh, list(dens.values()))
        roots, mult_Q, fac = [], [], []
        for b in base:
            v = _valuation(Qh, b)
            for r in _roots(_coeffs(b)):
                roots.append(r)
                mult_Q.append(v)
                fac.append(b)
        n = len(roots)
        if sum(mult_Q) != Qh.degree():
            raise ArithmeticError('coprime base does not account for Q')
        # retarded: decaying relative to the roots' own frequency scale, the
        # same rule as the double-precision tiers (audit W4b, ACC-65: an
        # absolute -1e-9 dropped a stable mode slower than 1e-9 silently,
        # which in faster time units was kept)
        from api._propagator import RETARDED_RTOL
        lam_scale = max((abs(r) for r in roots), default=mp.mpf(0)) or 1
        retarded = [mp.re(r) < -RETARDED_RTOL * lam_scale for r in roots]
        # entries: numerator, leading coefficient, per-root multiplicity
        entries = []
        for (i, j), D in dens.items():
            g = G_ft_F[i, j]
            P = _to_lambda(g.numerator(), PR_lam)
            lc = D.leading_coefficient()
            vb = {id(b): _valuation(D, b) for b in base}
            mult = {k: vb[id(fac[k])] for k in range(n) if vb[id(fac[k])]}
            if sum(mult.values()) != D.degree():
                raise ArithmeticError(f'entry ({i},{j}): coprime base does '
                                      'not account for its denominator')
            real = _is_real_poly(P * (1 / lc)) and _is_real_poly(D * (1 / lc))
            entries.append(_Entry(i, j, _coeffs(P), _mpc(lc), mult, real))
        jordan_roots = {k for k in range(n) if mult_Q[k] > 1}
        ret = [k for k in range(n) if retarded[k]]
        clusters = [_make_cluster(g, roots, mult_Q, n)
                    for g in _single_linkage(roots, ret)
                    if len(g) >= 2 or set(g) & jordan_roots]
        if not clusters:
            return ClusterAnalysis(
                status='harmless',
                nominal=_nominal(entries, roots, retarded, nf))
        for c in clusters:
            c.kappa = _kappa(entries, roots, c, mult_Q)
            c.active = (not math.isfinite(c.kappa)) or c.kappa > KAPPA_MAX
        kmax = max(c.kappa for c in clusters)
        jordan = any(c.active and any(c.mult[k] > 1 for k in c.members)
                     for c in clusters)
        info = [_cluster_report(c, roots) for c in clusters]
        nominal = _nominal(entries, roots, retarded, nf)
        if not any(c.active for c in clusters):
            return ClusterAnalysis(status='harmless', clusters=info,
                                   kappa_max=kmax, nominal=nominal)
        if not enabled():
            return ClusterAnalysis(
                status='disabled', clusters=info, kappa_max=kmax,
                jordan=jordan, nominal=nominal,
                reason=f'{ENV_SWITCH}=0: the deformation is switched off')
        active, reason = _separable(
            [c for c in clusters if c.active], roots, mult_Q, ret, n)
        if reason:
            return ClusterAnalysis(status='unresolved', clusters=info,
                                   kappa_max=kmax, jordan=jordan,
                                   nominal=nominal, reason=reason)
        # deformation radius: H_FRACTION of the distance to the other poles
        # and the stability boundary, at most H_TAU / tau_max (pointwise
        # accuracy at large tau) but never below SEPARABILITY * radius
        for c in active:
            h = H_FRACTION * c.d_lim
            if tau_max > 0:
                h = min(h, H_TAU / tau_max)
            c.h = max(h, SEPARABILITY * c.radius)
        # a cluster and its mirror image (conjugate in lambda) share h
        for c in active:
            for c2 in active:
                if c2 is not c and abs(mp.conj(c.centroid) - c2.centroid) <= (
                        1e-12 * max(1, abs(c.centroid))):
                    c.h = c2.h = min(c.h, c2.h)
        for c in active:
            co = [mp.mpc(1)]
            for k in c.members:
                for _ in range(c.mult[k]):
                    co = _polymul(co, [-roots[k], mp.mpc(1)])
            c.r_coeffs = co
            c.R_w = (c.d_lim / c.h) ** c.m
        N, err = _choose_nodes(active, tau_max)
        real_system = all(e.real for e in entries)
        deform = PoleDeformation(nf=nf, roots=roots, retarded=retarded,
                                 entries=entries, clusters=active,
                                 n_nodes=N, real_system=real_system,
                                 a_priori_error=err)
        info = ([_cluster_report(c, roots) for c in active]
                + [_cluster_report(c, roots) for c in clusters
                   if not c.active and not (set(c.members) & {
                       k for a in active for k in a.members})])
        return ClusterAnalysis(status='resolved', clusters=info,
                               kappa_max=kmax, jordan=jordan,
                               deformation=deform, nominal=nominal)


def _valuation(D, b):
    """The largest v with b^v | D (exact)."""
    v = 0
    while D.degree() >= b.degree():
        q, r = D.quo_rem(b)
        if r != 0:
            break
        D, v = q, v + 1
    return v


def _coprime_base(Q, dens):
    """Squarefree, pairwise coprime monic factors of ``Q`` whose roots each
    have a single multiplicity in ``Q`` and in every polynomial of ``dens``
    (all of which divide ``Q``): the squarefree parts of ``Q`` split by gcd
    against every denominator's multiplicity layers."""
    base = [s.monic() for s, _ in Q.squarefree_decomposition()
            if s.degree() >= 1]
    for D in [Q] + list(dens):
        new = []
        for b in base:
            # layers: g_t = the part of b with multiplicity >= t in D
            pieces, rest, prev = [], D, b
            while True:
                g = prev.gcd(rest)
                if g.degree() < 1:
                    break
                g = g.monic()
                if g.degree() < prev.degree():
                    pieces.append(prev // g)       # multiplicity exactly t-1
                prev = g
                rest = rest // g
            pieces.append(prev)                    # the top layer (or b/1)
            new.extend(x.monic() for x in pieces if x.degree() >= 1)
        base = new
    return base


def _single_linkage(roots, ret):
    lab = {k: k for k in ret}

    def find(a):
        while lab[a] != a:
            lab[a] = lab[lab[a]]
            a = lab[a]
        return a
    for a in ret:
        for b in ret:
            if a < b and abs(roots[a] - roots[b]) <= CLUSTER_RTOL * max(
                    abs(roots[a]), abs(roots[b])):
                lab[find(a)] = find(b)
    groups = {}
    for k in ret:
        groups.setdefault(find(k), []).append(k)
    return [sorted(g) for g in groups.values()]


def _make_cluster(members, roots, mult_Q, n):
    """A cluster of the given root indices: multiplicities, centroid,
    radius and ``d_lim`` (distance to the nearest other root of Q and to
    the stability boundary Re lambda = 0)."""
    members = sorted(members)
    mult = {k: mult_Q[k] for k in members}
    m = sum(mult.values())
    cen = sum(mult[k] * roots[k] for k in members) / m
    rad = float(max(abs(roots[k] - cen) for k in members))
    d_other = min([float(abs(roots[k] - cen)) - rad
                   for k in range(n) if k not in members] or [math.inf])
    d_stab = float(-mp.re(cen)) - rad
    return _Cluster(members=members, mult=mult, m=m, centroid=cen,
                    radius=rad, kappa=0.0, d_lim=min(d_other, d_stab))


def _separable(active, roots, mult_Q, ret, n):
    """Grow every active cluster by its nearest retarded neighbour until its
    deformation radius ``H_FRACTION * d_lim`` is at least ``SEPARABILITY``
    times its radius (overlapping clusters merge).  Returns ``(clusters,
    reason)``; ``reason`` is non-empty when a cluster cannot be separated
    (it reaches the stability boundary)."""
    clusters = list(active)
    for _ in range(len(roots) + 1):
        merged = False
        for i, c in enumerate(clusters):
            for c2 in clusters[i + 1:]:
                if set(c.members) & set(c2.members):
                    new = _make_cluster(set(c.members) | set(c2.members),
                                        roots, mult_Q, n)
                    new.kappa, new.active = max(c.kappa, c2.kappa), True
                    clusters = [x for x in clusters if x is not c
                                and x is not c2] + [new]
                    merged = True
                    break
            if merged:
                break
        if merged:
            continue
        bad = [c for c in clusters
               if not (H_FRACTION * c.d_lim >= SEPARABILITY * c.radius
                       and c.d_lim > 0)]
        if not bad:
            return clusters, ''
        c = bad[0]
        outside = [k for k in ret if k not in c.members]
        stab = float(-mp.re(c.centroid)) - c.radius
        if not outside or stab <= min(float(abs(roots[k] - c.centroid))
                                      - c.radius for k in outside):
            return clusters, (
                f'the cluster at omega = {complex(-1j * complex(c.centroid)):.6g} '
                f'(radius {c.radius:.3g}) lies too close to the stability '
                f'boundary (distance {stab:.3g}) to be split')
        near = min(outside, key=lambda k: abs(roots[k] - c.centroid))
        new = _make_cluster(set(c.members) | {near}, roots, mult_Q, n)
        new.kappa, new.active = c.kappa, True
        clusters = [x for x in clusters if x is not c] + [new]
    return clusters, 'cluster growth did not terminate'


def _polymul(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _kappa(entries, roots, c, mult_Q):
    """The cancellation factor of cluster ``c``: max over entries of
    ``sum|C_k| / max_r |sum C_k ((lambda_k - c)/s)^r| / s^0`` with ``s =
    max(d_lim, 2 radius)``, the scale on which the integrals vary (they are
    analytic up to the other poles and the stability boundary): the factor
    by which the residues exceed what they add up to.  Infinite for a
    repeated root."""
    members = c.members
    if any(mult_Q[k] > 1 for k in members):
        return math.inf
    s = max(c.d_lim if math.isfinite(c.d_lim) else 0.0, 2 * c.radius, 1e-300)
    worst = 0.0
    for e in entries:
        Cs = [_simple_residue(e, roots, k) if e.mult.get(k, 0) else 0
              for k in members]
        tot = sum(abs(x) for x in Cs)
        if tot == 0:
            continue
        mom = max(abs(sum(C * ((roots[k] - c.centroid) / s) ** r
                          for C, k in zip(Cs, members)))
                  for r in range(len(members)))
        worst = max(worst, float(tot / max(mom, mp.mpf(10) ** -(_DPS - 5))))
    return worst


def _simple_residue(e, roots, k):
    """Residue of ``P/D`` (lambda) at the simple root ``roots[k]`` of D."""
    lam = roots[k]
    den = e.lc
    for k2, v in e.mult.items():
        if k2 != k and v:
            den *= (lam - roots[k2]) ** v
    return _horner(e.num, lam) / den


def _laurent_residue(e, roots, k):
    """Residue (the 1/(lambda - rho) coefficient) at a multiple root."""
    v = e.mult[k]
    rho = roots[k]

    def phi(x):
        den = e.lc
        for k2, v2 in e.mult.items():
            if k2 != k and v2:
                den *= (x - roots[k2]) ** v2
        return _horner(e.num, x) / den
    return mp.diff(phi, rho, v - 1) / mp.factorial(v - 1)


def _nominal(entries, roots, retarded, nf):
    """(pole_vals, C_mats) at w = 0 to working precision: distinct retarded
    roots, residues (the Laurent 1/(lambda - rho) coefficient at a repeated
    root, which alone does NOT represent the entry)."""
    from sage.all import matrix, CDF
    rows = []
    for k, rho in enumerate(roots):
        if not retarded[k]:
            continue
        C = [[0j] * nf for _ in range(nf)]
        for e in entries:
            v = e.mult.get(k, 0)
            if v == 1:
                C[e.i][e.j] = complex(_simple_residue(e, roots, k))
            elif v > 1:
                C[e.i][e.j] = complex(_laurent_residue(e, roots, k))
        rows.append((complex(-1j * complex(rho)), C))
    rows.sort(key=lambda r: (r[0].imag, r[0].real))
    return [p for p, _ in rows], [matrix(CDF, C) for _, C in rows]


def _choose_nodes(active, tau_max):
    """Smallest even N in [N_NODES_MIN, N_NODES_MAX] whose a-priori
    trapezoid error is below ALIAS_TARGET: the analyticity term R_w^-N and
    the large-tau term (h tau_max)^(mN) / (mN)! (relative to the cluster's
    own decay at tau_max; best effort, N_NODES_MAX at most)."""
    def err(N):
        e = 0.0
        for c in active:
            e += math.exp(-N * math.log(c.R_w))
            x = c.h * tau_max
            if x > 0:
                mN = c.m * N
                e += math.exp(min(mN * math.log(x) - math.lgamma(mN + 1),
                                  700.0))
        return e
    for N in range(N_NODES_MIN, N_NODES_MAX + 1, 2):
        if err(N) <= ALIAS_TARGET:
            return N, err(N)
    return N_NODES_MAX, err(N_NODES_MAX)


def _cluster_report(c, roots):
    return dict(poles=[complex(-1j * complex(roots[k])) for k in c.members],
                multiplicities=[int(c.mult[k]) for k in c.members],
                relative_diameter=float(2 * c.radius / max(
                    abs(roots[k]) for k in c.members)),
                kappa=float(c.kappa), deformed=bool(c.active),
                h=None if c.h is None else float(c.h),
                d_lim=None if c.d_lim is None else float(c.d_lim))


def numeric_screen(pole_vals, C_mats):
    """The numerical tiers (the exact tier failed): close poles with a
    cancellation factor above ``KAPPA_MAX`` cannot be deformed (no exact
    rational propagator): status 'unresolved'."""
    ps = [complex(p) for p in pole_vals]
    if not has_close_pair(ps):
        return None
    lam = [1j * p for p in ps]
    n = len(ps)
    lab = list(range(n))

    def find(a):
        while lab[a] != a:
            a = lab[a]
        return a
    for a in range(n):
        for b in range(a + 1, n):
            if abs(ps[a] - ps[b]) <= CLUSTER_RTOL * max(abs(ps[a]), abs(ps[b])):
                lab[find(a)] = find(b)
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    Cn = [np.array([[complex(C[i, j]) for j in range(C.ncols())]
                    for i in range(C.nrows())]) for C in C_mats]
    info, kmax = [], 0.0
    for members in groups.values():
        if len(members) < 2:
            continue
        cen = sum(lam[k] for k in members) / len(members)
        s = max(abs(lam[k]) for k in members)
        kap = 0.0
        for i in range(Cn[0].shape[0]):
            for j in range(Cn[0].shape[1]):
                Cs = [Cn[k][i, j] for k in members]
                tot = sum(abs(x) for x in Cs)
                if tot == 0:
                    continue
                mom = max(abs(sum(C * ((lam[k] - cen) / s) ** r
                                  for C, k in zip(Cs, members)))
                          for r in range(len(members)))
                kap = max(kap, tot / max(mom, 1e-300))
        kmax = max(kmax, kap)
        info.append(dict(poles=[ps[k] for k in members],
                         multiplicities=[1] * len(members),
                         relative_diameter=float(max(abs(ps[a] - ps[b])
                                                     for a in members
                                                     for b in members) / s),
                         kappa=float(kap), deformed=False, h=None,
                         d_lim=None))
    if not info:
        return None
    if kmax <= KAPPA_MAX:
        return ClusterAnalysis(status='harmless', clusters=info,
                               kappa_max=kmax)
    return ClusterAnalysis(
        status='unresolved', clusters=info, kappa_max=kmax,
        reason='no exact rational propagator (the exact tier failed): '
               'poles and residues come from the numerical tier')


def lost_digits(kappa, max_ell):
    """Rough number of digits the direct pole-residue sums lose: kappa per
    cluster-carrying line, about 2 + 2 ell lines."""
    if not math.isfinite(kappa):
        return math.inf
    return max(0.0, (2 + 2 * max_ell) * math.log10(max(kappa, 1.0)))


# ── combining node results ──────────────────────────────────────────────────

def combine_values(values, nodes):
    """``sum weight * F`` over the nodes (``Re F`` for paired nodes)."""
    tot = None
    for v, (_, wt, paired) in zip(values, nodes):
        a = np.asarray(v, dtype=complex)
        a = wt * (a.real.astype(complex) if paired else a)
        tot = a if tot is None else tot + a
    return tot


def tail_estimate(values, nodes, N):
    """Error estimate of the trapezoid mean from the discrete Fourier tail
    of the node values (the last resolved coefficient ``|a_{N-1}|``, which
    also carries the rounding noise of the node runs); ``values`` per node,
    arrays of equal shape."""
    full = [None] * N
    for v, (w, _, paired) in zip(values, nodes):
        n = int(round((math.atan2(w.imag, w.real) / (2 * math.pi)) * N)) % N
        a = np.asarray(v, dtype=complex)
        full[n] = a
        if paired:
            full[(N - n) % N] = np.conj(a)
    if any(f is None for f in full):
        return None
    F = np.stack(full)                                  # (N, ...)
    a = np.fft.fft(F, axis=0) / N
    # the coefficients decay geometrically (ratio 1/R_w); the trapezoid
    # mean is off by a_N + a_2N + ..., bounded by the last resolved one
    return np.abs(a[N - 1])


def combine_callables(fns, nodes):
    def combined(*args):
        tot = 0j
        for fn, (_, wt, paired) in zip(fns, nodes):
            v = complex(fn(*args))
            tot += wt * (v.real if paired else v)
        return tot
    return combined


def warn(message):
    warnings.warn(message, PoleClusterWarning, stacklevel=3)

"""M2b (docs/integration_speedup_plan.md §3.1 L1b): the hardened scipy
quadrature fallback of Phase J.

``final_integral._integrate_polytope`` hands every region that no analytic
path answered (m >= 1, after the exact constant-row and structural-zero
tests) to scipy quadrature.  Before M2b that was scipy.nquad with scipy's
default tolerances (epsabs 1.49e-8, no breakpoints) over a ±200 box for the
directions whose bounds depend on inner variables.  A narrow peak (fast
poles) could then fall between every Gauss-Kronrod node of the first panel:
both rules returned ~0, the error estimate was ~0, and the panel was
accepted, losing the whole region.  An empty inner interval was still
sampled (at s = 0, outside the region).  With ``NQUAD_HARDENED`` (default
True; ``DAEDALUS_PHASE_J_NQUAD_HARDENED=0`` or the umbrella
``DAEDALUS_PHASE_J_LEGACY=1`` restore the old routines bit-for-bit) the
fallback is ``_integrate_polytope_hardened``:

* nested ``scipy.integrate.quad`` over the region's own bounds, widened by
  one ulp: the exact difference-bound-matrix (DBM) closure of its
  difference rows (``_dbm_closure``, Floyd-Warshall on exact rationals)
  and, when another row is present, an exact Fourier-Motzkin elimination
  of every row (``_fm_level_rows``); the Heaviside filter enforces every
  row pointwise;
* an empty interval is answered 0 without evaluating anything, and the
  integrand is never evaluated where a row is <= 0;
* an open direction is truncated at ``NQUAD_TAIL_K`` / κ beyond its
  farthest breakpoint -- κ = κ_min (the slowest decay rate of the modes)
  for unit difference rows that are edges (``_unit_edge_rows``, every
  Phase J region measured), else the rate that linear programming
  certifies along it (``_GeneralDecayCertificate``) -- and the strip up to
  a farther cut is added when the outermost cut's tail bound exceeds
  1e-11 × the result (the inner levels' cuts are moved out too when a
  bound of everything the cuts drop exceeds it, ``region_tail``; for unit
  edge rows that certificate is built on first need); or at the legacy
  ±200 without a decay
  certificate (all counted);
* epsrel = 1e-10 and epsabs = 1e-13 × min(a bound of |integrand| from the
  modes, the largest |integrand| at sample points of the region), never
  below 1e-16 × that bound, then a second pass with 1e-13 × |result| when
  that was more than 10x looser (floored at 1e-22 × the sampled largest
  |integrand|, counted and warned when that allows more than 1e-8
  relative error); the part (real or imaginary) of a level
  that is larger at its first node integrated first, the other with
  epsabs raised to 1e-10 × |first part|, both judged against the level's
  complex value; quad's subinterval limit raised by the modes'
  oscillation count, a call that reaches it repeated with
  ``NQUAD_LIMIT_MAX``, and a remaining QUADPACK flag counted and warned
  about; breakpoints at the external times, at every vertex coordinate of
  the inner slices (the kinks of the inner integral: from the closure's
  alternating paths, ``_dbm_kinks``, or by vertex enumeration,
  ``_VertexKinks``) and geometrically towards the ends of wide panels.

Everything here is model-free (hand-built constraint rows and modes) except
section F and the slow section G child, all built in a private cwd.
Section F uses the public ``single_population_spike_reset_test`` (its
guard-bailed m = 2 regions, served by the fallback, against the exact fan
formula; the warning aggregated over the grouped k = 2 τ-grid of
``compute_cumulants``; and, slow, over a call of the k = 3 tree's
callable) and the public ``ou_quartic`` (a cheap ``compute_cumulants``
build whose callables must open the warning scope).  The section G child
runs ``ou_quartic``, ``linear_hawkes``,
``single_population_linear_delta_spikes_test`` and ``multipopulation_test``
(no fallback calls: the default engine is bit-identical to the pre-M2b
one) and ``single_population_spike_reset_test`` (served by the fallback:
the flag-off engine and the legacy umbrella are bit-identical to it).

Exact references: the narrow-peak regions of sections B and B2, and the
close-pole, oscillating, far-mass, general-row, inner-cut (G2s) and
box-fibration (G4) regions of sections E and E2, were integrated exactly by
``Integrate`` in a local Mathematica 15 kernel (rational parameters, closed
form evaluated with ``N[..., 30]`` or ``N[..., 40]``); the m = 3 one of
section B also through an explicit piecewise decomposition of the region,
which agrees to 40 digits, and the close-pole and bounded oscillating ones
also by the closed-form triangle integral of each exponential term.
Section F compares with the harness's 50-digit mpmath fan formula
(``tests/tools/phase_j_subset_diff.py``, reference ``c``).

Run:  sage -python -m pytest tests/test_phase_j_nquad_hardening.py -q
"""
import contextlib
import math
import os
import random
import subprocess
import sys
import textwrap
import types
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import engine.integration.time_domain.final_integral as FI       # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The engine at this revision is the pre-M2b code (M2a, structural zeros).
# A later Phase J milestone that adds a number-moving flag must set it to its
# legacy value in the child of section G (or retire these comparisons).
_PRE_M2B_REV = '16b5564'
_FI_REL = 'engine/integration/time_domain/final_integral.py'
_GI_REL = 'engine/integration/time_domain/grouped_integral.py'


def _counter(name):
    return FI._RUNTIME_COUNTERS[name]


def _ems(*modes):
    """An ``EdgeModeSum`` with the given ``(C, λ)`` modes."""
    return FI.EdgeModeSum(ri=0, pi=0, delta_coeff=0j, modes=tuple(
        (complex(C), complex(lam)) for C, lam in modes))


def _resolved(rows, free):
    return [(list(a), c0 + sum(x * t for x, t in zip(e, free)))
            for (a, e, c0) in rows]


def _fast_eval(modes, rows, m, module=FI):
    """The per-diagram fast evaluator of ``module`` for these edge modes
    (the first ``len(modes)`` rows are the edges)."""
    return module._build_fast_subset_evaluator_from_modes(
        1.0 + 0j, list(modes), rows[:len(modes)], m)


class _Counting:
    """Wraps an integrand: counts calls and records the arguments."""

    def __init__(self, f):
        self.f = f
        self.calls = []
        mi = getattr(f, '_nquad_modes', None)
        if mi is not None:
            self._nquad_modes = mi

    def __call__(self, *args):
        self.calls.append(args)
        return self.f(*args)


@pytest.fixture
def hardened(monkeypatch):
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', True)


@pytest.fixture
def legacy_nquad(monkeypatch):
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', False)


def _rel(v, ref):
    return abs(complex(v) - ref) / abs(ref)


# ═══════════════════════════════════════════════════════════════════════
# A. the flag
# ═══════════════════════════════════════════════════════════════════════

def test_flag_initialisation_from_the_environment():
    f = FI._initial_phase_j_flags
    assert f({})['NQUAD_HARDENED'] is True
    for v in ('0', 'false', 'no', 'off', ' OFF '):
        assert f({'DAEDALUS_PHASE_J_NQUAD_HARDENED': v})[
            'NQUAD_HARDENED'] is False
    for v in ('', '1', 'true', 'yes', 'on'):
        assert f({'DAEDALUS_PHASE_J_NQUAD_HARDENED': v})[
            'NQUAD_HARDENED'] is True
    # the umbrella wins over the per-flag variable
    assert f({'DAEDALUS_PHASE_J_LEGACY': '1',
              'DAEDALUS_PHASE_J_NQUAD_HARDENED': '1'}) == {
        'THETA0_CONST_ROW_MODE': 'legacy_clip', 'STRUCTURAL_ZEROS': False,
        'NQUAD_HARDENED': False}
    with pytest.raises(ValueError):
        f({'DAEDALUS_PHASE_J_NQUAD_HARDENED': 'maybe'})
    # the module default (this process sets none of the variables)
    if not any(os.environ.get(v) for v in (
            'DAEDALUS_PHASE_J_LEGACY', 'DAEDALUS_PHASE_J_NQUAD_HARDENED')):
        assert FI.NQUAD_HARDENED is True and FI._nquad_hardened_on()


def test_flag_is_read_at_call_time(monkeypatch):
    fe = _fast_eval(MODES_A, ROWS_A, 2)
    args = (fe, _resolved(ROWS_A, [T_A]), [T_A], 2)
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', True)
    FI._reset_runtime_counters()
    on = FI._integrate_polytope(*args)
    assert _counter('nquad_hardened_calls') == 1
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', False)
    FI._reset_runtime_counters()
    off = FI._integrate_polytope(*args)
    assert _counter('nquad_hardened_calls') == 0
    assert on != off
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', 'yes')
    with pytest.raises(ValueError):
        FI._integrate_polytope(*args)


# ═══════════════════════════════════════════════════════════════════════
# B. narrow peaks: the default tolerance misses the mass, the hardened path
#    matches an exact reference
# ═══════════════════════════════════════════════════════════════════════
# m = 2.  Edges Δt_0 = t − s0 (two modes, one oscillating) and Δt_1 = s0 − s1;
# the region 0 < Δt_0 < 3, 0 < Δt_1 < 5, 2Δt_0 + Δt_1 < 7 (a three-term row)
# at t = 10.  No row bounds s1 alone, so the old routine integrated s1 over
# [-200, 200]: every node of its first Gauss-Kronrod panel lies outside the
# region (s1 in (3, 10)), where the filtered integrand is 0.
T_A = 10.0
ROWS_A = [((-1.0, 0.0), (1.0,), 0.0),      # Δt_0 = t − s0 > 0   (edge)
          ((1.0, -1.0), (0.0,), 0.0),      # Δt_1 = s0 − s1 > 0  (edge)
          ((-1.0, 1.0), (0.0,), 5.0),      # Δt_1 < 5
          ((1.0, 0.0), (-1.0,), 3.0),      # Δt_0 < 3
          ((1.0, 1.0), (-2.0,), 7.0)]      # 2Δt_0 + Δt_1 < 7
MODES_A = [_ems((2 - 1j, -12 + 3j), (1, -8)), _ems((1, -15))]
# Mathematica 15: Integrate[G0[u] G1[v], {u,0,1},{v,0,5}] + Integrate[G0[u]
# G1[v], {u,1,3},{v,0,7-2u}], G0[u] = (2-I) E^((-12+3I)u) + E^(-8u),
# G1[v] = E^(-15v); N[.., 40].
REF_A = complex('0.020098039215371682165877903004422540628666512'
                '-0.0026143790849673219385676940222087926372289719j')

# m = 3.  The chain s2 < s1 < s0 < t (edges Δt_0 = t − s0, Δt_1 = s0 − s1,
# Δt_2 = s1 − s2) with 0 < Δt_0 < 2, Δt_1 < 4, Δt_2 < 4 and
# 2Δt_0 + Δt_1 + Δt_2 < 5 at t = 10: the outermost s2 has no row of its own,
# so the old routine integrated it over [-200, 200].
T_B = 10.0
ROWS_B = [((-1.0, 0.0, 0.0), (1.0,), 0.0),     # Δt_0 (edge)
          ((1.0, -1.0, 0.0), (0.0,), 0.0),     # Δt_1 (edge)
          ((0.0, 1.0, -1.0), (0.0,), 0.0),     # Δt_2 (edge)
          ((1.0, 0.0, 0.0), (-1.0,), 2.0),     # Δt_0 < 2
          ((-1.0, 1.0, 0.0), (0.0,), 4.0),     # Δt_1 < 4
          ((0.0, -1.0, 1.0), (0.0,), 4.0),     # Δt_2 < 4
          ((1.0, 0.0, 1.0), (-2.0,), 5.0)]     # 2Δt_0 + Δt_1 + Δt_2 < 5
MODES_B = [_ems((1 + 2j, -10 + 4j)), _ems((3, -14), (-1, -9)),
           _ems((1, -11 - 2j))]
# Mathematica 15: Integrate[H0[u] H1[v] H2[w] Boole[2u+v+w<5], {u,0,2},
# {v,0,4}, {w,0,4}], H0 = (1+2I) E^((-10+4I)u), H1 = 3E^(-14v) - E^(-9v),
# H2 = E^((-11-2I)w); N[.., 40]; the same value (to 40 digits) from the
# explicit decomposition u < 1/2 (v < 1-2u: w < 4; else w < 5-2u-v) and
# u > 1/2 (v < 5-2u, w < 5-2u-v).
REF_B = complex('0.000498084295109259510204190553177064172112'
                '+0.001850027366810234684037411870880715966242j')

PEAKS = {'m2': (ROWS_A, MODES_A, T_A, REF_A, 2),
         'm3': (ROWS_B, MODES_B, T_B, REF_B, 3)}


@pytest.mark.parametrize('case', sorted(PEAKS))
def test_narrow_peak_old_default_misses_the_mass(case, legacy_nquad):
    rows, modes, t, ref, m = PEAKS[case]
    f = _Counting(_fast_eval(modes, rows, m))
    v = FI._integrate_polytope(f, _resolved(rows, [t]), [t], m)
    # every node of the first panel is outside the region: nothing found
    assert abs(v) < 1e-6 * abs(ref) and not f.calls


@pytest.mark.parametrize('case', sorted(PEAKS))
def test_narrow_peak_hardened_matches_the_exact_value(case, hardened):
    rows, modes, t, ref, m = PEAKS[case]
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(_fast_eval(modes, rows, m),
                               _resolved(rows, [t]), [t], m)
    assert _rel(v, ref) < 1e-12, (v, ref)
    assert _counter('nquad_hardened_calls') == 1
    assert _counter('nquad_hardened_quad_flags') == 0
    assert _counter('nquad_hardened_capped') == 0      # bounded region


def _pole_tuples(modes):
    """The grouped form of a product of mode sums: every combination of one
    mode per edge, ``(Π C, (λ per edge))``."""
    import itertools
    out = []
    for combo in itertools.product(*[ms.modes for ms in modes]):
        c = 1.0 + 0j
        for (ce, _l) in combo:
            c *= ce
        out.append((c, tuple(lam for (_c, lam) in combo)))
    return out


@pytest.mark.parametrize('case', sorted(PEAKS))
def test_grouped_mode_data_give_the_same_value(case, hardened):
    """The grouped dispatch passes its merged pole tuples (``kind='sum'``)
    with an integrand that carries no mode data."""
    rows, modes, t, ref, m = PEAKS[case]
    fe = _fast_eval(modes, rows, m)

    def plain(*args):                     # no ``_nquad_modes`` attribute
        return fe(*args)
    info = FI._NquadModes.from_pole_tuples(rows, _pole_tuples(modes))
    assert info.kind == 'sum'
    assert info.decay_rates() == fe._nquad_modes.decay_rates()
    v = FI._integrate_polytope(plain, _resolved(rows, [t]), [t], m,
                               mode_info=info)
    assert _rel(v, ref) < 1e-12, (v, ref)


# ═══════════════════════════════════════════════════════════════════════
# B2. peaks at kinks that the direct switches of an inner bound do not
#     give: rows that are not difference rows, and difference rows whose
#     vertex is reached through two inner variables
# ═══════════════════════════════════════════════════════════════════════
# Every edge is a single exponential mode; the regions are chosen so that the
# integrand of the outer level has a peak of width ~1/12 at a vertex
# coordinate far (>= 40) from every external time and every closure kink, in
# an interval wide enough for its panels there to be ~64 wide: without that
# kink as a breakpoint the peak falls between all nodes and the value is
# ~0.  The exact values are Mathematica 15 ``Integrate`` (rational
# parameters, ``N[.., 30]``), as iterated integrals over the region's
# pieces.
E12, E05, E005 = _ems((1, -12)), _ems((1, -0.5)), _ems((1, -0.05))
KINK_PEAKS = {
    # s0 < 0, s0 + s1 + 70 > 0, s1 < 0 (edges, rates 12, 12, 0.05): the
    # three-term row makes s1 > -70, a bound no single row states (the
    # closure leaves s1 open below).  Integrate[E^(12 s0) E^(-12(s0+s1+70))
    # E^(s1/20), {s1,-70,0}, {s0,-s1-70,0}].
    'm2-general-end': (
        [((-1.0, 0.0), (0.0,), 0.0), ((1.0, 1.0), (0.0,), 70.0),
         ((0.0, -1.0), (0.0,), 0.0)], [E12, E12, _ems((1, -0.2))], 2,
        5.97190978959758606768159395695e-9, 'fm'),
    # the same with s1 > -100 (rate 0.05) and s0 + s1 + 40 > 0: bounded,
    # the projection's end -40 is inside the closure's interval (-100, 0).
    'm2-general-bounded': (
        [((-1.0, 0.0), (0.0,), 0.0), ((1.0, 1.0), (0.0,), 40.0),
         ((0.0, -1.0), (0.0,), 0.0), ((0.0, 1.0), (0.0,), 100.0)],
        [E12, E12, E005, E005], 2, 4.67912986047601881710836696052e-5, 'fm'),
    # s0 < -70 - s1 and s0 < 0 (rates 12) inside the box (-300, 0)^2 (rate
    # 0.05 each): the fast part of the integrand of s1 is e^(-12|s1 + 70|)/24,
    # a peak at the vertex s0 = 0 = -70 - s1 in the middle of the interval;
    # the projection is the whole box.  Integrate over s1 < -70 (s0 < 0)
    # and s1 > -70 (s0 < -70 - s1), both with s0 > -300.
    'm2-general-interior': (
        [((-1.0, -1.0), (0.0,), -70.0), ((-1.0, 0.0), (0.0,), 0.0),
         ((1.0, 0.0), (0.0,), 300.0), ((0.0, -1.0), (0.0,), 0.0),
         ((0.0, 1.0), (0.0,), 300.0)], [E12, E12, E005, E005, E005], 2,
        6.52553902987459874819793042077e-16, 'vertex'),
    # m = 3: s0 < 0, s0 + s2 + 40 > 0, s1 > s2, s1 < 0, s2 > -100 (rates
    # 12, 12, 0.5, 0.05, 0.05): s2 > -40 only through s0.
    'm3-general': (
        [((-1.0, 0.0, 0.0), (0.0,), 0.0), ((1.0, 0.0, 1.0), (0.0,), 40.0),
         ((0.0, 1.0, -1.0), (0.0,), 0.0), ((0.0, -1.0, 0.0), (0.0,), 0.0),
         ((0.0, 0.0, 1.0), (0.0,), 100.0)], [E12, E12, E05, E005, E005], 3,
        1.03980661856707244541462219727e-4, 'fm'),
    # difference rows only: s1 > -70, s0 < s2 + 1, s0 < s1 + 1 (rates 12),
    # s1 < 0, s0 > -300, s2 < 0 (rates 0.05).  The fast part of the
    # exponent, minimised over the slice, is 12|s2 + 70|: a peak at the
    # vertex s1 = -70, s0 = s1 + 1 = s2 + 1, whose s2-coordinate is the
    # three-entry path D[0][1] − D[z][1] − D[0][2] of the closure.
    'm3-dbm-path': (
        [((0.0, 1.0, 0.0), (0.0,), 70.0), ((-1.0, 0.0, 1.0), (0.0,), 1.0),
         ((-1.0, 1.0, 0.0), (0.0,), 1.0), ((0.0, -1.0, 0.0), (0.0,), 0.0),
         ((1.0, 0.0, 0.0), (0.0,), 300.0), ((0.0, 0.0, -1.0), (0.0,), 0.0)],
        [E12, E12, E12, E005, E005, E005], 3,
        5.12767279156562626919245997384e-12, 'path'),
}


def _loose_tolerance(monkeypatch):
    """The first pass's epsabs from the modes' bound alone (no sample of
    |integrand|) and no second pass: the regime where QUADPACK accepts a
    panel whose nodes all miss a peak (the sampled scale is a lower bound of
    the supremum, often far below it, and makes the quadrature refine until
    it finds the peak anyway)."""
    monkeypatch.setattr(FI, '_NQUAD_PRESAMPLES', 0)
    monkeypatch.setattr(FI, '_NQUAD_RERUN_MAX', 0)


def _without(what, monkeypatch):
    """Remove the kink source a case needs (to show that it is needed)."""
    _loose_tolerance(monkeypatch)
    if what == 'path':
        monkeypatch.setattr(FI, '_dbm_path_kinks',
                            lambda D, m: [((), ())] * m)
    else:
        monkeypatch.setattr(FI._VertexKinks, 'at',
                            lambda self, k, outer: ())
        if what == 'fm':
            monkeypatch.setattr(FI, '_NQUAD_FM_MAX_ROWS', 0)


@pytest.mark.parametrize('case', sorted(KINK_PEAKS))
def test_peak_at_a_vertex_kink_matches_the_exact_value(case, hardened,
                                                       monkeypatch):
    rows, modes, m, ref, what = KINK_PEAKS[case]
    fe = _fast_eval(modes, rows, m)
    args = (fe, _resolved(rows, [0.0]), [0.0], m)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args)
    assert _rel(v, ref) < 1e-12, (v, ref)
    assert _counter('nquad_hardened_quad_flags') == 0
    assert _counter('nquad_hardened_general_rows') == (what != 'path')
    assert _counter('nquad_hardened_fm_capped') == 0
    assert _counter('nquad_hardened_kinks_incomplete') == 0
    # with the kink but a loose (bound-based) tolerance: still exact
    _loose_tolerance(monkeypatch)
    assert _rel(FI._integrate_polytope(*args), ref) < 1e-12
    # without that kink (and, for an end, without the exact projection) the
    # peak falls between all nodes and the region is lost
    _without(what, monkeypatch)
    lost = FI._integrate_polytope(*args)
    assert _rel(lost, ref) > 0.99, (lost, ref)


# ═══════════════════════════════════════════════════════════════════════
# C. empty intervals and points outside the region are never evaluated
# ═══════════════════════════════════════════════════════════════════════
# s0 + s1 > 10 and s0 + s1 < 5 (three-term rows: not in the DBM), inside the
# box s0, s1 in (-1, 1): the closure cannot see the contradiction, every
# inner interval of s0 is empty.
EMPTY_ROWS = [((1.0, 1.0), (), -10.0), ((-1.0, -1.0), (), 5.0),
              ((1.0, 0.0), (), 1.0), ((-1.0, 0.0), (), 1.0),
              ((0.0, 1.0), (), 1.0), ((0.0, -1.0), (), 1.0)]


def _rows_hold(rows, free, s):
    return all(c0 + sum(a * x for a, x in zip(a_int, s))
               + sum(e * t for e, t in zip(a_ext, free)) > 0.0
               for (a_int, a_ext, c0) in rows)


def test_empty_inner_intervals_are_not_sampled(hardened, monkeypatch):
    f = _Counting(lambda s0, s1: 1.0 + 0j)
    # the Fourier-Motzkin elimination of the three-term rows proves the
    # region empty: 0 without quadrature
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(f, _resolved(EMPTY_ROWS, []), [], 2)
    assert v == 0 and isinstance(v, complex)
    assert not f.calls
    assert _counter('nquad_hardened_general_rows') == 1
    assert _counter('nquad_hardened_empty') == 1
    # without the elimination (system over its row cap) the closure cannot
    # see the contradiction: every inner interval of s0 is empty, and none
    # is sampled
    monkeypatch.setattr(FI, '_NQUAD_FM_MAX_ROWS', 0)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(f, _resolved(EMPTY_ROWS, []), [], 2)
    assert v == 0 and isinstance(v, complex)
    assert not f.calls
    assert _counter('nquad_hardened_fm_capped') == 1
    assert _counter('nquad_hardened_empty') == 0
    assert _counter('nquad_hardened_empty_intervals') > 0


def test_old_routine_samples_empty_inner_intervals(legacy_nquad):
    """The pre-M2b behaviour the hardened path removes: the inner bounds
    (0, 0) are handed to quad, which evaluates the integrand there."""
    f = _Counting(lambda s0, s1: 1.0 + 0j)
    rows = [((1.0, -1.0), (), 0.0), ((-1.0, 1.0), (), 0.0),   # s0 = s1 only
            ((0.0, 1.0), (), 1.0), ((0.0, -1.0), (), 1.0)]
    # the order rows s0 > s1 > s0 are a cycle the structural-zero test
    # answers; with it off the region reaches quadrature
    with _flag('STRUCTURAL_ZEROS', False):
        v = FI._integrate_polytope(f, _resolved(rows, []), [], 2)
    assert v == 0
    assert len(f.calls) > 0          # sampled although the region is empty


@contextlib.contextmanager
def _flag(name, value):
    saved = getattr(FI, name)
    setattr(FI, name, value)
    try:
        yield
    finally:
        setattr(FI, name, saved)


@pytest.mark.parametrize('case', sorted(PEAKS))
def test_integrand_is_only_evaluated_inside_the_region(case, hardened,
                                                       monkeypatch):
    rows, modes, t, ref, m = PEAKS[case]
    f = _Counting(_fast_eval(modes, rows, m))
    # with mode data the innermost variable is integrated in closed form:
    # the integrand is never called at all
    v = FI._integrate_polytope(f, _resolved(rows, [t]), [t], m)
    assert not f.calls and _rel(v, ref) < 1e-12
    # quadrature on every level: only points inside the region
    monkeypatch.setattr(FI, '_NQUAD_ANALYTIC_INNERMOST', False)
    v = FI._integrate_polytope(f, _resolved(rows, [t]), [t], m)
    assert f.calls and _rel(v, ref) < 1e-12
    bad = [a for a in f.calls if not _rows_hold(rows, [t], a[:m])]
    assert not bad, bad[:3]


def _random_region(rng, m):
    """A bounded random region in the Phase J shape: the chain
    s_0 < ... < s_{m-1} < t (edge rows), a random lower bound per edge
    (difference / scalar rows), sometimes a 3-term row, and random modes
    (decaying, some oscillating) on the edges."""
    rows = []
    for i in range(m):
        a = [0.0] * m
        a[i] = -1.0
        if i + 1 < m:
            a[i + 1] = 1.0
            rows.append((tuple(a), (0.0,), 0.0))
        else:
            rows.append((tuple(a), (1.0,), 0.0))
    for i in range(m):                       # Δt_i < width_i
        a = [-x for x in rows[i][0]]
        rows.append((tuple(a), tuple(-x for x in rows[i][1]),
                     rng.choice((0.5, 1.0, 2.0, 4.0))))
    if m >= 2 and rng.random() < 0.5:
        a = [0.0] * m
        a[0] = a[m - 1] = 1.0
        rows.append((tuple(a), (-2.0,), rng.uniform(1.0, 4.0)))
    modes = [_ems(*[(complex(rng.uniform(-2, 2), rng.uniform(-1, 1)),
                     complex(-rng.uniform(0.2, 15.0), rng.choice(
                         (0.0, rng.uniform(-6, 6)))))
                    for _ in range(rng.randint(1, 2))]) for _ in range(m)]
    return rows, modes, [rng.uniform(-1.0, 1.0)]


@pytest.mark.parametrize('ms', [
    pytest.param((1, 2) * 6, id='m1-m2'),
    pytest.param((3,) * 4, id='m3', marks=pytest.mark.slow)])
def test_closed_form_innermost_matches_quadrature(ms, hardened, monkeypatch):
    """The closed-form innermost level against quadrature on every level
    (``_NQUAD_ANALYTIC_INNERMOST = False``), on random bounded regions."""
    rng = random.Random(3 + len(ms))
    for trial, m in enumerate(ms):
        rows, modes, free = _random_region(rng, m)
        fe = _fast_eval(modes, rows, m)
        args = (fe, _resolved(rows, free), free, m)
        monkeypatch.setattr(FI, '_NQUAD_ANALYTIC_INNERMOST', True)
        a = FI._integrate_polytope(*args)
        monkeypatch.setattr(FI, '_NQUAD_ANALYTIC_INNERMOST', False)
        q = FI._integrate_polytope(*args)
        scale = FI._NquadModes.bound(fe._nquad_modes, [-50] * m, [50] * m,
                                     free)
        assert abs(a - q) <= 1e-9 * abs(q) + 1e-13 * scale, (trial, a, q)


# The fast-pole 2-cycle of plan Appendix C.2 (s1 < 0, s0 < τ, s1 > s0,
# s0 > s1): the pre-M2a fallback raised OverflowError evaluating the fast
# evaluator far outside the region.
TWO_CYCLE = [((0.0, -1.0), (0.0,), 0.0), ((-1.0, 0.0), (1.0,), 0.0),
             ((-1.0, 1.0), (0.0,), 0.0), ((1.0, -1.0), (0.0,), 0.0)]
FAST = _ems((1.0, -12.0))


def test_two_cycle_reaches_no_evaluation_without_structural_zeros(
        hardened, monkeypatch):
    monkeypatch.setattr(FI, 'STRUCTURAL_ZEROS', False)
    free = [0.2]
    f = _Counting(_fast_eval([FAST] * 4, TWO_CYCLE, 2))
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(f, _resolved(TWO_CYCLE, free), free, 2,
                               raw_rows=TWO_CYCLE)
    assert v == 0 and not f.calls
    assert _counter('nquad_hardened_empty') == 1        # the DBM closure
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', False)
    with pytest.raises(OverflowError):
        FI._integrate_polytope(f, _resolved(TWO_CYCLE, free), free, 2,
                               raw_rows=TWO_CYCLE)


def test_tied_constant_edge_kept_by_the_tie_order(hardened, legacy_nquad,
                                                 monkeypatch):
    """At an exact external-time tie the Θ(0) rule can keep (DROP) a
    constant edge row Δt ≡ 0 (``_const_row_verdict``); the edge then
    contributes G(0).  One free time t = t_1 − t_0 with t_0 == t_1 == 0
    (``IDENTITY``: the row −t > 0 holds by the tie order).  Exact:
    G1(0)·∫_0^3 G0(u) du.  (A first version of the hardened fallback read
    the edge's Δt <= 0 as an empty region and returned 0: found on the
    spike-reset k = 4 tree at its tied slice points.)"""
    ident = FI._TieContext((0.0, 0.0), (1,), 0)
    rows = [((-1.0,), (1.0,), 0.0),       # Δt_0 = t − s0 (edge)
            ((0.0,), (-1.0,), 0.0),       # Δt_1 = −t ≡ 0 at the tie (edge)
            ((1.0,), (-1.0,), 3.0)]       # s0 > t − 3
    c0, l0, c1, l1 = 1.5 - 0.5j, -2.0 + 1.0j, 0.75 + 0.25j, -1.0
    fe = _fast_eval([_ems((c0, l0)), _ems((c1, l1))], rows, 1)
    exact = c1 * c0 * (np.exp(3 * l0) - 1) / l0
    free = [0.0]
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', True)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(fe, _resolved(rows, free), free, 1,
                               raw_rows=rows, tie_ctx=ident)
    assert _counter('theta0_tie_ordered') == 1
    assert _counter('nquad_hardened_calls') == 1
    assert _rel(v, exact) < 1e-13, (v, exact)
    monkeypatch.setattr(FI, 'NQUAD_HARDENED', False)       # old fallback
    v_old = FI._integrate_polytope(fe, _resolved(rows, free), free, 1,
                                   raw_rows=rows, tie_ctx=ident)
    assert _rel(v_old, exact) < 1e-8
    # the bound keeps the constant edge's box range (no clip to Δt > 0)
    B = fe._nquad_modes.bound([-3.0], [0.0], free)
    assert B == pytest.approx(abs(c0) * abs(c1))


def test_constant_row_and_closure_emptiness(hardened):
    f = _Counting(lambda *a: 1.0 + 0j)
    FI._reset_runtime_counters()
    # a constant row <= 0 (Θ(0) = 0) that reaches the routine directly
    assert FI._integrate_polytope_hardened(
        f, [((0.0, 0.0), 0.0), ((1.0, 0.0), 1.0)], [], 2) == 0
    # x0 > 1 and x0 < 1: a zero-weight cycle through the constant node
    assert FI._integrate_polytope_hardened(
        f, [((1.0,), -1.0), ((-1.0,), 1.0)], [], 1) == 0
    # a positive-width strip is not empty: ∫_1^{1+1e-9} 1 ds
    v = FI._integrate_polytope_hardened(
        f, [((1.0,), -1.0), ((-1.0,), 1.0 + 1e-9)], [], 1)
    assert v == pytest.approx(1e-9, rel=1e-6, abs=0.0)
    assert _counter('nquad_hardened_empty') == 2
    assert len(f.calls) > 0


# ═══════════════════════════════════════════════════════════════════════
# D. the DBM bounds equal brute-force bounds (randomized systems)
# ═══════════════════════════════════════════════════════════════════════

def _lp_range(rows, m, k, outer):
    """Brute force: [min, max] of s_k over {s : rows >= 0} with s_j fixed to
    ``outer`` for j > k, by linear programming (scipy HiGHS).  ``None`` if
    infeasible; ±inf when unbounded."""
    from scipy.optimize import linprog
    n = k + 1
    A, b = [], []
    for (a_int, c) in rows:
        a = np.asarray(a_int, float)
        cc = c + float(np.dot(a[k + 1:], outer)) if k + 1 < m else c
        A.append(-a[:n])
        b.append(cc)
    out = []
    for sign in (1.0, -1.0):
        obj = np.zeros(n)
        obj[k] = sign
        r = linprog(obj, A_ub=np.array(A), b_ub=np.array(b),
                    bounds=[(None, None)] * n, method='highs')
        if r.status == 2:
            return None
        if r.status == 3:
            out.append(-sign * math.inf)
        else:
            assert r.status == 0, r.message
            out.append(sign * r.fun)
    return out[0], out[1]


def _fm_range(rows, m, k, outer):
    """Brute force, EXACT: the closed projection [lo, hi] of
    {s : a·s + c >= 0 for every row} onto s_k with s_j = ``outer`` for
    j > k, by Fourier-Motzkin elimination of s_0..s_{k-1} in rational
    arithmetic (each normal direction keeps its tightest row).  ``None`` if
    infeasible; ±inf when unbounded."""
    from fractions import Fraction as Fr
    sys_ = {}

    def add(a, c):
        nz = [x for x in a if x != 0]
        if not nz:
            if c < 0:
                raise _Infeasible
            return
        sc = abs(nz[0])
        a = tuple(x / sc for x in a)
        c = c / sc
        if a not in sys_ or c < sys_[a]:
            sys_[a] = c

    class _Infeasible(Exception):
        pass
    try:
        for (a_int, c0) in rows:
            a = [Fr(float(x)) for x in a_int]
            c = Fr(float(c0)) + sum((a[j] * Fr(outer[j - k - 1])
                                     for j in range(k + 1, m)), Fr(0))
            add(a[:k + 1], c)
        for v in range(k):
            pos = [(a, c) for a, c in sys_.items() if a[v] > 0]
            neg = [(a, c) for a, c in sys_.items() if a[v] < 0]
            rest = {a: c for a, c in sys_.items() if a[v] == 0}
            sys_.clear()
            sys_.update(rest)
            for (ap, cp) in pos:
                for (an, cn) in neg:
                    wp, wn = -an[v], ap[v]
                    add(tuple(wp * x + wn * y for x, y in zip(ap, an)),
                        wp * cp + wn * cn)
    except _Infeasible:
        return None
    lo, hi = -math.inf, math.inf
    for a, c in sys_.items():
        assert all(x == 0 for x in a[:k])
        if a[k] > 0:
            lo = max(lo, -c / a[k])
        else:
            hi = min(hi, c / -a[k])
    if lo > hi:
        return None
    return lo, hi


def _random_system(rng, m, n_rows, general=False):
    rows = []
    for _ in range(n_rows):
        kind = rng.random()
        a = [0.0] * m
        if kind < 0.3 or m == 1:
            j = rng.randrange(m)
            a[j] = rng.choice((1.0, -1.0, 2.0, -0.5))
        elif kind < 0.9 or not general:
            i, j = rng.sample(range(m), 2)
            c = rng.choice((1.0, 1.0, 1.0, 3.0))
            a[i], a[j] = c, -c
        else:
            for j in rng.sample(range(m), min(m, 3)):
                a[j] = rng.choice((1.0, -1.0, 2.0))
        rows.append((a, float(rng.randint(-8, 8)) / rng.choice((1, 2, 4))))
    return rows


def _feasible_points(rows, m, rng, n=40, box=30.0):
    pts = []
    for _ in range(20000):
        s = [rng.uniform(-box, box) for _ in range(m)]
        if all(c + sum(a * x for a, x in zip(a_int, s)) > 0
               for (a_int, c) in rows):
            pts.append(s)
            if len(pts) >= n:
                break
    return pts


def test_dbm_bounds_equal_brute_force_bounds():
    """Difference rows only: at every level and at random feasible outer
    values, the closure's interval is the exact projection (LP)."""
    rng = random.Random(20261003)
    n_cmp = n_open = 0
    for trial in range(60):
        m = rng.choice((1, 2, 3, 4))
        rows = _random_system(rng, m, rng.randint(m, 3 * m + 2))
        D, general, verdict = FI._dbm_closure(rows, m)
        assert not general
        pts = _feasible_points(rows, m, rng, n=6)
        if verdict == 'EMPTY':
            assert not pts, (rows, pts)
            continue
        tabs = FI._dbm_level_tables(D, m)
        for s in pts:
            for k in range(m):
                outer = tuple(s[k + 1:])
                L, U = FI._dbm_interval(tabs[k], outer)
                exact = _fm_range(rows, m, k, list(outer))
                lp = _lp_range(rows, m, k, list(outer))
                assert exact is not None and lp is not None
                for got, want, alt in ((L, exact[0], lp[0]),
                                       (U, exact[1], lp[1])):
                    if math.isinf(want):
                        assert got == want == alt, (rows, k, outer, L, U)
                        n_open += 1
                    else:
                        # equal to the exact projection up to the rounding
                        # of one float addition, and to the LP
                        assert abs(got - float(want)) <= 4e-16 * (
                            1 + abs(got)), (rows, k, outer, L, U, exact)
                        assert abs(got - alt) <= 1e-9 * (1 + abs(alt))
                # with the integrator's one-ulp widening: never inside the
                # exact projection
                assert math.nextafter(L, -math.inf) <= exact[0]
                assert math.nextafter(U, math.inf) >= exact[1]
                n_cmp += 1
    assert n_cmp > 100 and n_open > 10


def test_level_intervals_never_cut_the_region():
    """With rows that are not difference rows (in the filter only, or in the
    level of their innermost variable): every point of the region lies
    inside the interval of every level given its outer coordinates, and the
    interval contains the exact projection."""
    rng = random.Random(7)
    n_pts = 0
    for trial in range(60):
        m = rng.choice((2, 3, 4))
        rows = _random_system(rng, m, rng.randint(m + 1, 3 * m + 2),
                              general=True)
        D, general, verdict = FI._dbm_closure(rows, m)
        pts = _feasible_points(rows, m, rng, n=8)
        if verdict == 'EMPTY':
            assert not pts
            continue
        tabs = FI._dbm_level_tables(D, m)
        gen = FI._general_rows_by_level(general, m)
        for s in pts:
            for k in range(m):
                outer = tuple(s[k + 1:])
                L, U = FI._level_interval(tabs[k], gen[k], outer)
                assert L < s[k] < U, (rows, k, s, L, U)
                exact = _fm_range(rows, m, k, list(outer))
                # a superset of the exact projection (to the rounding of a
                # few float operations on the non-difference rows)
                tol = 1e-14 * (1 + abs(s[k]))
                assert L <= exact[0] + tol and U >= exact[1] - tol, (
                    rows, k, outer, L, U, exact)
            n_pts += 1
    assert n_pts > 100


def _exact_vertex_coords(rows, m, k, outer):
    """Brute force, EXACT: the s_k-coordinate of every vertex of the
    closure of {(s_0..s_k) : a·s + c > 0} with s_j = ``outer`` for j > k:
    every (k+1)-subset of the rows, solved by Gaussian elimination in
    rationals, kept if every row holds (>= 0)."""
    import itertools
    from fractions import Fraction as Fr
    d = k + 1
    R = []
    for (a_int, c0) in rows:
        a = [Fr(float(x)) for x in a_int]
        c = Fr(float(c0)) + sum((a[j] * Fr(outer[j - d])
                                 for j in range(d, m)), Fr(0))
        R.append((a[:d], c))
    out = set()
    for sub in itertools.combinations(range(len(R)), d):
        M = [list(R[i][0]) + [-R[i][1]] for i in sub]
        ok = True
        for col in range(d):                # Gauss-Jordan
            piv = next((r for r in range(col, d) if M[r][col] != 0), None)
            if piv is None:
                ok = False
                break
            M[col], M[piv] = M[piv], M[col]
            for r in range(d):
                if r != col and M[r][col] != 0:
                    f = M[r][col] / M[col][col]
                    M[r] = [x - f * y for x, y in zip(M[r], M[col])]
        if not ok:
            continue
        x = [M[i][d] / M[i][i] for i in range(d)]
        if all(sum(ai * xi for ai, xi in zip(a, x)) + c >= 0 for a, c in R):
            out.add(x[k])
    return sorted(out)


def _boxed(rows, m, box=12.0):
    return rows + [([1.0 if j == i else 0.0 for j in range(m)], box)
                   for i in range(m)] + [
        ([-1.0 if j == i else 0.0 for j in range(m)], box)
        for i in range(m)]


def test_closure_kinks_contain_every_vertex_coordinate():
    """Difference rows only: at every level k >= 1 and random outer values,
    every vertex coordinate of the slice over (s_0..s_k) is an end of s_k's
    interval or a kink of ``_dbm_kinks`` (direct switches, and the
    alternating closure paths of ``_dbm_path_kinks`` for k >= 2) -- and the
    paths are needed: without them some vertex is missed."""
    rng = random.Random(31)
    n_vert = n_path_only = 0
    for trial in range(80):
        m = rng.choice((2, 3, 3, 4))
        rows = _boxed(_random_system(rng, m, rng.randint(m, 3 * m)), m)
        D, general, verdict = FI._dbm_closure(rows, m)
        assert not general
        if verdict == 'EMPTY':
            continue
        tabs = FI._dbm_level_tables(D, m)
        kc, ko = FI._dbm_kinks(D, m)
        pc = FI._dbm_path_kinks(D, m)
        for s in _feasible_points(rows, m, rng, n=3, box=12.0):
            for k in range(1, m):
                outer = tuple(s[k + 1:])
                L, U = FI._dbm_interval(tabs[k], outer)
                cands = ([L, U] + list(kc[k])
                         + [outer[oi] + off for (oi, off) in ko[k]])
                direct = set(cands) - set(pc[k][0]) - {
                    outer[oi] + off for (oi, off) in pc[k][1]}
                for v in _exact_vertex_coords(rows, m, k, list(outer)):
                    v = float(v)
                    near = lambda xs: any(                  # noqa: E731
                        abs(v - x) <= 1e-9 * (1 + abs(v)) for x in xs)
                    assert near(cands), (rows, k, outer, v)
                    n_vert += 1
                    n_path_only += not near(direct)
    assert n_vert > 300 and n_path_only > 0


def test_vertex_kinks_equal_the_exact_vertices():
    """Rows that are not difference rows: ``_VertexKinks.at`` returns the
    s_k-coordinates of the exact vertices (and nothing far from one)."""
    rng = random.Random(41)
    n = 0
    for trial in range(60):
        m = rng.choice((2, 3))
        rows = _boxed(_random_system(rng, m, rng.randint(m, 2 * m + 2),
                                     general=True), m)
        rows_t = tuple((float(c), tuple((j, float(a))
                                        for j, a in enumerate(a_int)
                                        if abs(float(a)) > 1e-15))
                       for (a_int, c) in rows)
        rows_t = tuple(r for r in rows_t if r[1])
        vk = FI._VertexKinks(rows_t, m)
        for s in _feasible_points(rows, m, rng, n=3, box=12.0):
            for k in range(1, m):
                outer = tuple(s[k + 1:])
                got = sorted(set(float(x) for x in vk.at(k, outer)))
                want = [float(x) for x in
                        _exact_vertex_coords(rows, m, k, list(outer))]
                for v in want:
                    assert any(abs(v - x) <= 1e-9 * (1 + abs(v))
                               for x in got), (rows, k, outer, v, got)
                for x in got:
                    assert any(abs(v - x) <= 1e-7 * (1 + abs(v))
                               for v in want), (rows, k, outer, x, want)
                n += len(want)
    assert n > 300


def test_fm_level_intervals_equal_the_exact_projection():
    """With rows that are not difference rows, the level intervals from the
    Fourier-Motzkin rows are the exact projection (``_fm_range``, LP), and
    'EMPTY' only for empty regions."""
    rng = random.Random(53)
    n_cmp = n_empty = 0
    for trial in range(80):
        m = rng.choice((2, 3, 4))
        rows = _random_system(rng, m, rng.randint(m + 1, 3 * m + 2),
                              general=True)
        D, general, verdict = FI._dbm_closure(rows, m)
        fm = FI._fm_level_rows(rows, m)
        pts = _feasible_points(rows, m, rng, n=6)
        if verdict == 'EMPTY' or fm == 'EMPTY':
            assert not pts
            n_empty += fm == 'EMPTY' and verdict is None   # FM only
            continue
        assert fm is not None
        tabs = FI._dbm_level_tables(D, m)
        for s in pts:
            for k in range(m):
                outer = tuple(s[k + 1:])
                L, U = FI._level_interval(tabs[k], fm[k], outer)
                exact = _fm_range(rows, m, k, list(outer))
                lp = _lp_range(rows, m, k, list(outer))
                for got, want, alt in ((L, exact[0], lp[0]),
                                       (U, exact[1], lp[1])):
                    if math.isinf(want):
                        assert got == want == alt, (rows, k, outer, L, U)
                    else:
                        assert abs(got - float(want)) <= 1e-13 * (
                            1 + abs(got)), (rows, k, outer, L, U, exact)
                        assert abs(got - alt) <= 1e-9 * (1 + abs(alt))
                assert L < s[k] < U
                n_cmp += 1
    assert n_cmp > 200 and n_empty > 0


def test_dbm_closure_is_exact_rational():
    # x0 − x1 < 0.1 and x1 − x2 < 0.2, x2 < 0.3: x0 < 0.6 exactly as the sum
    # of the three binary fractions, then rounded UP to a float.
    rows = [((-1.0, 1.0, 0.0), 0.1), ((0.0, -1.0, 1.0), 0.2),
            ((0.0, 0.0, -1.0), 0.3)]
    D, _g, verdict = FI._dbm_closure([(list(a), c) for a, c in rows], 3)
    assert verdict is None
    from fractions import Fraction
    exact = Fraction(0.1) + Fraction(0.2) + Fraction(0.3)
    assert D[0][3] == exact
    up = FI._frac_up(exact)
    assert Fraction(up) >= exact and math.nextafter(up, -math.inf) < exact
    assert FI._dbm_closure([([math.nan], 1.0)], 1)[2] == 'NONFINITE'


# ═══════════════════════════════════════════════════════════════════════
# E. open directions, the decay certificate and the magnitude bound
# ═══════════════════════════════════════════════════════════════════════

def test_open_direction_is_truncated_from_its_finite_end(hardened):
    """s1 < s0 < t, nothing below: ∫∫ 144 e^{-12(t-s0)} e^{-12(s0-s1)} = 1
    (u e^{-12u} over u = t − s1 > 0, times 144).  Truncated at 40/12 from
    the finite end: 1 − 41·e^{-40} ≈ 1 − 1.7e-16."""
    t = 0.7
    rows = [((-1.0, 0.0), (1.0,), 0.0), ((1.0, -1.0), (0.0,), 0.0)]
    fe = _fast_eval([_ems((144.0, -12.0)), _ems((1.0, -12.0))], rows, 2)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(fe, _resolved(rows, [t]), [t], 2)
    assert abs(v - 1.0) < 1e-12
    assert _counter('nquad_hardened_capped') == 1
    assert _counter('nquad_hardened_uncertified') == 0
    assert _counter('nquad_hardened_cap_span_max') == pytest.approx(
        FI.NQUAD_TAIL_K / 12.0)


def test_truncation_scales_with_the_row_coefficient(hardened):
    """An open side bounded by a row whose coefficient a is below 1: the
    integrand decays like exp(−κ·a·d), so the truncation is K/(κ·a).
    s0 < t, the edge Δt = (t − s0)/4 with one mode of rate 1:
    ∫_{-∞}^t e^{−(t−s0)/4} ds0 = 4.  (Truncated at K/κ = 40 the dropped
    tail was e^{−10} = 4.5e-5 relative, counted as certified.)"""
    t = 0.0
    rows = [((-0.25,), (0.25,), 0.0)]
    fe = _fast_eval([_ems((1.0, -1.0))], rows, 1)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(fe, _resolved(rows, [t]), [t], 1)
    assert _rel(v, 4.0) < 1e-12, v
    assert _counter('nquad_hardened_capped') == 1
    assert _counter('nquad_hardened_cap_span_max') == pytest.approx(
        FI.NQUAD_TAIL_K / 0.25)
    # m = 2: the edge −s0/4 + s1 > 0 bounds the open s0 (s1 in (-5, 0),
    # unit-rate edges −s0/4 + s1, −s1, s1 + 5): the integrand is
    # e^{s0/4 − s1 − 5}, ∫_{-∞}^{4 s1} ds0 gives 4 e^{s1}, and the total is
    # ∫_{-5}^0 4 e^{−5} ds1 = 20 e^{−5}.
    rows = [((-0.25, 1.0), (0.0,), 0.0), ((0.0, -1.0), (0.0,), 0.0),
            ((0.0, 1.0), (0.0,), 5.0)]
    fe = _fast_eval([_ems((1.0, -1.0))] * 3, rows, 2)
    v = FI._integrate_polytope(fe, _resolved(rows, [0.0]), [0.0], 2)
    ref = 20.0 * math.exp(-5.0)
    assert _rel(v, ref) < 1e-12, (v, ref)


# Mass far from the finite end: s0 < 0, s0 < s1, s1 < t = 60 (edges -s0,
# s1 - s0, t - s1; rates 1, 3/4, 7/10).  The s1-marginal peaks at the kink
# s1 = 0 (where s0's upper bound switches from s1 to 0), 60 from the finite
# end s1 = 60: above it, it decays like e^{-s1/20} over 0 < s1 < 60; below
# it, like e^{1.7·s1} (3.0 % of the integral).  A cut 40/κ_min = 57.1 from
# the finite end dropped s1 < 2.86: 16.6 % of the integral.  (Found in the
# public spike-reset model at τ = 60: half of four regions' values.)
# Exact: Mathematica 15 Integrate, 40 (35 e^3 − 34) / (119 e^45); the
# fractions also by Integrate.
FAR_ROWS = [((-1.0, 0.0), (0.0,), 0.0), ((-1.0, 1.0), (0.0,), 0.0),
            ((0.0, -1.0), (1.0,), 0.0)]
FAR_MODES = [_ems((1.0, -1.0)), _ems((1.0, -0.75)), _ems((1.0, -0.7))]
FAR_REF = 6.43699885971114806191781234557e-18
# The same region reflected s -> -s (open ABOVE: s0 > 0, s0 > s1, s1 > t =
# -60), the same modes and value: the strip of a moved cut lies above the
# old cut (``strip(cut, new)``).
FAR_ROWS_ABOVE = [((1.0, 0.0), (0.0,), 0.0), ((1.0, -1.0), (0.0,), 0.0),
                  ((0.0, 1.0), (-1.0,), 0.0)]
FAR_CASES = {'below': (FAR_ROWS, 60.0), 'above': (FAR_ROWS_ABOVE, -60.0)}


@pytest.mark.parametrize('side', sorted(FAR_CASES))
def test_truncation_reaches_mass_far_from_the_finite_end(side, hardened,
                                                         monkeypatch):
    rows, t = FAR_CASES[side]
    fe = _fast_eval(FAR_MODES, rows, 2)
    args = (fe, _resolved(rows, [t]), [t], 2)
    # the cut is measured from the farthest breakpoint (the kink s1 = 0)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args)
    assert _rel(v, FAR_REF) < 1e-12, v
    assert _counter('nquad_hardened_tail_widened') == 0
    # from the finite end: the tail bound (relative to the result) moves
    # the cut out.  The first pass's scale is floored at 1e-3 × the modes'
    # bound (1, against an integral of 6.4e-18), so a second pass runs, at
    # the moved cut ...
    monkeypatch.setattr(FI, '_NQUAD_TAIL_ANCHOR', False)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args)
    assert _rel(v, FAR_REF) < 1e-12, v
    assert _counter('nquad_hardened_tail_widened') == 1
    assert _counter('nquad_hardened_reruns') == 1
    # ... and with the first pass's epsabs from the sample of |integrand|
    # alone (no floor), no second pass: the strip between the two cuts
    # (which holds the kink: its breakpoints are kept) is integrated and
    # added -- below the old cut, or above it for the reflected region
    monkeypatch.setattr(FI, '_NQUAD_SCALE_FLOOR', 0.0)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args)
    assert _rel(v, FAR_REF) < 1e-12, v
    assert _counter('nquad_hardened_tail_widened') == 1
    assert _counter('nquad_hardened_reruns') == 0
    # neither: the cut at 40/κ_min from the finite end loses 16.6 %
    monkeypatch.setattr(FI, '_NQUAD_TAIL_REL', math.inf)
    v = FI._integrate_polytope(*args)
    assert 0.16 < _rel(v, FAR_REF) < 0.17, v


def test_a_moved_cut_adds_only_the_strip(hardened, monkeypatch):
    """When only the outermost cut has to move (``tail_k``), the strip
    between the two cuts is integrated and added; the first pass is kept.
    s < t, one edge with modes e^{-u} − e^{-(1+1e-7)u}: the integral
    (1e-7/(1 + 1e-7)) is ~1e-7 of the modes' bound S = 2, so the tail
    bound 2·e^{-40} exceeds 1e-11 × the result and the cut moves out.
    (The first pass's scale is the sampled |integrand| alone here: with its
    default floor, 1e-3 × the bound, a second pass would run, which takes
    the moved cut with it.)"""
    monkeypatch.setattr(FI, '_NQUAD_SCALE_FLOOR', 0.0)
    t = 0.3
    rows = [((-1.0,), (1.0,), 0.0)]
    fe = _fast_eval([_ems((1.0, -1.0), (-1.0, -(1.0 + 1e-7)))], rows, 1)
    calls = []
    real = FI._NquadModes.integrate_innermost

    def spy(self, L, U, outer, free):
        calls.append((L, U))
        return real(self, L, U, outer, free)
    monkeypatch.setattr(FI._NquadModes, 'integrate_innermost', spy)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(fe, _resolved(rows, [t]), [t], 1)
    assert _rel(v, 1e-7 / (1.0 + 1e-7)) < 1e-8, v
    assert _counter('nquad_hardened_tail_widened') == 1
    assert _counter('nquad_hardened_reruns') == 0
    assert len(calls) == 2, calls
    (l1, u1), (l2, u2) = calls
    assert u1 == pytest.approx(t, abs=1e-12) and l1 == t - 40.0
    assert u2 == l1 and l2 < l1                    # the strip only


# The decay certificate with rows that are not unit difference rows that
# are edges (``_GeneralDecayCertificate``).  D7: s0 < 0 the only edge (rate
# 1), rows 2·s1 < s0 < s1/4 (not edges): along s1 -> −∞ the integrand
# e^{s0} decays like e^{s1/4}, whatever the rows' scaling.  Before, the
# certificate took the rows' smallest coefficient: written with the
# integer row (−4, 1) the region was cut 40 from the end (5.2e-5 low, D7c
# with (−8, 1): 7.2e-3), written as (−1, 1/4) 160 (exact).  D8 (m = 3): s1
# in (2·s2, s2/2), s0 in (2·s1, s1/2): the decay rate 1/4 along s2 comes
# from two rows together (each has the normalised coefficient 1/2).  N7b:
# the slices s0, s1 in (1025·s2, s2) of the edge −s2 (modes 1 and
# −8(1 − 2^-18) at rates 1 and 2: the integral cancels to 1/2^17 of its
# parts) are 1024·|s2| wide; a tail bound for chain-like slices did not
# move the cut (9.2e-10 low).  Exact: Mathematica 15 Integrate (Boole
# form): 7/2, 15/2, 45/8, 8.
_N7C, _N7D = 1025.0, 2.0 ** -18
GEN_CASES = {
    'D7': (2, [((-1.0, 0.0), (), 0.0), ((1.0, -2.0), (), 0.0),
               ((-4.0, 1.0), (), 0.0)], [_ems((1, -1))], 3.5, 160.0),
    'D7b': (2, [((-1.0, 0.0), (), 0.0), ((1.0, -2.0), (), 0.0),
                ((-1.0, 0.25), (), 0.0)], [_ems((1, -1))], 3.5, 160.0),
    'D7c': (2, [((-1.0, 0.0), (), 0.0), ((1.0, -2.0), (), 0.0),
                ((-8.0, 1.0), (), 0.0)], [_ems((1, -1))], 7.5, 320.0),
    'D8': (3, [((-1.0, 0.0, 0.0), (), 0.0), ((0.0, 1.0, -2.0), (), 0.0),
               ((0.0, -2.0, 1.0), (), 0.0), ((1.0, -2.0, 0.0), (), 0.0),
               ((-2.0, 1.0, 0.0), (), 0.0)], [_ems((1, -1))], 45 / 8, 160.0),
    'N7b': (3, [((0.0, 0.0, -1.0), (), 0.0), ((0.0, 1.0, -_N7C), (), 0.0),
                ((0.0, -1.0, 1.0), (), 0.0), ((1.0, 0.0, -_N7C), (), 0.0),
                ((-1.0, 0.0, 1.0), (), 0.0)],
            [_ems((1, -1), (-8 * (1 - _N7D), -2))], 8.0, None),
}


@pytest.mark.parametrize('case', sorted(GEN_CASES))
def test_decay_certificate_of_general_rows(case, hardened):
    m, rows, modes, ref, span = GEN_CASES[case]
    fe = _fast_eval(modes, rows, m)
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(fe, _resolved(rows, []), [], m)
    tol = 1e-10 if case == 'N7b' else 1e-12   # N7b cancels to 2^-17
    assert _rel(v, ref) < tol, v
    assert _counter('nquad_hardened_capped') == 1
    assert _counter('nquad_hardened_uncertified') == 0
    if span is not None:
        # the same region written with other row scalings: the same cut
        assert _counter('nquad_hardened_cap_span_max') == pytest.approx(
            span, rel=1e-6)
    else:
        # the slices' volume moves the cut out (a strip is added)
        assert _counter('nquad_hardened_tail_widened') == 1


# The inner levels' cuts with rows that are not unit difference rows that
# are edges (``tail_in``).  G2s: s3 in (-1, 0) (edge -s3, mode 2^22·e^{-Δ}),
# the inner open level s2 < s3 (edge G(s3 − s2), G(x) = e^{-x} − 8(1 −
# d)·e^{-2x} with d = 2^-22, so ∫ x²·G = 2d: the modes cancel to 2^-21 of
# their parts), and s1, s0 in (2·s2 − s3, s2) through rows that are not
# edges, so the slices over (s0, s1) have the area (s3 − s2)^2.  Cut 40
# beyond its anchor, the inner level dropped (Σ_{i<=2} 40^i/i!)·e^{-40}/(2d)
# = 1.5e-8 of the integral (unchecked before).  Exact (Mathematica 15,
# Integrate of the Boole form): 2(1 − e^{-1}).
G2S_D = 2.0 ** -22
G2S_EDGES = [((0.0, 0.0, -1.0, 1.0), (0.0,), 0.0),
             ((0.0, 0.0, 0.0, -1.0), (0.0,), 0.0)]
G2S_MODES = [_ems((1.0, -1.0), (-8.0 * (1.0 - G2S_D), -2.0)),
             _ems((2.0 ** 22, -1.0))]
G2S_ROWS = [((0.0, 0.0, 0.0, -1.0), (0.0,), 0.0),
            ((0.0, 0.0, 0.0, 1.0), (0.0,), 1.0),
            ((0.0, 0.0, -1.0, 1.0), (0.0,), 0.0),
            ((0.0, -1.0, 1.0, 0.0), (0.0,), 0.0),
            ((0.0, 1.0, -2.0, 1.0), (0.0,), 0.0),
            ((-1.0, 0.0, 1.0, 0.0), (0.0,), 0.0),
            ((1.0, 0.0, -2.0, 1.0), (0.0,), 0.0)]
G2S_REF = 2.0 * (1.0 - math.exp(-1.0))


def test_inner_cuts_are_checked_with_general_rows(hardened, monkeypatch):
    fe = FI._build_fast_subset_evaluator_from_modes(1.0 + 0j, G2S_MODES,
                                                    G2S_EDGES, 4)
    args = (fe, _resolved(G2S_ROWS, [0.0]), [0.0], 4)
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(*args)
    # the rest is the rounding of the cancelling inner values (~1e-10,
    # flagged: 'did not reach its tolerance')
    assert _rel(v, G2S_REF) < 2e-9, v
    assert _counter('nquad_hardened_general_rows') == 1
    assert _counter('nquad_hardened_tail_widened') == 1
    assert _counter('nquad_hardened_cap_span_max') > FI.NQUAD_TAIL_K + 10
    # without the check (the outermost level is bounded, so only the inner
    # cuts are concerned): 1.5e-8 low
    monkeypatch.setattr(FI, '_NQUAD_TAIL_REL', math.inf)
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(*args)
    assert 1.2e-8 < (G2S_REF - v.real) / G2S_REF < 1.8e-8, v
    assert _counter('nquad_hardened_tail_widened') == 0


def test_unchecked_inner_cuts_are_counted(hardened, monkeypatch):
    """G2sb, the m = 3 form of G2s (s2 in (-1, 0), s1 < s2 with G(s2 −
    s1) = e^{-x} − 4(1 − d)e^{-2x}, d = 2^-22, s0 in (2·s1 − s2, s1); exact
    1 − e^{-1}, Mathematica 15): checked, the inner cut moves; with the cuts
    measured from the finite ends (``_NQUAD_TAIL_ANCHOR`` off) the bound
    does not hold, so the cut is left unchecked and counted."""
    d = 2.0 ** -22
    edges = [((0.0, -1.0, 1.0), (0.0,), 0.0), ((0.0, 0.0, -1.0), (0.0,), 0.0)]
    modes = [_ems((1.0, -1.0), (-4.0 * (1.0 - d), -2.0)),
             _ems((2.0 ** 22, -1.0))]
    rows = [((0.0, 0.0, -1.0), (0.0,), 0.0), ((0.0, 0.0, 1.0), (0.0,), 1.0),
            ((0.0, -1.0, 1.0), (0.0,), 0.0), ((-1.0, 1.0, 0.0), (0.0,), 0.0),
            ((1.0, -2.0, 1.0), (0.0,), 0.0)]
    fe = FI._build_fast_subset_evaluator_from_modes(1.0 + 0j, modes, edges, 3)
    ref = 1.0 - math.exp(-1.0)
    for anchored in (True, False):
        monkeypatch.setattr(FI, '_NQUAD_TAIL_ANCHOR', anchored)
        FI._reset_runtime_counters()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
            v = FI._integrate_polytope(fe, _resolved(rows, [0.0]), [0.0], 3)
        assert _counter('nquad_hardened_tail_widened') == int(anchored)
        assert _counter('nquad_hardened_inner_unchecked') == int(not anchored)
        if anchored:
            assert _rel(v, ref) < 3e-10, v          # unchecked: 7.0e-10


@pytest.mark.parametrize('scale', (1.0, 3.0))
def test_region_tail_bounds_the_mass_beyond_the_level(scale):
    """``_GeneralDecayCertificate.region_tail(K)`` bounds S·∫ e^{−φ} over
    {φ >= φ_min + K} on the region.  s0 < s1 < 0 (rows written as a
    general region), edges −s1 and s1 − s0 at rate 1: φ = −s0, φ_min = 0,
    the mass is S·(K + 1)·e^{−K}.  The widths of {φ <= τ} are τ in both
    variables (slope 1), so the bound is S·e^{−K}·Σ_j p_j·j! with Σ_j p_j
    y^j = (K + y)²: S·(K² + 2K + 2)·e^{−K}.  It must EQUAL that up to its
    rounding margins (relative 1e-6 on the widths, 1e-7 on φ_min; measured
    4e-6 to 7e-6 above it), with the widths measured at the first K and
    extrapolated to larger ones (first order below), or measured again for
    a smaller K (second order).  Without the j! weights the bound would be
    S·(K + 1)²·e^{−K} -- still above the mass, so only the two-sided check
    sees that (5.5e-4 below the closed form at K = 41.5, 20 % at K = 1)."""
    rows_t = ((0.0, ((1, -1.0),)), (0.0, ((0, -1.0), (1, 1.0))))
    edges = [(((1, -1.0),), 0.0, 1.0), (((0, -1.0), (1, 1.0)), 0.0, 1.0)]
    for order in ((1.0, 10.0, 40.0, 41.5), (41.5, 40.0, 10.0, 1.0)):
        cert = FI._GeneralDecayCertificate(rows_t, 2, (scale, edges), None,
                                           True)
        assert cert.ok, cert.why
        for K in order:
            T = cert.region_tail(K)
            exact = scale * (K + 1.0) * math.exp(-K)
            closed = scale * (K * K + 2 * K + 2) * math.exp(-K)
            assert exact <= T, (K, T, exact)
            assert 0.0 <= T / closed - 1.0 <= 2e-5, (order, K, T / closed)


def test_unit_edge_rows_keep_the_closed_form_outer_bound(hardened, monkeypatch):
    """Unit difference rows that are edges (every Phase J region measured)
    keep the κ_min certificate and the closed-form bound for the outermost
    cut; the linear-programming certificate is built only on first need, to
    check the inner levels' cuts, and those cuts are then checked (not left
    in ``nquad_hardened_inner_unchecked``)."""
    calls = []
    real = FI._GeneralDecayCertificate

    def counted(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(FI, '_GeneralDecayCertificate', counted)
    FI._reset_runtime_counters()
    fe = _fast_eval(FAR_MODES, FAR_ROWS, 2)
    v = FI._integrate_polytope(fe, _resolved(FAR_ROWS, [60.0]), [60.0], 2)
    assert _rel(v, FAR_REF) < 1e-12, v
    assert len(calls) <= 1                     # only the inner-cut check
    assert FI._RUNTIME_COUNTERS['nquad_hardened_inner_unchecked'] == 0
    rows_t = tuple((c, tuple((j, a) for j, a in enumerate(av) if a))
                   for (av, c) in _resolved(FAR_ROWS, [60.0]))
    er = FI._edge_rows_at(fe._nquad_modes, [60.0])
    assert FI._unit_edge_rows(rows_t, er)
    # a looser parallel copy of an edge's row changes nothing (only the
    # tightest row per direction bounds the region)
    assert FI._unit_edge_rows(rows_t + ((5.0, ((0, -1.0),)),), er)
    # a row that is not an edge, or a coefficient other than ±1: not unit
    assert not FI._unit_edge_rows(rows_t + ((5.0, ((0, 1.0), (1, -1.0))),),
                                  er)
    assert not FI._unit_edge_rows(
        rows_t[:1] + ((0.0, ((0, -0.5), (1, 0.5))),) + rows_t[2:], er)


# Slices of the outermost level unbounded in an inner variable, with a row
# that is not an edge (s0 + s1/2 < 5, redundant).  U: s0 < s1 < 0, edges
# −s1 and s1 − s0 (rate 1): the integrand e^{s0} decays as s0 -> −∞, so the
# box fibration bounds the tail; exact 1.  V: s0 in (s1 − 3, s1), s1 < s2 <
# 0, edges −s2, s2 − s1, s1 − s0 (rate 1) and s0 − s1 + 3 (rate 2), and the
# redundant row 10 − s0 − s2/2: the integrand e^{2 s1 − s0 − 6} grows as
# s0 -> −∞ inside the box fibration, so no certificate: the legacy cap,
# counted and warned about.  Exact (Mathematica 15): 1 and (e^3 − 1)/e^6.
def test_unbounded_slices_with_general_rows(hardened, fresh_warning_state):
    rows = [((0.0, -1.0), (), 0.0), ((-1.0, 1.0), (), 0.0),
            ((-1.0, -0.5), (), 5.0)]
    fe = _fast_eval([_ems((1, -1))] * 2, rows, 2)
    FI._reset_runtime_counters()
    msgs = _warnings_of([lambda: FI._integrate_polytope(
        fe, _resolved(rows, []), [], 2, diag_meta=_meta(None, model=(
            'model', 'u')))])
    assert len(msgs) == 1 and 'decay certificate' not in msgs[0], msgs
    assert _counter('nquad_hardened_uncertified') == 0
    assert _counter('nquad_hardened_capped') == 1
    v = FI._integrate_polytope(fe, _resolved(rows, []), [], 2)
    assert _rel(v, 1.0) < 1e-12, v
    rows = [((0.0, 0.0, -1.0), (), 0.0), ((0.0, -1.0, 1.0), (), 0.0),
            ((-1.0, 1.0, 0.0), (), 0.0), ((1.0, -1.0, 0.0), (), 3.0),
            ((-1.0, 0.0, -0.5), (), 10.0)]
    fe = _fast_eval([_ems((1, -1))] * 3 + [_ems((1, -2))], rows, 3)
    FI._reset_runtime_counters()
    msgs = _warnings_of([lambda: FI._integrate_polytope(
        fe, _resolved(rows, []), [], 3, diag_meta=_meta(None, model=(
            'model', 'v')))])
    assert _counter('nquad_hardened_uncertified') == 1
    v = FI._integrate_polytope(fe, _resolved(rows, []), [], 3)
    ref = (math.exp(3.0) - 1.0) / math.exp(6.0)
    assert _rel(v, ref) < 1e-10, v
    assert len(msgs) == 1 and ('1 with an open direction without a decay '
                               'certificate') in msgs[0], msgs


def test_an_edge_that_can_be_negative_has_no_certificate(hardened):
    """The decay bound needs every edge's Δt >= 0 on the region.  s0 < 0
    (an edge), and the edge s0 + 5 that is not a row: below s0 = −5 its
    mode e^{-3Δt} grows, and the integral diverges -- whatever the slowest
    rates say.  No certificate: the legacy cap, counted."""
    rows = [((-1.0,), (), 0.0), ((1.0,), (), 5.0)]
    fe = _fast_eval([_ems((1, -1)), _ems((1, -0.5), (1, -3))], rows, 1)
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        FI._integrate_polytope(fe, _resolved(rows[:1], []), [], 1)
    assert _counter('nquad_hardened_uncertified') == 1
    assert _counter('nquad_hardened_capped') == 0


def test_wide_open_interval_with_fast_and_slow_modes(hardened):
    """One variable, s < t, g(u) = e^{-20u} + 1e-3 e^{-0.01u}: the slow mode
    sets the truncation (40/0.01 = 4000 wide), the fast one a peak of width
    1/20 at the finite end.  ∫_0^{4000} g = 1/20 + 0.1·(1 − e^{-40})."""
    t = 0.3
    rows = [((-1.0,), (1.0,), 0.0)]
    fe = _fast_eval([_ems((1.0, -20.0), (1e-3, -0.01))], rows, 1)
    v = FI._integrate_polytope(fe, _resolved(rows, [t]), [t], 1)
    ref = 1 / 20 + 0.1 * (1 - math.exp(-40))
    assert _rel(v, ref) < 1e-12


def test_uncertified_open_direction_uses_the_legacy_cap(hardened):
    """No decay certificate (a mode with Re λ = 0, or no mode data): the
    open side is capped at the legacy distance 200 and counted."""
    t = 0.0
    rows = [((-1.0,), (1.0,), 0.0)]
    fe = _fast_eval([_ems((1.0, 0.0))], rows, 1)     # g(u) = 1
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(fe, _resolved(rows, [t]), [t], 1)
    assert v == pytest.approx(FI.NQUAD_UNCERTIFIED_CAP, rel=1e-12)
    assert _counter('nquad_hardened_uncertified') == 1
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(lambda s, tt: math.exp(-(tt - s)),
                               _resolved(rows, [t]), [t], 1)
    assert v == pytest.approx(1.0, rel=1e-8)
    assert _counter('nquad_hardened_no_modes') == 1
    assert _counter('nquad_hardened_uncertified') == 1


def test_no_mode_data_tolerance_scales_with_the_integrand(hardened):
    """Without mode data the absolute tolerance comes from a sample of the
    integrand on the region, so a small integrand is as accurate as a large
    one.  (With scipy's absolute 1.49e-8 the value at prefactor 1e-6 was
    off by 2.8e-4 relative, at 1e-9 by 8.3e-3.)  s0 < s1 < t, edges
    G0(u) = e^{-12u} + e^{-u/4}/2, G1(v) = e^{-12v} + e^{-v/2}: the
    integral is (1/12 + 2)^2 = 625/144 times the prefactor."""
    rows = [((-1.0, 1.0), (0.0,), 0.0), ((0.0, -1.0), (1.0,), 0.0)]
    modes = [_ems((1, -12), (0.5, -0.25)), _ems((1, -12), (1, -0.5))]
    for pref in (1.0, 1e-6, 1e-9):
        fe = FI._build_fast_subset_evaluator_from_modes(
            pref + 0j, list(modes), rows, 2)
        FI._reset_runtime_counters()
        v = FI._integrate_polytope(lambda *a: fe(*a),
                                   _resolved(rows, [3.0]), [3.0], 2)
        assert _counter('nquad_hardened_no_modes') == 1
        assert _rel(v, pref * 625 / 144) < 1e-10, (pref, v)


@pytest.mark.parametrize('kind', ('product', 'sum'))
def test_magnitude_bound_is_rigorous(kind):
    """|integrand| <= bound at random points of the region inside random
    boxes, for random modes (incl. Re λ > 0 and oscillating ones)."""
    rng = random.Random(11)
    n_checked = 0
    for trial in range(40):
        m = rng.choice((1, 2, 3))
        rows = [(tuple(1.0 if j == i + 1 else -1.0 if j == i else 0.0
                       for j in range(m)) if i + 1 < m else
                 tuple(-1.0 if j == i else 0.0 for j in range(m)),
                 (1.0 if i + 1 >= m else 0.0,), 0.0) for i in range(m)]
        modes = [_ems(*[(complex(rng.uniform(-2, 2), rng.uniform(-2, 2)),
                         complex(rng.uniform(-3, 0.5), rng.uniform(-5, 5)))
                        for _ in range(rng.randint(1, 3))])
                 for _ in range(m)]
        fe = _fast_eval(modes, rows, m)
        info = (fe._nquad_modes if kind == 'product' else
                FI._NquadModes.from_pole_tuples(rows, _pole_tuples(modes)))
        t = rng.uniform(-1, 1)
        lo = [t - rng.uniform(1, 6) for _ in range(m)]
        hi = [t + rng.uniform(-0.5, 1) for _ in range(m)]
        B = info.bound(lo, hi, [t])
        for _ in range(200):
            s = [rng.uniform(lo[j], hi[j]) for j in range(m)]
            if not _rows_hold(rows, [t], s):
                continue
            assert abs(fe(*s, t)) <= B * (1 + 1e-12), (s, B)
            n_checked += 1
    assert n_checked > 300


# ═══════════════════════════════════════════════════════════════════════
# E2. the absolute tolerance follows the result; oscillation and the
#     subinterval limit; the cost of the closed-form innermost level
# ═══════════════════════════════════════════════════════════════════════
# Close poles: -D < s0 < s1 < 0, the slow edge s1 − s0 (rate 1/8) and two
# close-pole edges -s1 and s0 + D, each 2048·(e^{-2Δ} − e^{-(2+1/2048)Δ})
# (residues ±2048, so the modes' bound is ~1.7e7 while the integrand stays
# below ~0.034 and the integral is 3e-7).  With epsabs = 1e-13 × bound
# (1.7e-6 > |I|) QUADPACK stopped at its first estimates: 3.1e-5 relative,
# no flag.  Mathematica 15: Integrate of the expanded exponentials over the
# triangle, term by term, N[.., 30]; the same value from the closed-form
# triangle integral of each exponential (agree to 58 digits).
_EPS_CP = 1.0 / 2048
CP_ROWS = [((-1.0, 1.0), (0.0,), 0.0), ((0.0, -1.0), (0.0,), 0.0),
           ((1.0, 0.0), (0.0,), 100.0)]
CP_MODES = [_ems((1.0, -0.125)),
            _ems((1 / _EPS_CP, -2.0), (-1 / _EPS_CP, -(2.0 + _EPS_CP))),
            _ems((1 / _EPS_CP, -2.0), (-1 / _EPS_CP, -(2.0 + _EPS_CP)))]
CP_REF = 3.01361467730702528959793276470810703e-7


@pytest.mark.parametrize('grouped', (False, True), ids=('product', 'sum'))
def test_close_poles_tolerance_follows_the_result(grouped, hardened,
                                                  monkeypatch):
    # The grouped form pre-expands the products of modes into pole tuples
    # with coefficients ±2048² that cancel: its integrand (and so any
    # quadrature of it) is accurate to ~2e-10 relative here, the
    # per-diagram product form to ~3e-13.
    tol = 1e-9 if grouped else 1e-11
    fe = _fast_eval(CP_MODES, CP_ROWS, 2)
    info = (FI._NquadModes.from_pole_tuples(CP_ROWS, _pole_tuples(CP_MODES))
            if grouped else None)
    f = (lambda *a: fe(*a)) if grouped else fe
    args = (f, _resolved(CP_ROWS, [0.0]), [0.0], 2)
    assert (info or fe._nquad_modes).bound([-100.0] * 2, [0.0] * 2,
                                           [0.0]) > 1e7
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args, mode_info=info)
    assert _rel(v, CP_REF) < tol, v
    assert _counter('nquad_hardened_quad_flags') == 0
    # the bound alone (no sample): the second pass with epsabs from the
    # first result recovers the accuracy
    monkeypatch.setattr(FI, '_NQUAD_PRESAMPLES', 0)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args, mode_info=info)
    assert _rel(v, CP_REF) < tol, v
    assert _counter('nquad_hardened_reruns') == 1
    # ... which the bound-based first pass alone does not reach
    monkeypatch.setattr(FI, '_NQUAD_RERUN_MAX', 0)
    v = FI._integrate_polytope(*args, mode_info=info)
    assert _rel(v, CP_REF) > 1e-6, v


# Oscillating modes over an open past: s0 < s1 < t = 3, edges s1 − s0 with
# modes e^{(-1/4 ± 100i)u} and t − s1 with e^{(-1/2 ± 7i)v} + 0.3 e^{-12v}:
# the region is a product, so the integral is
# [2 Re 1/(1/4 − 100i)]·[2 Re 1/(1/2 − 7i) + 1/40] = 357/157600985
# (Mathematica 15 Integrate agrees).  Before: 1.9e-8 relative, no flag.
OSC_ROWS = [((-1.0, 1.0), (0.0,), 0.0), ((0.0, -1.0), (1.0,), 0.0)]
OSC_MODES = [_ems((1, -0.25 + 100j), (1, -0.25 - 100j)),
             _ems((1, -0.5 + 7j), (1, -0.5 - 7j), (0.3, -12))]
OSC_REF = 357.0 / 157600985.0


@pytest.mark.parametrize('grouped', (False, True), ids=('product', 'sum'))
def test_oscillating_open_region_matches_the_exact_value(grouped, hardened):
    fe = _fast_eval(OSC_MODES, OSC_ROWS, 2)
    info = (FI._NquadModes.from_pole_tuples(OSC_ROWS,
                                            _pole_tuples(OSC_MODES))
            if grouped else None)
    f = (lambda *a: fe(*a)) if grouped else fe
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(f, _resolved(OSC_ROWS, [3.0]), [3.0], 2,
                               mode_info=info)
    assert _rel(v, OSC_REF) < 1e-10, v
    assert _counter('nquad_hardened_quad_flags') == 0
    assert _counter('nquad_hardened_capped') == 1


# Oscillation over a bounded region: -60 < s0 < s1 < t = 3, the same edges
# plus s0 + 60 (rate 1/8).  The outer integrand oscillates ~2000 times over
# its interval, beyond the base limit of 200 subintervals: before, quad
# stopped at the limit (ier != 0, only a counter) and the value was 20 %
# off.  Mathematica 15 (term-by-term Integrate over the triangle; the same
# from the closed-form triangle integrals), N[.., 30].
OSCB_ROWS = OSC_ROWS + [((1.0, 0.0), (0.0,), 60.0)]
OSCB_MODES = OSC_MODES + [_ems((1, -0.125))]
OSCB_REF = 4.13992667045096732117375143423914901699916e-10


def test_oscillation_sets_the_subinterval_limit(hardened, monkeypatch):
    monkeypatch.setattr(FI, '_NQUAD_WARNED', set())
    fe = _fast_eval(OSCB_MODES, OSCB_ROWS, 2)
    assert fe._nquad_modes.osc_bound() >= 200.0
    args = (fe, _resolved(OSCB_ROWS, [3.0]), [3.0], 2)
    # the starting limit from the oscillation count
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args)
    assert _rel(v, OSCB_REF) < 1e-10, v
    assert _counter('nquad_hardened_quad_flags') == 0
    assert _counter('nquad_hardened_quad_retries') == 0
    # without it: the call that reaches the base limit is repeated with
    # NQUAD_LIMIT_MAX
    monkeypatch.setattr(FI, '_NQUAD_OSC_MIN', math.inf)
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(*args)
    assert _rel(v, OSCB_REF) < 1e-10, v
    assert _counter('nquad_hardened_quad_retries') > 0
    assert _counter('nquad_hardened_quad_flags') == 0
    # without both: the flag persists, is counted and warned about
    monkeypatch.setattr(FI, 'NQUAD_LIMIT_MAX', FI.NQUAD_LIMIT)
    FI._reset_runtime_counters()
    msgs = _warnings_of([lambda: FI._integrate_polytope(*args)])
    assert _counter('nquad_hardened_quad_flags') > 0
    assert any('did not reach its tolerance' in x for x in msgs), msgs


# The first pass's tolerance scale.  G4: s2 < 0 (edge −s2, rate 1/4), s1 <
# s2 (edge s2 − s1: rates 1 and 1 ± 2i), s0 in (s1, (s1 + s2)/2 + 2)
# (edges s0 − s1: e^{-u} − e^{-3u}, and (s1 + s2)/2 + 2 − s0: rate 1/2; a
# row that is not a difference row).  The 64 samples of |integrand|, spread
# over the truncated sides, gave 1.7e-12 (the integrand is ~0.4 near s2 =
# 0, the bound 4), so epsabs was 1.7e-25; with the pole tuples (grouped) the
# imaginary parts of the level values carry rounding noise above that, and
# the calls that chased it ran to 12800 subintervals: no result in 25 min
# (0.3 s per diagram).  Two changes each fix it (measured): the
# scale is floored at 1e-3 × the bound (first-pass epsabs 4e-16, pinned
# here), and a part that is only noise is integrated second with epsabs
# raised to 1e-10 × the other part (``test_a_noise_part_does_not_drive_the
# _subdivision``; with it, floors of 1e-6 and 0 take 0.3 s too).  Exact
# (Mathematica 15, iterated Integrate): 1.43354957602536370237161216998800...
# G4c, a Phase-J-shaped variant (±1 ordering rows that are edges): the edge
# −s2 at rate 1/16 (the truncated sides 640 long), s0 in (s1, s1 + 3)
# (edges s0 − s1 and s1 + 3 − s0); before, its grouped evaluation also ran
# past five million evaluations (35 s); now 0.4 s.  Exact (Mathematica
# 15): 192·(1 − 5e^6 + 4e^{15/2})/(25e^9).
G4_EDGES = [((0.0, 0.0, -1.0), (0.0,), 0.0), ((0.0, -1.0, 1.0), (0.0,), 0.0),
            ((1.0, -1.0, 0.0), (0.0,), 0.0), ((-1.0, 0.5, 0.5), (0.0,), 2.0)]
G4_MODES = [_ems((1, -0.25)), _ems((1, -1), (0.5, -1 + 2j), (0.5, -1 - 2j)),
            _ems((1, -1), (-1, -3)), _ems((1, -0.5))]
G4_REF = 1.4335495760253637023716121699880
G4C_EDGES = G4_EDGES[:3] + [((-1.0, 1.0, 0.0), (0.0,), 3.0)]
G4C_MODES = [_ems((1, -1.0 / 16))] + G4_MODES[1:]
G4C_REF = 4.9436828817291746333637691506929
G4_CASES = {'general': (G4_EDGES, G4_MODES, G4_REF),
            'unit': (G4C_EDGES, G4C_MODES, G4C_REF)}


@pytest.mark.parametrize('grouped', (False, True), ids=('product', 'sum'))
@pytest.mark.parametrize('case', sorted(G4_CASES))
def test_tolerance_scale_is_floored_by_the_bound(case, grouped, hardened,
                                                 monkeypatch):
    edges, modes, ref = G4_CASES[case]
    import scipy.integrate as si
    real, n, eps = si.quad, [0], []

    def budgeted(*a, **kw):                 # fail fast instead of hanging
        eps.append(kw.get('epsabs'))
        r = real(*a, **kw)
        n[0] += r[2]['neval'] if len(r) > 2 and isinstance(r[2], dict) else 0
        if n[0] > 1_000_000:                # measured: 50 000 and 90 000
            raise AssertionError('quadrature budget exceeded')
        return r
    monkeypatch.setattr(si, 'quad', budgeted)
    fe = _fast_eval(modes, edges, 3)
    f, info = fe, None
    if grouped:                       # the merged pole tuples' mode data
        f = lambda *a: fe(*a)         # noqa: E731
        info = FI._NquadModes.from_pole_tuples(edges, _pole_tuples(modes))
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(f, _resolved(edges, [0.0]), [0.0], 3,
                                   mode_info=info)
    assert _rel(v, ref) < 1e-12, v
    assert _counter('nquad_hardened_quad_flags') == 0
    assert _counter('nquad_hardened_quad_retries') == 0
    # the first pass's epsabs: 1e-13 × the floor 1e-3 × the bound 4 (the
    # sample alone: 1.7e-25 for G4, 1.0e-27 for G4c)
    assert abs(eps[0] - 4e-16) <= 1e-9 * 4e-16, eps[0]


# A part of a level's complex value that is only the rounding noise of the
# other is judged against the complex value (max(epsabs, epsrel·|re + i·
# im|)), not refined to an epsabs below its noise.  Region: 0 < s1 < 1/2,
# 10 < s0 < 21/2 (FLAG_ROWS), a real or an imaginary constant without mode
# data; every inner quad call of the zero part is faked to stop at its
# subinterval limit with an error estimate of 1e-12 (5e-11 is the complex
# value's tolerance), or 1e-9 (above it: repeated, then flagged).
def _faking_zero_part(abserr):
    import scipy.integrate as si
    real = si.quad

    def fake(func, a, b, *args, **kw):
        r = real(func, a, b, *args, **kw)
        if a < 5.0 or func(0.5 * (a + b)) != 0.0:     # outer, or not zero
            return r
        lim = kw.get('limit', 50)
        return (r[0], abserr, {'last': lim, 'neval': 21 * lim},
                f'The maximum number of subdivisions ({lim}) has been '
                'achieved (fake)')
    return fake


@pytest.mark.parametrize('const', (1.0, 1j), ids=('real', 'imaginary'))
def test_a_noise_part_is_judged_against_the_complex_value(const, hardened,
                                                          fresh_warning_state,
                                                          monkeypatch):
    import scipy.integrate as si
    rows = _resolved(FLAG_ROWS, [])
    monkeypatch.setattr(si, 'quad', _faking_zero_part(1e-12))
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(lambda *a: const, rows, [], 2)
    assert abs(v - 0.25 * const) < 1e-12, v
    assert _counter('nquad_hardened_quad_retries') == 0
    assert _counter('nquad_hardened_quad_flags') == 0
    monkeypatch.setattr(si, 'quad', _faking_zero_part(1e-9))
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(lambda *a: const, rows, [], 2)
    assert abs(v - 0.25 * const) < 1e-12, v
    assert _counter('nquad_hardened_quad_retries') > 0
    assert _counter('nquad_hardened_quad_flags') > 0


def _faking_first_part(abserr):
    """Like ``_faking_zero_part``, for the nonzero part: the part that is
    larger at the first node, so the one integrated first."""
    import scipy.integrate as si
    real = si.quad

    def fake(func, a, b, *args, **kw):
        r = real(func, a, b, *args, **kw)
        if a < 5.0 or func(0.5 * (a + b)) == 0.0:     # outer, or zero
            return r
        lim = kw.get('limit', 50)
        return (r[0], abserr, {'last': lim, 'neval': 21 * lim},
                f'The maximum number of subdivisions ({lim}) has been '
                'achieved (fake)')
    return fake


@pytest.mark.parametrize('const', (1.0, 1j), ids=('real', 'imaginary'))
def test_the_first_part_is_judged_once_the_other_is_known(const, hardened,
                                                          fresh_warning_state,
                                                          monkeypatch):
    """The part integrated first ends with a QUADPACK message: its judgement
    waits until the other part is known, then it is accepted against the
    complex value without a repeat (error estimate 1e-12; the tolerance is
    5e-11) -- or, with 1e-9, repeated with NQUAD_LIMIT_MAX and flagged."""
    import scipy.integrate as si
    rows = _resolved(FLAG_ROWS, [])
    monkeypatch.setattr(si, 'quad', _faking_first_part(1e-12))
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(lambda *a: const, rows, [], 2)
    assert abs(v - 0.25 * const) < 1e-12, v
    assert _counter('nquad_hardened_quad_retries') == 0
    assert _counter('nquad_hardened_quad_flags') == 0
    monkeypatch.setattr(si, 'quad', _faking_first_part(1e-9))
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(lambda *a: const, rows, [], 2)
    assert abs(v - 0.25 * const) < 1e-12, v
    assert _counter('nquad_hardened_quad_retries') > 0
    assert _counter('nquad_hardened_quad_flags') > 0


@pytest.mark.parametrize('pref', (1.0, 1j), ids=('real', 'imaginary'))
def test_a_noise_part_does_not_drive_the_subdivision(pref, hardened,
                                                     monkeypatch):
    """G4 grouped (the merged pole tuples' mode data), times a prefactor 1
    or i, with the scale floor off: epsabs 1.7e-25 from the sample, below
    the rounding noise of the part of each level's value that is zero
    (the imaginary part for 1, the real part for i).  That part is
    integrated second, with epsabs raised to 1e-10 × |the other part|
    (the part that is larger at the first node goes first).  Measured:
    ~52 000 evaluations, 0.3 s; with the judgement against the
    complex value alone, and for i also with the raised epsabs but the
    real part always first, more than three million."""
    import scipy.integrate as si
    real, n = si.quad, [0]

    def budgeted(*a, **kw):                 # fail fast instead of hanging
        r = real(*a, **kw)
        n[0] += r[2]['neval'] if len(r) > 2 and isinstance(r[2], dict) else 0
        if n[0] > 1_000_000:
            raise AssertionError('quadrature budget exceeded')
        return r
    monkeypatch.setattr(si, 'quad', budgeted)
    monkeypatch.setattr(FI, '_NQUAD_SCALE_FLOOR', 0.0)
    modes = [_ems(*[(pref * c, lam) for (c, lam) in G4_MODES[0].modes])
             ] + G4_MODES[1:]
    fe = _fast_eval(modes, G4_EDGES, 3)
    info = FI._NquadModes.from_pole_tuples(G4_EDGES, _pole_tuples(modes))
    FI._reset_runtime_counters()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        v = FI._integrate_polytope(lambda *a: fe(*a), _resolved(G4_EDGES,
                                                                [0.0]),
                                   [0.0], 3, mode_info=info)
    assert _rel(v, pref * G4_REF) < 1e-12, v
    assert _counter('nquad_hardened_quad_flags') == 0
    assert _counter('nquad_hardened_quad_retries') == 0


def _many_modes(rng, kind, n_edges, n_modes):
    """Random mode data (decaying, oscillating) on the chain-like edges
    t − s0... with s_0 in every edge."""
    edges = []
    for e in range(n_edges):
        ip = ((0, -1.0), (1, 1.0)) if e % 2 == 0 else ((0, 1.0),)
        edges.append((ip, ((0, 1.0),) if e % 2 == 0 else (), 0.3 * e))
    if kind == 'product':
        ct = tuple(tuple((complex(rng.uniform(-1, 1), rng.uniform(-1, 1)),
                          complex(-rng.uniform(0.1, 3), rng.uniform(-2, 2)))
                         for _ in range(n_modes)) for _ in range(n_edges))
    else:
        ct = tuple((complex(rng.uniform(-1, 1), rng.uniform(-1, 1)),
                    tuple(complex(-rng.uniform(0.1, 3), rng.uniform(-2, 2))
                          for _ in range(n_edges)))
                   for _ in range(n_modes ** n_edges))
    return FI._NquadModes(kind, 0.7 - 0.2j, tuple(edges), ct)


@pytest.mark.parametrize('kind', ('product', 'sum'))
def test_closed_form_innermost_numpy_matches_the_scalar_loop(kind):
    """The numpy evaluation (many terms) against the scalar loop and a
    50-digit mpmath sum of the same exponential integrals."""
    import mpmath as mp
    rng = random.Random(23)
    mi = _many_modes(rng, kind, 3, 8)                  # 512 terms
    assert mi.innermost_terms() == 512 and mi.innermost_available()
    for (L, U, outer, free) in ((-1.3, -0.2, (-0.5,), [0.4]),
                                (-0.9, -0.9 + 1e-3, (-0.1,), [0.2]),
                                (-30.0, -0.5, (0.3,), [0.9])):
        a = mi.integrate_innermost(L, U, outer, free)
        b = mi._integrate_innermost_scalar(L, U, outer, free)
        with mp.workdps(50):
            ref = mp.mpc(0)
            x = mi._expansion0()
            if kind == 'product':
                import itertools
                for combo in itertools.product(*mi.cterms):
                    c = mp.mpc(mi.pref)
                    o = al = mp.mpc(0)
                    for (C, lam), ec in zip(combo, x[2]):
                        cv = FI._NquadModes._cval(ec, outer, free)
                        c *= mp.mpc(C)
                        o += mp.mpc(lam) * mp.mpf(cv)
                        al += mp.mpc(lam) * mp.mpf(ec[0])
                    ref += c * (mp.exp(o + al * U) - mp.exp(o + al * L)) / al
            else:
                for (B, lams) in mi.cterms:
                    o = al = mp.mpc(0)
                    for lam, ec in zip(lams, x[2]):
                        o += mp.mpc(lam) * mp.mpf(
                            FI._NquadModes._cval(ec, outer, free))
                        al += mp.mpc(lam) * mp.mpf(ec[0])
                    ref += mp.mpc(mi.pref) * mp.mpc(B) * (
                        mp.exp(o + al * U) - mp.exp(o + al * L)) / al
            ref = complex(ref)
        assert abs(a - ref) <= 1e-12 * abs(ref), (a, ref)
        assert abs(b - ref) <= 1e-11 * abs(ref), (b, ref)


@pytest.mark.parametrize('kind', ('product', 'sum'))
def test_closed_form_innermost_takes_the_numpy_path(kind, monkeypatch):
    """4096 terms (3 edges with s_0 and 16 modes each; grouped: 4096 pole
    tuples), expanded once per mode object (``_expansion0``) and evaluated
    with numpy: per call O(edges x modes) exponentials (per diagram: the
    per-edge factors at s_0 = L and U, combined by outer products) or one
    vectorised exponential per end over the pole tuples (grouped) -- never
    the scalar loop over the terms (``_integrate_innermost_scalar``, which
    took 1.9 ms per call at 4096 terms, 15x the numpy path).  Counted, not
    timed: a wall-clock ratio was flaky on a loaded machine."""
    rng = random.Random(5)
    mi = _many_modes(rng, kind, 3, 16)
    assert mi.innermost_terms() == 4096
    x = mi._expansion0()
    assert mi._expansion0() is x                    # built once
    args = (-1.3, -0.2, (-0.5,), [0.4])
    ref = mi._integrate_innermost_scalar(*args)

    def no_scalar(*a, **k):
        raise AssertionError('scalar loop used for 4096 terms')
    monkeypatch.setattr(FI._NquadModes, '_integrate_innermost_scalar',
                        no_scalar)
    n_exp = [0, 0]                       # numpy exp calls, elements
    n_cexp = [0]
    real_exp = np.exp

    def counting_exp(a, *k, **kw):
        n_exp[0] += 1
        n_exp[1] += np.size(a)
        return real_exp(a, *k, **kw)

    class _CountingCmath:
        def __getattr__(self, name):
            import cmath
            return getattr(cmath, name)

        @staticmethod
        def exp(z):
            import cmath
            n_cexp[0] += 1
            return cmath.exp(z)
    monkeypatch.setattr(FI.np, 'exp', counting_exp)
    monkeypatch.setattr(FI, 'cmath', _CountingCmath())
    a = mi.integrate_innermost(*args)
    monkeypatch.undo()
    assert abs(a - ref) <= 1e-12 * abs(ref), (a, ref)
    if kind == 'product':
        # two ends x three edges with s_0, 16 modes each
        assert n_exp == [6, 6 * 16], n_exp
    else:
        assert n_exp == [2, 2 * 4096], n_exp
    assert n_cexp[0] == 0


def test_closed_form_innermost_cap(hardened, monkeypatch):
    """Beyond _NQUAD_INNERMOST_MAX_TERMS terms s_0 is integrated by quad
    (counted), with the same value."""
    rows = [((-1.0, 1.0), (0.0,), 0.0), ((0.0, -1.0), (1.0,), 0.0),
            ((1.0, 0.0), (0.0,), 3.0)]
    modes = [_ems(*[(1.0 / (j + 1), -0.5 - 0.3 * j) for j in range(4)])] * 3
    fe = _fast_eval(modes, rows, 2)
    ref = FI._integrate_polytope(fe, _resolved(rows, [0.5]), [0.5], 2)
    monkeypatch.setattr(FI, '_NQUAD_INNERMOST_MAX_TERMS', 15)
    fe = _fast_eval(modes, rows, 2)                 # a fresh mode object
    FI._reset_runtime_counters()
    v = FI._integrate_polytope(fe, _resolved(rows, [0.5]), [0.5], 2)
    assert _counter('nquad_hardened_innermost_capped') == 1
    assert abs(v - ref) <= 1e-10 * abs(ref), (v, ref)


# ═══════════════════════════════════════════════════════════════════════
# F. public model: the guard-bailed m = 2 regions of the spike-reset model
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def private_cwd(tmp_path_factory):
    """A private empty cwd for the model and diagram caches (fresh-clone
    state): which regions reach the fallback depends on the diagram
    representatives that a cache selects, and the developer's caches must
    not be read or written."""
    prev = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('phase_j_nquad_hardening_cache'))
    try:
        yield
    finally:
        os.chdir(prev)


def _spike_harness(taus, grouped=False):
    from tests.tools import phase_j_subset_diff as H
    from tests.tools import phase_j_zoo_baseline as Z
    model = H.load_model('single_population_spike_reset_test')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FI.PhaseJNquadFallbackWarning)
        run = H.run(model, k=2, max_ell=1, external_fields=[('n', 1),
                                                             ('n', 2)],
                    parameters=Z.P_SPIKE, tau_grid=list(taus),
                    use_grouped_phase_j=grouped)
    H.compute_references(run, kinds=('c',), ref_max_m=2, scope='fallback')
    return H, run


def _check_spike(taus, grouped, monkeypatch):
    # a fresh registry: the key must come from THIS run (and its source)
    monkeypatch.setattr(FI, '_NQUAD_WARNED', set())
    H, run = _spike_harness(taus, grouped)
    rows, _orph = H.reconstruction_check(run)
    assert all(r['ok'] for r in rows)
    by_key = H._first_record_by_ref_key(run)
    n = 0
    for key, refs in run['refs'].items():
        r = by_key[key]
        assert r['path'] == 'nquad' and r['m'] == 2
        for name in ('c150', 'c400'):
            ref = refs[name]
            d = abs(r['value'] - ref)
            assert d <= max(1e-8 * abs(ref), 1e-14), (name, r['value'], ref)
        n += 1
    assert n > 0                       # the fallback was exercised
    c = run['counters']
    assert c['nquad_hardened_calls'] == c['nquad_calls'] > 0
    assert c['nquad_hardened_quad_flags'] == 0
    # compute_cumulants names the model in the warnings' key (model,
    # source, loop order), for the per-diagram and the grouped build alike
    name = H.load_model('single_population_spike_reset_test')['name']
    src = 'grouped' if grouped else 'per_diagram'
    assert ('fallback', ('model', name), src, 1) in FI._NQUAD_WARNED, sorted(
        map(repr, FI._NQUAD_WARNED))
    assert all(k[1] == ('model', name) and k[2] == src
               for k in FI._NQUAD_WARNED), sorted(map(repr, FI._NQUAD_WARNED))
    return n


@pytest.mark.parametrize('grouped', (False, True), ids=('perdiag',
                                                        'grouped'))
def test_spike_reset_fallback_regions_match_the_exact_fan(grouped, hardened,
                                                          private_cwd,
                                                          monkeypatch):
    """Every m = 2 region the fallback served at τ = 0 (evaluated at
    −1e-6) and τ = 10 is within max(1e-8·|ref|, 1e-14) of the 50-digit fan
    formula (box 150 and box 400), per diagram and grouped.  Before M2b the
    same regions were off by up to 6.8e-5 relative / 1.5e-10 absolute (18
    of 40 over the five-point grid; measured with the harness, plan §10)."""
    _check_spike([0.0, 10.0], grouped=grouped, monkeypatch=monkeypatch)


def test_compute_cumulants_scopes_its_callables(hardened, private_cwd,
                                                fresh_warning_state):
    """The callables ``compute_cumulants`` returns open an
    ``nquad_warning_scope`` per call (``api.compute._phase_j_scoped``):
    every region of one call is aggregated into one warning.  A cheap model
    (no fallback) shows the wrapping; direct calls show the aggregation."""
    from api import compute_cumulants
    from api.compute import _phase_j_scoped
    from tests.tools import phase_j_zoo_baseline as Z
    import daedalus as dd
    model, _ = dd.load_model('ou_quartic')
    res = compute_cumulants(model, k=2, max_ell=0,
                            external_fields=[('dx', 1)] * 2,
                            parameters=Z.P_OU, use_cache=True,
                            parallel=False, verbose=False,
                            tau_grid=np.array([0.0, 1.0]))
    for fn in list(res['total_C_by_ell'].values()) + [res['total_C']]:
        assert fn.__qualname__ == '_phase_j_scoped.<locals>.scoped', fn
    td = _typed(_EDGES_A, _LEGS_A)
    calls = [_region_a(_meta(td, sid=j)) for j in range(3)]
    scoped = _phase_j_scoped(lambda: [c() for c in calls])
    msgs = _warnings_of([scoped])
    assert len(msgs) == 1 and 'answered 3 integration region(s)' in msgs[0]


def test_compute_cumulants_aggregates_its_tau_grid(hardened, private_cwd,
                                                   fresh_warning_state):
    """``compute_cumulants``' own τ-grid evaluation is one evaluation (an
    ``nquad_warning_scope`` around ``total_C_batch``): the grouped
    spike-reset k = 2, ℓ = 1 build at τ = 0 and 10 serves 4 regions of 2
    grouped diagrams, reported by ONE warning with those counts.  (Without
    the scope the first region warned at once, alone: '1 integration
    region(s) of 1 diagram(s)'.)"""
    from api import compute_cumulants
    from tests.tools import phase_j_zoo_baseline as Z
    import daedalus as dd
    model, _ = dd.load_model('single_population_spike_reset_test')
    FI._reset_runtime_counters()
    msgs = _warnings_of([lambda: compute_cumulants(
        model, k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
        parameters=Z.P_SPIKE, use_cache=True, parallel=False,
        verbose=False, tau_grid=np.array([0.0, 10.0]),
        use_grouped_phase_j=True)])
    assert _counter('nquad_hardened_calls') == 4
    assert len(msgs) == 1, msgs
    assert ('answered 4 integration region(s) of 2 diagram(s) of model '
            f"'{model['name']}' (grouped, loop order 1)") in msgs[0], msgs[0]


@pytest.mark.slow
def test_public_callables_aggregate_the_warnings(hardened, private_cwd,
                                                 fresh_warning_state):
    """One call of a callable that ``compute_cumulants`` returns is one
    evaluation: the spike-reset k = 3 tree at (0, 0.4, 1) has a few hundred
    fallback regions over dozens of typed diagrams (before: one warning per
    diagram), reported by ONE warning with their counts; a second call (or
    the master ``total_C``) adds none."""
    from api import compute_cumulants
    from tests.tools import phase_j_zoo_baseline as Z
    import daedalus as dd
    model, _ = dd.load_model('single_population_spike_reset_test')
    res = compute_cumulants(model, k=3, max_ell=0,
                            external_fields=[('n', 1), ('n', 2), ('n', 1)],
                            parameters=Z.P_SPIKE, use_cache=True,
                            parallel=False, verbose=False,
                            tau_grid=np.array([0.0]))
    FI._reset_runtime_counters()
    msgs = _warnings_of([lambda: res['total_C_by_ell'][0](0.0, 0.4, 1.0)])
    n = _counter('nquad_hardened_calls')
    assert n > 10
    assert len(msgs) == 1, msgs
    assert f'answered {n} integration region(s)' in msgs[0], msgs[0]
    assert (f"of model '{model['name']}' (per_diagram, loop order 0)"
            in msgs[0]), msgs[0]
    assert _warnings_of([lambda: res['total_C_by_ell'][0](0.0, 0.4, 1.0),
                         lambda: res['total_C'](0.0, 0.7, 0.7)]) == []


@pytest.mark.slow
@pytest.mark.parametrize('grouped', (False, True), ids=('perdiag',
                                                        'grouped'))
def test_spike_reset_fallback_regions_full_grid(grouped, hardened,
                                                private_cwd, monkeypatch):
    _check_spike([0.0, 1.0, 3.0, 5.0, 10.0], grouped=grouped,
                 monkeypatch=monkeypatch)


# ═══════════════════════════════════════════════════════════════════════
# G. the flag off is the pre-M2b engine, bit for bit (in-process)
# ═══════════════════════════════════════════════════════════════════════

def _git_source(rel, rev):
    try:
        return subprocess.run(
            ['git', '-C', _REPO, 'show', f'{rev}:{rel}'], check=True,
            capture_output=True, text=True, timeout=60).stdout
    except Exception:                                   # noqa: BLE001
        pytest.skip(f'git revision {rev} unavailable')


@pytest.fixture(scope='module')
def pre_m2b_fi():
    """The pre-M2b ``final_integral`` exec'd into a fresh module object in
    THIS process (plan §4.4: bit-identity is an in-process comparison).
    It is not registered in ``sys.modules``."""
    src = _git_source(_FI_REL, _PRE_M2B_REV)
    mod = types.ModuleType('_pre_m2b_final_integral')
    mod.__file__ = FI.__file__
    mod.__package__ = 'engine.integration.time_domain'
    exec(compile(src, FI.__file__, 'exec'), mod.__dict__)
    assert not hasattr(mod, 'NQUAD_HARDENED')
    return mod


def _hand_built_regions():
    """(name, rows, modes, free, m): the peaks, an open direction, the
    empty-interval box, a positive-width cycle and a 3-term row."""
    out = [('m2', ROWS_A, MODES_A, [T_A], 2),
           ('m3', ROWS_B, MODES_B, [T_B], 3),
           ('open', [((-1.0, 0.0), (1.0,), 0.0),
                     ((1.0, -1.0), (0.0,), 0.0)],
            [_ems((2.0, -1.5)), _ems((1.0, -0.7), (0.5, -2.0 + 1j))], [0.4],
            2),
           ('m1', [((-1.0,), (1.0,), 0.0), ((1.0,), (-1.0,), 2.0)],
            [_ems((1.0, -0.3 + 2j))], [0.1], 1),
           ('strip', [((1.0, -1.0), (0.0,), 1e-3), ((-1.0, 1.0), (0.0,), 0.5),
                      ((0.0, -1.0), (1.0,), 0.0)],
            [_ems((1.0, -1.0)), _ems((1.0, -2.0))], [0.0], 2)]
    return out


def test_flag_off_is_the_pre_m2b_engine_on_hand_built_regions(pre_m2b_fi,
                                                              legacy_nquad):
    for (name, rows, modes, free, m) in _hand_built_regions():
        new = FI._integrate_polytope(_fast_eval(modes, rows, m),
                                     _resolved(rows, free), free, m)
        old = pre_m2b_fi._integrate_polytope(
            _fast_eval(modes, rows, m, module=pre_m2b_fi),
            _resolved(rows, free), free, m)
        assert np.array_equal(np.array([new]).view(np.uint64),
                              np.array([old]).view(np.uint64)), (name, new,
                                                                old)
    # the evaluators themselves: bit-identical values
    fe_new = _fast_eval(MODES_B, ROWS_B, 3)
    fe_old = _fast_eval(MODES_B, ROWS_B, 3, module=pre_m2b_fi)
    for s in ((9.5, 8.0, 7.0), (1.0, -2.0, -30.0), (9.99, 9.98, 9.9)):
        assert fe_new(*s, T_B) == fe_old(*s, T_B)


_CHILD = textwrap.dedent(r'''
    import os, sys, types, subprocess, warnings
    import numpy as np
    ROOT, REV, CWD = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, ROOT)
    warnings.simplefilter('ignore')
    ENVV = ('DAEDALUS_PHASE_J_LEGACY', 'DAEDALUS_PHASE_J_THETA0_CONST_ROW',
            'DAEDALUS_PHASE_J_STRUCTURAL_ZEROS',
            'DAEDALUS_PHASE_J_NQUAD_HARDENED')
    for v in ENVV:
        os.environ.pop(v, None)
    import daedalus as dd
    import engine.integration.time_domain as TD
    import engine.integration.time_domain.final_integral as FI0
    import engine.integration.time_domain.grouped_integral as GI0
    import engine.integration.time_domain.pipeline as PL
    import api._grouped_phase_j as GP
    from api import compute_cumulants
    from tests.tools import phase_j_zoo_baseline as Z
    rel = 'engine/integration/time_domain/'
    src = {n: subprocess.check_output(
        ['git', '-C', ROOT, 'show', f'{REV}:{rel}{n}.py']).decode()
        for n in ('final_integral', 'grouped_integral')}
    new = {n: open(os.path.join(ROOT, rel + n + '.py')).read()
           for n in ('final_integral', 'grouped_integral')}
    loaded = {'final_integral': [FI0], 'grouped_integral': [GI0]}

    def load(which, env=None):
        for v in ENVV:
            os.environ.pop(v, None)
        os.environ.update(env or {})
        mods = {}
        for n in ('final_integral', 'grouped_integral'):
            name = f'engine.integration.time_domain.{n}'
            m = types.ModuleType(name)
            m.__file__ = getattr(TD, n).__file__
            m.__package__ = 'engine.integration.time_domain'
            sys.modules[name] = m
            setattr(TD, n, m)
            exec(compile((src if which == 'old' else new)[n], m.__file__,
                         'exec'), m.__dict__)
            mods[n] = m
        for v in ENVV:
            os.environ.pop(v, None)
        fi, gi = mods['final_integral'], mods['grouped_integral']
        for mod in list(sys.modules.values()):
            d = getattr(mod, '__dict__', None)
            if not isinstance(d, dict) or mod is fi or mod is gi:
                continue
            for k, v in list(d.items()):
                for n, m in mods.items():
                    if isinstance(v, types.ModuleType) and any(
                            v is o for o in loaded[n]):
                        setattr(mod, k, m)
                    elif (isinstance(v, (types.FunctionType, type))
                          and getattr(v, '__module__', '') == m.__name__
                          and k in m.__dict__):
                        setattr(mod, k, m.__dict__[k])
        for n, m in mods.items():
            loaded[n].append(m)
        assert PL.integrate_diagram is fi.integrate_diagram
        assert GP.integrate_grouped_diagram is gi.integrate_grouped_diagram
        assert gi._fi_mod is fi
        return fi

    os.chdir(CWD)
    # (name, model, kwargs, flag-on run?): the first group makes no
    # fallback call, so the DEFAULT engine (flag on) must equal the old one;
    # the second is served by the fallback, so the flag-OFF engine must.
    CFGS = [
        ('ou', 'ou_quartic', dict(k=2, max_ell=2,
                                  external_fields=[('dx', 1)] * 2,
                                  parameters=Z.P_OU), True),
        ('linh', 'linear_hawkes', dict(k=2, max_ell=1,
                                       external_fields=[('n', 1), ('n', 2)],
                                       parameters=Z.P_LINH), True),
        ('ld', 'single_population_linear_delta_spikes_test',
         dict(k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
              parameters=Z.P_LD), True),
        ('mp', 'multipopulation_test',
         dict(k=2, max_ell=1, external_fields=[('nE', 1), ('nE', 2)],
              parameters=Z.P_MP), True),
        ('spike', 'single_population_spike_reset_test',
         dict(k=2, max_ell=1, external_fields=[('n', 1), ('n', 2)],
              parameters=Z.P_SPIKE), False),
    ]
    MODELS = {}

    def run_all(fi, which):
        out = {}
        for name, mname, kw, on in CFGS:
            if which == 'on' and not on:
                continue
            if which == 'off' and on:
                continue
            model = MODELS.setdefault(mname, dd.load_model(mname)[0])
            for grouped in (False, True):
                fi._reset_runtime_counters()
                res = compute_cumulants(
                    model, tau_grid=np.array([0.0, 2.5, 10.0]),
                    use_cache=True, parallel=False, verbose=False,
                    use_grouped_phase_j=grouped, **kw)
                v = np.concatenate([np.asarray(a, complex) for _l, a in
                                    sorted(res['C_tau_by_ell'].items())])
                calls = fi._RUNTIME_COUNTERS['nquad_calls']
                out[(name, grouped)] = (v.view(np.uint64).copy(), calls)
        return out

    fi = load('new')
    run_all(fi, 'on'), run_all(fi, 'off')            # warm the caches
    fi = load('old')
    old_on, old_off = run_all(fi, 'on'), run_all(fi, 'off')
    bad = []
    fi = load('new')                                 # defaults: flag on
    assert fi.NQUAD_HARDENED is True
    got = run_all(fi, 'on')
    for key in old_on:
        if got[key][1] != 0 or old_on[key][1] != 0:
            bad.append(('made fallback calls', key, got[key][1]))
        if not np.array_equal(old_on[key][0], got[key][0]):
            bad.append(('on', key))
    fi = load('new', {'DAEDALUS_PHASE_J_NQUAD_HARDENED': '0'})
    assert fi.NQUAD_HARDENED is False
    got = run_all(fi, 'off')
    for key in old_off:
        if old_off[key][1] == 0:
            bad.append(('no fallback call to compare', key))
        if not np.array_equal(old_off[key][0], got[key][0]):
            bad.append(('off', key))
    fi = load('new', {'DAEDALUS_PHASE_J_LEGACY': '1'})
    umb = run_all(fi, 'off')
    fi = load('old', {'DAEDALUS_PHASE_J_LEGACY': '1'})
    old_umb = run_all(fi, 'off')
    for key in old_umb:
        if not np.array_equal(old_umb[key][0], umb[key][0]):
            bad.append(('umbrella', key))
    print('BAD', bad)
    print('RESULT', 'OK' if not bad else 'FAIL')
''')


@pytest.mark.slow
def test_bit_identical_to_the_pre_m2b_engine(tmp_path):
    """Models without fallback calls (``ou_quartic``, ``linear_hawkes``,
    ``single_population_linear_delta_spikes_test``,
    ``multipopulation_test``): the default engine (flag on) equals the
    engine of ``_PRE_M2B_REV`` bit for bit.  The spike-reset model (served
    by the fallback): the flag-off engine equals it, and so does the legacy
    umbrella.  Per diagram and grouped, IN ONE PROCESS (the sources are
    exec'd into fresh module objects side by side, in a child process)."""
    _git_source(_FI_REL, _PRE_M2B_REV)                  # skip without git
    child = tmp_path / 'ab_child.py'
    child.write_text(_CHILD)
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    env = dict(os.environ, PYTHONHASHSEED='0')
    r = subprocess.run([sys.executable, str(child), _REPO, _PRE_M2B_REV,
                        str(cwd)], capture_output=True, text=True,
                       timeout=3600, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    assert 'RESULT OK' in r.stdout, r.stdout[-3000:]


# ═══════════════════════════════════════════════════════════════════════
# H. the one-time warning and the counters
# ═══════════════════════════════════════════════════════════════════════

def _typed(edges, legs, n_vertices=4):
    """A TypedDiagram with the given edge types and external legs (the
    prediagram only needs ``vertices()``)."""
    from engine.diagrams.type_assignment import TypedDiagram
    D = types.SimpleNamespace(vertices=lambda: list(range(n_vertices)))
    return TypedDiagram((D, None, (), ()), {}, dict(edges), dict(legs), {})


def _warnings_of(calls):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        for c in calls:
            c()
    return [str(x.message) for x in w
            if issubclass(x.category, FI.PhaseJNquadFallbackWarning)]


@pytest.fixture
def fresh_warning_state(monkeypatch):
    """Empty warning registries and no open scope (restored afterwards)."""
    monkeypatch.setattr(FI, '_NQUAD_WARNED', set())
    monkeypatch.setattr(FI, '_NQUAD_PENDING', {})
    monkeypatch.setattr(FI, '_NQUAD_LOGGED', set())
    monkeypatch.setitem(FI._NQUAD_SCOPE, 'depth', 0)
    monkeypatch.setitem(FI._NQUAD_SCOPE, 'pid', None)


def _meta(td, sid=5, src='per_diagram', loop=1, model=('model', 'model_a'),
          serial=1):
    return {'source': src, 'diagram_serial': serial, 'loop_number': loop,
            'subset_id': sid, 'diagram': td, 'model': model}


_EDGES_A = {(0, 2, 0): (('nt', 1), ('n', 1)), (1, 2, 0): (('nt', 2), ('n', 2))}
_LEGS_A = {0: ('n', 1), 1: ('n', 2)}
_LEGS_B = {0: ('n', 1), 1: ('n', 1)}


def _region_a(md, f=None):
    fe = f if f is not None else _fast_eval(MODES_A, ROWS_A, 2)
    return lambda: FI._integrate_polytope(fe, _resolved(ROWS_A, [T_A]),
                                          [T_A], 2, diag_meta=md)


def test_one_warning_per_model_source_and_loop_order(hardened,
                                                     fresh_warning_state):
    """The regions of one evaluation (an ``nquad_warning_scope``, which the
    callables of ``compute_cumulants`` open) are aggregated per (model,
    source, loop order) into ONE warning with their counts, issued when the
    scope ends; each key warns once per process.  (Before: one warning per
    typed diagram, hundreds for a k = 4 evaluation.)"""
    td1, td2 = _typed(_EDGES_A, _LEGS_A), _typed(_EDGES_A, _LEGS_B)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FI.nquad_warning_scope():
            _region_a(_meta(td1))()
            _region_a(_meta(td1, sid=6))()            # another subset
            _region_a(_meta(td2))()                   # another diagram
            with FI.nquad_warning_scope():            # nested: no flush
                _region_a(_meta(td2, sid=7))()
            assert not w                              # deferred
    msgs = [str(x.message) for x in w
            if issubclass(x.category, FI.PhaseJNquadFallbackWarning)]
    assert len(msgs) == 1, msgs
    assert ('4 integration region(s) of 2 diagram(s) of model '
            "'model_a' (per_diagram, loop order 1)") in msgs[0], msgs[0]
    assert 'First: δ-subset 0b101' in msgs[0]
    assert "external legs ['n1', 'n2']" in msgs[0]
    assert not any('serial' in x for x in msgs)
    # the same model again (a rebuild: new serial, new TypedDiagram objects,
    # other parameter values): nothing
    with FI.nquad_warning_scope():
        msgs = _warnings_of([_region_a(_meta(_typed(_EDGES_A, _LEGS_A),
                                             serial=2))])
    assert msgs == []
    # another loop order, the grouped source, another model: one each
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FI.nquad_warning_scope():
            _region_a(_meta(td1, loop=2))()
            _region_a(_meta([td1, td2], src='grouped'))()
            _region_a(_meta(td1, model=('model', 'model_b')))()
            _region_a(_meta(td1, model=('model', 'model_b')))()
    msgs = sorted(str(x.message) for x in w
                  if issubclass(x.category, FI.PhaseJNquadFallbackWarning))
    assert len(msgs) == 3, msgs
    assert any("'model_a' (per_diagram, loop order 2)" in x for x in msgs)
    assert any("'model_a' (grouped, loop order 1)" in x
               and 'a group of 2 typed diagrams' in x for x in msgs)
    assert any("2 integration region(s) of 1 diagram(s) of model "
               "'model_b'" in x for x in msgs)
    # outside any scope: at once, once per key
    msgs = _warnings_of([_region_a(_meta(td1, model=('model', 'm_c')))] * 2)
    assert len(msgs) == 1 and '1 integration region(s)' in msgs[0]
    # regions without mode data are counted in the message
    fe = _fast_eval(MODES_A, ROWS_A, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FI.nquad_warning_scope():
            _region_a(_meta(td1, model=('model', 'm_d')),
                      f=lambda *a: fe(*a))()
            _region_a(_meta(td1, model=('model', 'm_d')))()
            assert not w and FI._NQUAD_PENDING        # until the scope ends
    msgs = [str(x.message) for x in w]
    assert len(msgs) == 1 and '2 integration region(s)' in msgs[0]
    assert "1 without the integrand's exponential modes" in msgs[0]
    assert ('fallback', ('model', 'm_d'), 'per_diagram', 1) in FI._NQUAD_WARNED


def test_warning_scope_in_a_forked_child_warns_at_once(hardened,
                                                       fresh_warning_state):
    """A process that inherited an open scope (a forked worker of a
    parallel batch) never sees it close: it warns at once instead."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FI.nquad_warning_scope():
            FI._NQUAD_SCOPE['pid'] = -1          # as after a fork
            _region_a(_meta(_typed(_EDGES_A, _LEGS_A)))()
            assert len(w) == 1
            FI._NQUAD_SCOPE['pid'] = os.getpid()
    assert len(w) == 1 and FI._NQUAD_SCOPE['depth'] == 0


def test_debug_log_names_each_diagram_once(hardened, fresh_warning_state,
                                           caplog):
    td1, td2 = _typed(_EDGES_A, _LEGS_A), _typed(_EDGES_A, _LEGS_B)
    with caplog.at_level('DEBUG', logger=FI.__name__):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for md in (_meta(td1), _meta(td1, sid=6), _meta(td2),
                       _meta(_typed(_EDGES_A, _LEGS_A), serial=9)):
                _region_a(md)()
    recs = [r.getMessage() for r in caplog.records if r.name == FI.__name__]
    assert len(recs) == 2, recs
    assert "['n1', 'n2']" in recs[0] and "['n1', 'n1']" in recs[1]


def test_warning_key_includes_the_model(hardened, fresh_warning_state):
    """Two models whose typed diagrams have the same structure each warn
    once; rebuilding a model (a new ``propagator_data`` dict, other
    parameter values) does not warn again.  An unnamed model is keyed by a
    serial stamped into its ``propagator_data``."""
    td = _typed(_EDGES_A, _LEGS_A)

    def call(pd, serial):
        return _region_a(_meta(td, sid=3, serial=serial,
                               model=FI._model_identity(pd)))
    pd_u1, pd_u2 = {}, {}
    msgs = _warnings_of([call({'model_name': 'model_a'}, 1),
                         call({'model_name': 'model_b'}, 2),
                         call({'model_name': 'model_a'}, 3),   # rebuild
                         call(pd_u1, 4), call(pd_u1, 5), call(pd_u2, 6)])
    assert len(msgs) == 4, msgs
    assert sum('an unnamed model' in x for x in msgs) == 2
    assert FI._model_identity({'model_name': 'm'}) == ('model', 'm')
    assert pd_u1['_nquad_warn_serial'] != pd_u2['_nquad_warn_serial']
    assert FI._model_identity(pd_u1) == FI._model_identity(pd_u1)


# The flag rule: an inner quad call whose own error estimate is above its
# requested tolerance (1e-13 absolute, 1e-10 relative) but which, times the
# widths of its enclosing levels, stays within NQUAD_EPSREL × the region's
# scale is not a flag (the rounding of an inner call's own values can keep
# it above a tight absolute tolerance without any loss at the region's
# tolerance); above that it is counted and warned about.  Region: 0 < s1 <
# 1/2 (outer), 10 < s0 < 21/2 (inner), integrand 1 without mode data (scale
# = its sampled maximum 1; |I| = 1/4, no second pass), so an inner call
# returns 1/2 (relative tolerance 5e-11) with the enclosing width 1/2:
# 1.5e-10 is not counted (1.5e-10 x 1/2 <= 1e-10), 1e-8 is.
FLAG_ROWS = [((0.0, 1.0), (), 0.0), ((0.0, -1.0), (), 0.5),
             ((1.0, 0.0), (), -10.0), ((-1.0, 0.0), (), 10.5)]


def _faking_quad(abserr):
    import scipy.integrate as si
    real = si.quad

    def fake(func, a, b, *args, **kw):
        r = real(func, a, b, *args, **kw)
        if a < 5.0:                                   # the outer level
            return r
        return (r[0], abserr, {'last': 1, 'neval': 21},
                'The occurrence of roundoff error is detected (fake)')
    return fake


@pytest.mark.parametrize('abserr,flagged', ((1.5e-10, False), (1e-8, True)),
                         ids=('within-region-tolerance', 'above'))
def test_flag_rule_integrates_the_error_over_the_enclosing_levels(
        abserr, flagged, hardened, fresh_warning_state, monkeypatch):
    import scipy.integrate as si
    monkeypatch.setattr(si, 'quad', _faking_quad(abserr))
    rows = _resolved(FLAG_ROWS, [])
    FI._reset_runtime_counters()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FI.nquad_warning_scope():
            for _ in range(2):                        # two regions
                v = FI._integrate_polytope(lambda *a: 1.0, rows, [], 2,
                                           diag_meta=_meta(None))
    assert abs(v - 0.25) < 1e-12, v
    assert _counter('nquad_hardened_reruns') == 0
    flags = _counter('nquad_hardened_quad_flags')
    msgs = [str(x.message) for x in w if 'did not reach' in str(x.message)]
    if flagged:
        assert flags > 0
        assert len(msgs) == 1 and 'on 2 of 2 region(s)' in msgs[0], msgs
        assert '(fake)' in msgs[0]
        # once per model, source and loop order and process: a later
        # evaluation that flags the same key again adds no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with FI.nquad_warning_scope():
                FI._integrate_polytope(lambda *a: 1.0, rows, [], 2,
                                       diag_meta=_meta(None))
        assert _counter('nquad_hardened_quad_flags') > flags
        assert [str(x.message) for x in w] == []
    else:
        assert flags == 0 and msgs == [], msgs


_HARDENED_COUNTERS = (
    'nquad_hardened_calls', 'nquad_hardened_empty',
    'nquad_hardened_empty_intervals', 'nquad_hardened_capped',
    'nquad_hardened_cap_span_max', 'nquad_hardened_uncertified',
    'nquad_hardened_no_modes', 'nquad_hardened_general_rows',
    'nquad_hardened_fm_capped', 'nquad_hardened_kinks_incomplete',
    'nquad_hardened_quad_flags', 'nquad_hardened_quad_retries',
    'nquad_hardened_reruns', 'nquad_hardened_tail_widened',
    'nquad_hardened_inner_unchecked', 'nquad_hardened_innermost_overflow',
    'nquad_hardened_floor_limited',
    'nquad_hardened_innermost_capped')


def test_counters_reset():
    """Every hardened-fallback counter is registered (a counter missing from
    the registry fails here), and the reset zeroes each of them."""
    missing = [k for k in _HARDENED_COUNTERS
               if k not in FI._RUNTIME_COUNTERS]
    assert not missing, missing
    registered = sorted(k for k in FI._RUNTIME_COUNTERS
                        if k.startswith('nquad_hardened_'))
    assert registered == sorted(_HARDENED_COUNTERS)
    for i, k in enumerate(_HARDENED_COUNTERS):
        FI._RUNTIME_COUNTERS[k] = i + 1.5 if k.endswith('_max') else i + 1
    FI._reset_runtime_counters()
    assert all(_counter(k) == 0 for k in _HARDENED_COUNTERS)

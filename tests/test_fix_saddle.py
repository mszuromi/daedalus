"""Regression tests for the mean-field saddle solver + verification gate.

Two correctness fixes are covered:

FIX A — ``api/_mean_field_dae._seed_box_default`` keyed its per-field
sampling box by the INTERNAL field name while iterating NATURAL names, so
the lookup always missed and ``or 'positive'`` always fired: every state
variable was sampled on ``[0, 5·scale]`` and the symmetric / negative
half-line was dead code.  Consequence: a double-well at ``μ<0`` (roots 0
and ``±√(-μ/ε)``) never found its NEGATIVE well, so the wrong saddle was
silently selected.  The fix resolves the domain off the saddle PARAMETER
(``<natural>star``) and, for fields with no declared domain under a
stability-classified theory, probes the SYMMETRIC box so both wells are
found.  (Gated on ``stability_analysis`` so stability-OFF theories — whose
``fixed_point_index`` ranges over EVERY sorted root — keep their exact
historical selection.)

FIX B — ``engine/core/field_theory._mf_numerical_residual`` bound every
leftover free symbol to 1.0 and evaluated ONCE, so a residual that is
non-zero in general but happens to vanish at all-ones was silently
soft-PASSED by the MF-sector verification gate.  The fix distinguishes a
genuinely un-pinned symbol (probe several distinct points) from a model
PARAMETER missing a value (substitute the real value if available, else
give up — do NOT probe, which would false-FAIL a residual that is zero at
the true parameter value).

Run:  sage -python -m pytest tests/test_fix_saddle.py -q
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from sage.all import SR  # noqa: E402

from api.theory import TemporalTheoryBuilder  # noqa: E402
from api._mean_field_dae import (  # noqa: E402
    solve_mean_field_dae, _seed_box_default, _state_variables,
)
from engine.core.field_theory import (  # noqa: E402
    FieldTheory, _mf_numerical_residual, _model_parameter_symbol_names,
)


def _double_well_model(stability=True):
    """The shipped OU-quartic double-well, built so we can drive μ<0."""
    return (
        TemporalTheoryBuilder('OU Quartic Double Well')
        .population('pop', size=1)
        .physical_field('x', population='pop', description='variable')
        .parameter('mu', default=0.1, domain='real')
        .parameter('eps', default=0.1, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .set_action_text(
            'sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
        .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
        .stability_analysis(stability)
        .build()
    )


# ── FIX A ────────────────────────────────────────────────────────────


def test_seed_box_symmetric_for_undeclared_domain_field():
    """A field with no declared domain (``x``) under a stability-classified
    theory must get a SYMMETRIC seed box, not the old positive-only one."""
    model = _double_well_model(stability=True)
    sv = _state_variables(model)
    boxes = _seed_box_default(sv, model, {'mu': -1.0, 'eps': 1.0, 'D': 1.0})
    lo, hi = boxes['x']
    assert lo < 0.0 < hi, boxes      # negative half-line is sampled
    assert lo == -hi                 # symmetric


def test_double_well_finds_both_wells_at_mu_negative():
    """At μ=-1, ε=1 the saddle equation μ·x + ε·x³ = 0 has roots {-1, 0, +1};
    the two non-trivial wells (±1) are the stable ones.  The OLD positive-only
    box never sampled the negative well, so only {0, +1} were found and the
    -1 well was silently missing.  Both wells must now be present."""
    model = _double_well_model(stability=True)
    fund = {'mu': -1.0, 'eps': 1.0, 'D': 1.0}
    res = solve_mean_field_dae(model, fund, n_starts=64, fixed_point_index=0)

    all_roots = sorted(round(r['values']['xstar'][0], 4)
                       for r in res['mf_all_roots'])
    assert all_roots == [-1.0, 0.0, 1.0], all_roots

    stable = sorted(round(v['xstar'][0], 4) for v in res['mf_stable_roots'])
    assert stable == [-1.0, 1.0], stable       # BOTH wells, not just +1

    # fixed_point_index selects over the (sorted) stable subset, so both
    # wells are individually addressable.
    neg = solve_mean_field_dae(model, fund, fixed_point_index=0)['mf_values']
    pos = solve_mean_field_dae(model, fund, fixed_point_index=1)['mf_values']
    assert round(neg['xstar'][0], 4) == -1.0
    assert round(pos['xstar'][0], 4) == 1.0


def test_stability_off_keeps_positive_box():
    """Stability-OFF theories select fixed_point_index over EVERY sorted
    root; widening their box would silently shift that selection.  Such a
    field keeps the positive box so the shipped selection is preserved."""
    model = _double_well_model(stability=False)
    sv = _state_variables(model)
    boxes = _seed_box_default(sv, model, {'mu': -1.0, 'eps': 1.0, 'D': 1.0})
    assert boxes['x'][0] == 0.0      # positive box retained


# ── FIX B ────────────────────────────────────────────────────────────


def _double_well_ns(fundamental_defaults=None):
    model = _double_well_model(stability=True)
    if fundamental_defaults is not None:
        model['fundamental_defaults'] = fundamental_defaults
    ft = FieldTheory(model)
    ns, _R, _n = ft._build_namespace()
    ft._ns = ns
    return ns, model


def test_residual_zero_at_allones_but_nonzero_in_general_is_rejected():
    """The core FIX-B bug: a residual that vanishes at the all-ones point
    but is NON-ZERO in general must NOT be soft-passed.  ``s - 1`` (with an
    un-pinned dummy ``s``) is zero at s=1 — the old code returned 0.0 and
    the MF-sector gate soft-PASSED it.  The multi-point probe must now
    report a non-zero residual so the gate rejects it."""
    ns, model = _double_well_ns()
    s = SR.var('an_unpinned_dummy_qq')
    resid = s - 1
    mag = _mf_numerical_residual(resid, ns, model)
    assert mag is not None and mag > 1e-6, mag


def test_residual_identically_zero_passes():
    """Control: a residual identically zero in an un-pinned dummy must
    still report ~0 (the gate soft-passes it)."""
    ns, model = _double_well_ns()
    s = SR.var('an_unpinned_dummy_qq')
    mag = _mf_numerical_residual(0 * s, ns, model)
    assert mag is not None and mag < 1e-9, mag


def test_true_saddle_residual_softpasses():
    """A genuine saddle-equation residual (μ·x* + ε·x*³, which vanishes by
    construction) must report ~0 so a correct theory still passes."""
    ns, model = _double_well_ns()
    mu, eps = SR.var('mu'), SR.var('eps')
    xstar = ns.xstar[0]
    resid = mu * xstar + eps * xstar ** 3
    mag = _mf_numerical_residual(resid, ns, model)
    assert mag is not None and mag < 1e-9, mag


def test_missing_parameter_is_not_probed_to_a_false_failure():
    """Case (ii): a model PARAMETER the solver can't bind (here ``eps`` is
    omitted from an explicit ``fundamental_defaults``) must NOT be probed at
    arbitrary points — a naive probe would FALSE-FAIL a residual that is
    genuinely zero at the true parameter value.  The function returns None
    (⇒ the gate treats it as un-checkable, not a spurious soft-pass)."""
    ns, model = _double_well_ns(fundamental_defaults={'mu': -1.0, 'D': 1.0})
    eps = SR.var('eps')
    mu = SR.var('mu')
    xstar = ns.xstar[0]
    # Zero at the true saddle, but references the unbound parameter eps.
    resid = mu * xstar + eps * xstar ** 3
    mag = _mf_numerical_residual(resid, ns, model)
    assert mag is None, mag


def test_parameter_symbol_names_include_indexed_forms():
    """The parameter-vs-dummy classifier must recognise indexed parameter
    symbols (vector / matrix), else an indexed parameter would be
    mis-treated as an un-pinned dummy and probed."""
    model = (
        TemporalTheoryBuilder('indexed-param probe')
        .population('E', size=2)
        .physical_field('x', population='E')
        .parameter('a', default=[0.3, 0.4], indexed_by=['E'], domain='real')
        .parameter('mu', default=1.0, domain='real')
        .set_action_text('sum(xt[i]*((Dt+mu)*x[i] + a[i]*x[i]^2) for i in E)')
        .equation(lhs='(Dt+mu)*x[i]', rhs='-a[i]*x[i]^2', population='E')
        .build()
    )
    names = _model_parameter_symbol_names(model)
    assert 'mu' in names
    assert {'a1', 'a2'} <= names      # indexed vector forms recognised

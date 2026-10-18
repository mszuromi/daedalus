"""
api.analyze — a standalone model analyzer: traits, findings and an
assumptions ledger for a model dict, without running the diagram
enumeration or Phase J.

    from api.analyze import analyze
    report = analyze(model)                       # defaults: k=2, ell<=1
    report = analyze(model,
                     question={'k': 2, 'max_ell': 1,
                               'external_fields': [('x', 1), ('x', 1)]},
                     policy={'mode': 'strict',
                             'allow_taylor_truncation': True})
    report.ok, report.codes, report.traits.n_poles, print(report.summary())

``model`` is any dict the builders emit (``TemporalModelBuilder``,
``TemporalTheoryBuilder``, ``SpatialModelBuilder``, ``api.sde.SDE``).
The analyzer reads it and never writes to it, and it neither reads nor
writes the ``saved_models/`` caches.

**Declared by the user** (inputs, never inferred): the model, the
question (:class:`Question`: ``k``, ``external_fields``, ``max_ell``,
``accuracy``, ``budget``, the parameter point, ``fixed_point_index``) and
the policy (:class:`Policy`: ``strict``/``permissive``, consents,
the stochastic interpretation, assertions).

**Detected** by gates G0-G5, cheapest first.  Each returns
:class:`Finding` objects (``code``, ``level`` ``error|warning|info``,
``message``, ``remedy``) instead of raising; ``strict`` promotes warnings
to errors.

- G0 declarations: names, parameter domains and values, the parameter
  point, the question's external fields (with the pipeline's own error
  text), the interpretation convention when the noise is state dependent.
- G1 structure: action evaluation, polynomial or not, vertex species,
  state-dependent noise, unpaired fields (singular operator), undriven
  fields (no noise reaches them through the linear operator), spatial.
- G2 time homogeneity and the mean field: every saddle, its linear
  stability, marginal modes, the validity of ``fixed_point_index``,
  several stable saddles.
- G3 the propagator class from ``G(omega) = K(omega)^-1``: rational or
  not (with the offending factor), poles, repeated and close poles, the
  fast/slow ratio, the instantaneous (delta) component, causality.
- G4 the noise class: white, colored (rational spectrum or not, Markov
  embedded), cumulants of order >= 3, cross-correlations, a non-negative
  covariance at the saddle.

- G5 numerical regime advisories (D3): the loop-parameter estimate ``g_eff``
  (the one-loop/tree size from the couplings, the tree variance and the
  noise amplitude; definition in ``_g_eff``) with a warning when it is
  not small, and for a non-polynomial action a check that the Taylor series
  of the response about the saddle still shrinks at 3 tree standard
  deviations.  The pole statistics (rate range, fast/slow ratio, close and
  repeated poles) are traits and G3 findings with remedies.  Read-only:
  they never change a number and never raise (``strict`` does not promote
  the two warnings: ``_NEVER_PROMOTE``).

``report.table()`` (``str(report)``) is the one-table summary of at most 100
columns; ``report.summary()`` is the long form.

The assumptions ledger lists every consent, convention, assertion and
approximation a result for this question would depend on.

**Wired into** ``compute_cumulants`` (D2): :class:`Preflight` runs the same
gates at the pipeline's stage boundaries on the pipeline's own objects
(see the section "wired mode" below), stops with :class:`AnalyzerError` on
any ``error`` (``Policy.downgrade`` accepts chosen ones), records the
backend the router chose (``api.routing``; a standalone ``analyze`` report
says ``unrouted``) and adds the G6 run diagnostics after the run (the M8
certificate verdicts, the error-accounting counters and a total-relative
accuracy estimate).  ``compute_cumulants(..., preflight=False)`` or
``DAEDALUS_PREFLIGHT=0`` turns it off; the numbers are the same either way.
"""
from __future__ import annotations

import contextlib
import io
import math
import operator
import re
import time
import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

import numpy as np

__all__ = [
    'analyze', 'ModelTraits', 'Finding', 'LedgerEntry', 'ModelReport',
    'Question', 'Policy', 'LEVELS', 'FINDING_CODES',
]

LEVELS = ('error', 'warning', 'info')

#: Every finding code ``analyze`` can emit, with its gate and default level
#: (``strict`` promotes ``warning`` to ``error``).
FINDING_CODES = {
    # G0 declarations
    'G0_MODEL_INVALID':          ('G0', 'error'),
    'G0_NAME_MISSING':           ('G0', 'error'),
    'G0_DUPLICATE_NAME':         ('G0', 'error'),
    'G0_DOMAIN_MISSING':         ('G0', 'info'),
    'G0_DOMAIN_UNKNOWN':         ('G0', 'warning'),
    'G0_DOMAIN_VIOLATION':       ('G0', 'error'),
    'G0_DOMAIN_BOUNDARY':        ('G0', 'warning'),
    'G0_PARAMETERS_MISSING':     ('G0', 'error'),
    'G0_UNKNOWN_PARAMETER':      ('G0', 'warning'),
    'G0_QUESTION_INVALID':       ('G0', 'error'),
    'G0_EXTERNAL_FIELD':         ('G0', 'error'),
    'G0_POLICY_INVALID':         ('G0', 'error'),
    'G0_CONVENTION_MISSING':     ('G0', 'warning'),
    'G0_NAMESPACE':              ('G0', 'error'),
    # G1 structure
    'G1_ACTION_EVAL':            ('G1', 'error'),
    'G1_EXPAND_FAILED':          ('G1', 'error'),
    'G1_NONPOLYNOMIAL':          ('G1', 'info'),
    'G1_TAYLOR_CONSENT_MISSING': ('G1', 'warning'),
    'G1_STATE_DEPENDENT_NOISE':  ('G1', 'info'),
    'G1_UNPAIRED_FIELD':         ('G1', 'error'),
    'G1_SINGULAR_PROPAGATOR':    ('G1', 'error'),
    'G1_UNDRIVEN_FIELD':         ('G1', 'warning'),
    'G1_SPATIAL':                ('G1', 'info'),
    # G2 time homogeneity and mean field
    'G2_TIME_DEPENDENT':         ('G2', 'error'),
    'G2_NOISE_STRUCTURE':        ('G2', 'error'),
    'G2_NO_SADDLE':              ('G2', 'error'),
    'G2_MF_FAILED':              ('G2', 'error'),
    'G2_UNSTABLE_SADDLE':        ('G2', 'error'),
    'G2_MARGINAL_MODE':          ('G2', 'error'),
    'G2_FIXED_POINT_INDEX':      ('G2', 'warning'),   # the pipeline clamps it
    'G2_MULTISTABLE':            ('G2', 'warning'),
    'G2_SINGLE_SADDLE_SOLVER':   ('G2', 'info'),
    'G2_STABILITY_UNAVAILABLE':  ('G2', 'warning'),
    'G2_ASSERTION_VIOLATED':     ('G2', 'error'),
    'G2_SKIPPED':                ('G2', 'info'),
    # G2-K (api._kernel_stability): stability with the true kernel images
    'G2_KERNEL_STABILITY':       ('G2', 'info'),
    'G2_KERNEL_UNSTABLE':        ('G2', 'warning'),   # rational-kernel models (never refused by it)
    'G2_NEUTRAL_DELAY':          ('G2', 'error'),
    # G3 propagator
    'G3_UNRESOLVED_SYMBOLS':     ('G3', 'error'),
    'G3_NONRATIONAL_KERNEL':     ('G3', 'error'),
    'G3_POLES_UNAVAILABLE':      ('G3', 'warning'),
    'G3_REPEATED_POLE':          ('G3', 'info'),      # resolved (api._pole_clusters)
    'G3_CLOSE_POLES':            ('G3', 'info'),
    'G3_POLE_CLUSTER_UNRESOLVED': ('G3', 'warning'),
    'G3_RESIDUES_HIGH_PRECISION': ('G3', 'info'),
    'G3_RESIDUES_NUMERICAL_TIER': ('G3', 'warning'),
    'G3_STIFF':                  ('G3', 'info'),
    'G3_DELTA_COMPONENT':        ('G3', 'info'),
    'G3_IMPROPER':               ('G3', 'error'),
    'G3_ACAUSAL_POLE':           ('G3', 'error'),
    'G3_SKIPPED':                ('G3', 'info'),
    # delay kernels (api.delay): exp(-i omega d) factors, approximated by Pade
    'G3_DELAY_KERNEL':           ('G3', 'info'),
    'G3_DELAY_APPROXIMATED':     ('G3', 'warning'),   # info when the target is met
    # G4 noise
    'G4_NO_NOISE':               ('G4', 'warning'),
    'G4_NON_GAUSSIAN':           ('G4', 'info'),
    'G4_CROSS_CORRELATED':       ('G4', 'info'),
    'G4_COLORED_RATIONAL':       ('G4', 'info'),
    'G4_NONRATIONAL_NOISE':      ('G4', 'warning'),
    'G4_MARKOV_CONSENT_MISSING': ('G4', 'warning'),
    'G4_NEGATIVE_VARIANCE':      ('G4', 'error'),
    # G3 for spatial models (D2): K(omega, q) = i omega + M + V q + D q^2
    'G3_SPATIAL_FORM':           ('G3', 'info'),
    'G3_MASSLESS':               ('G3', 'warning'),
    'G3_SPATIAL_UNSTABLE':       ('G3', 'error'),
    # G5 numerical regime advisories (D3)
    'G5_LOOP_PARAMETER':         ('G5', 'warning'),
    'G5_TAYLOR_RADIUS':          ('G5', 'warning'),
    'G5_ADVISORY_SKIPPED':       ('G5', 'info'),
    # G6 after the run (D2; recorded, never raised)
    'G6_ACCURACY':               ('G6', 'info'),
    'G6_ACCURACY_LOW':           ('G6', 'warning'),
    'G6_DIVERGENT_SUBSETS':      ('G6', 'warning'),
    'G6_DIVERGENT_CANCELLED':    ('G6', 'info'),
    'G6_ILL_CONDITIONED':        ('G6', 'info'),
    'G6_NQUAD_FALLBACK':         ('G6', 'info'),
    'G6_COUNTERS_INCOMPLETE':    ('G6', 'info'),
    'G6_NOT_EVALUATED':          ('G6', 'info'),
    'G6_DIAGRAMS_SKIPPED':       ('G6', 'warning'),
    # S3: the Phase J engine census and the state-space route's fallbacks
    'G6_PHASE_J_ENGINE':         ('G6', 'info'),
    'G6_SS_FALLBACK':            ('G6', 'warning'),
    'G6_SS_PARTIAL':             ('G6', 'warning'),
    # the realization was exactly Hurwitz but numerically unusable (graded
    # beyond double precision): the pole route took the run
    'G6_SS_ILL_CONDITIONED':     ('G6', 'warning'),
    # spatial quadrature not certified (result['status'] = 'approximate'):
    # derivative-vertex loops, equal-time diagrams left on the product grid
    'G6_SPATIAL_DERIVATIVE_LOOP': ('G6', 'warning'),
    'G6_SPATIAL_EQUAL_TIME':     ('G6', 'warning'),
    # a bug in a gate (caught, never raised; a warning in compute_cumulants)
    'INTERNAL_ERROR':            ('G?', 'error'),
}

#: Codes ``Policy.downgrade`` cannot turn into warnings: the run cannot go
#: on past them (the pipeline itself would fail, or produce no number).
_NEVER_DOWNGRADE = frozenset({
    'INTERNAL_ERROR', 'G0_MODEL_INVALID', 'G0_NAMESPACE', 'G0_POLICY_INVALID',
    'G0_PARAMETERS_MISSING', 'G1_ACTION_EVAL', 'G1_EXPAND_FAILED',
    'G2_MF_FAILED', 'G2_NO_SADDLE'})

#: The total-relative accuracy threshold of G6 (see ``_g6_accuracy``): the
#: acceptance tolerance every Phase J route applies to ONE subset
#: (``dbm_integral.COND_RTOL`` for the DBM / DD closed forms, the same value
#: as ``final_integral.NQUAD_EPSREL`` for the quadrature fallback), applied
#: to the cumulant.  ``Question.accuracy``, when declared, replaces it.
_G6_DEFAULT_RTOL = 1e-10
#: item 60: the a posteriori realization error (api._realization.
#: _aposteriori_error) enters the G6 estimate above this value (the
#: realization is refused outright above 1e-2)
_G6_REALIZATION_FLOOR = 1e-6

# Relative tolerances (decided in the D1 brief).
_MARGINAL_RTOL = 1e-8      # |Re lambda| (or |Im omega_pole|) / scale
# |p_i - p_j| / max(|p_i|, |p_j|): the pipeline's pole-cluster linkage
# (api._pole_clusters.CLUSTER_RTOL; a cluster is deformed when its
# cancellation factor exceeds KAPPA_MAX, see _g3_pole_findings)
from api._pole_clusters import CLUSTER_RTOL as _CLOSE_POLE_RTOL  # noqa: E402
_STIFF_RATIO = 1e3         # fast/slow decay-rate ratio flagged as stiff
_EXACT_POLE_BUDGET_SEC = 5  # SIGALRM budget for the exact CF[omega] inverse
_SUPPORTED_DOMAINS = ('positive', 'real', 'integer', 'complex')


# ── user declarations ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    """What the user asks of the model (declared, never inferred).

    ``k``/``max_ell`` set the Taylor order the structure is read at,
    ``max(k + 2*max_ell, 2)`` as in ``compute_cumulants`` (defaults
    ``k=2``, ``max_ell=1`` give order 4).  ``parameters`` is the parameter
    point (default: the model's declared defaults).  ``accuracy`` and
    ``budget`` are recorded for the later cost and routing stages.
    """
    k: Optional[int] = None
    external_fields: Optional[tuple] = None
    max_ell: Optional[int] = None
    accuracy: Optional[float] = None
    budget: Optional[float] = None
    parameters: Optional[dict] = None
    fixed_point_index: Optional[int] = None
    #: the expansion order, when it is not the default ``max(k + 2*max_ell, 2)``
    #: (``compute_cumulants(taylor_order=...)``)
    taylor_order: Optional[int] = None


@dataclass(frozen=True)
class Policy:
    """How permissive the user is (declared, never inferred).

    ``mode``: ``'permissive'`` (default) or ``'strict'`` (warnings become
    errors, except the G5 regime advisories).  Consents: ``allow_taylor_truncation``,
    ``allow_markov_embedding``, ``allow_numeric_fallback``.
    ``interpretation``: the stochastic convention the action is written in
    (``'ito'`` or ``'stratonovich'``); needed when the noise is state
    dependent.  ``assertions``: facts the user vouches for, e.g.
    ``{'saddle_confirmed': True, 'positive_at_saddle': ['nstar']}``;
    the checkable ones are checked, the rest are echoed in the ledger.
    ``downgrade``: error codes the user accepts as warnings (e.g.
    ``('G4_NEGATIVE_VARIANCE',)``); applied after ``strict``, recorded in
    the ledger as consents.  ``INTERNAL_ERROR`` and the codes of
    ``_NEVER_DOWNGRADE`` cannot be downgraded.
    """
    mode: str = 'permissive'
    allow_taylor_truncation: bool = False
    allow_markov_embedding: bool = False
    allow_numeric_fallback: bool = False
    interpretation: Optional[str] = None
    assertions: dict = field(default_factory=dict)
    downgrade: tuple = ()

    def __post_init__(self):
        dg = self.downgrade
        if isinstance(dg, str):
            dg = (dg,)
        object.__setattr__(self, 'downgrade', tuple(dg or ()))


# ── outputs ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One result of a gate.  ``level`` is ``error``, ``warning`` or ``info``."""
    code: str
    level: str
    message: str
    remedy: str = ''
    gate: str = ''


@dataclass(frozen=True)
class LedgerEntry:
    """One consent, convention, assertion or approximation a result for
    the question would depend on.  ``declared``: the user declared it in
    the policy (or the model); ``satisfied``: ``True``/``False`` when the
    analyzer could check it, ``None`` otherwise."""
    kind: str            # 'consent' | 'convention' | 'assertion' | 'approximation'
    name: str
    detail: str
    declared: bool = False
    satisfied: Optional[bool] = None


@dataclass(frozen=True)
class ModelTraits:
    """What ``analyze`` detected.  ``None`` means "not determined" (the
    gate was skipped or failed; a finding says why).

    **The** ``n_poles`` **convention.**  ``n_poles`` is the degree of
    ``Q(omega)``, the least common multiple of the reduced denominators of
    the entries of ``G(omega) = K(omega)^-1`` at the parameter point (exact
    arithmetic in ``QQ[i][omega]``), so a repeated root counts with its
    multiplicity and a pole shared by several entries counts once.  It is
    the number of time-domain modes the residue machinery sums over, which
    is also ``len(prop['pole_vals'])`` for a stable model whose ``Q`` is
    squarefree (``prop['pole_vals']`` keeps only the retarded roots).  It is
    NOT the degree of ``det K`` (pole-zero cancellations are removed) nor the
    sum of the entries' degrees.  ``poles`` lists the distinct roots of
    ``Q`` (all half planes), sorted by ``(Im, Re)``; ``pole_source`` says
    where they came from: ``'exact'`` (``analyze``'s own inverse) or
    ``'pipeline'`` (``compute_cumulants``' tier-1 residue path, the same
    arithmetic; ``prop['pole_analysis']``)."""
    name: str = '<unnamed>'
    # declarations / structure (G0, G1)
    physical_fields: tuple = ()           # declared base names
    n_physical: int = 0                   # scalar components (after populations)
    n_response: int = 0
    n_populations: int = 0
    n_parameters: int = 0
    taylor_order: int = 0
    spatial: bool = False
    spatial_dim: int = 0
    polynomial: Optional[bool] = None
    nonpolynomial_factors: tuple = ()
    vertex_species: tuple = ()            # ((n_response, n_physical), ...) total degree >= 3
    max_vertex_degree: int = 0
    state_dependent_noise: Optional[bool] = None
    undriven_fields: tuple = ()
    # G2
    time_homogeneous: Optional[bool] = None
    mf_solver: Optional[str] = None       # 'dae' (every model)
    n_saddles: Optional[int] = None
    n_stable_saddles: Optional[int] = None
    saddle_index: Optional[int] = None    # index used (pipeline semantics)
    saddle: Optional[tuple] = None        # ((saddle_name, (values...)), ...)
    stable: Optional[bool] = None         # stability of the selected saddle
    stability_source: Optional[str] = None  # 'linearization' | 'poles'
    eigenvalues: tuple = ()               # finite linearization eigenvalues
    n_marginal_modes: Optional[int] = None
    # G3
    propagator_class: Optional[str] = None  # 'rational' | 'delay' | 'non-rational' | 'spatial'
    nonrational_factors: tuple = ()
    #: the delays of the kernel images (exp(-i omega d) factors), by kernel
    delays: tuple = ()
    #: G2-K (stability with the true kernel images, ``api._kernel_stability``):
    #: 'stable' | 'unstable' | 'marginal' | 'neutral' | 'acausal' | 'undetermined'
    kernel_stability: Optional[str] = None
    n_unstable_roots: Optional[int] = None
    n_poles: Optional[int] = None         # with multiplicity (degree of Q(omega))
    poles: tuple = ()                     # distinct roots of Q(omega)
    repeated_poles: Optional[bool] = None
    n_close_pole_pairs: Optional[int] = None
    # near-degenerate / repeated poles (api._pole_clusters): 'harmless',
    # 'resolved', 'unresolved' or 'disabled'; None without close poles
    pole_cluster_status: Optional[str] = None
    pole_cluster_kappa: Optional[float] = None   # largest cancellation factor
    fast_slow_ratio: Optional[float] = None
    pole_rate_min: Optional[float] = None   # min decay rate Im(pole) > 0
    pole_rate_max: Optional[float] = None
    # G5 (D3): tree std of the physical components, the loop-parameter
    # estimate g_eff (definition: ``_g_eff``) with its largest contributions,
    # and the largest ratio of successive Taylor terms at 3 tree std
    tree_std: tuple = ()
    loop_parameter_estimate: Optional[float] = None
    loop_parameter_detail: str = ''
    #: False: ``g_eff`` is a lower bound (terms on fields with no finite tree
    #: variance, e.g. a delta part of G, were left out)
    loop_parameter_complete: Optional[bool] = None
    taylor_radius_ratio: Optional[float] = None
    delta_component: Optional[bool] = None
    causal: Optional[bool] = None
    pole_source: Optional[str] = None     # 'exact' | 'pipeline'
    #: how the pipeline's exact tier computed the residues: 'double'
    #: (certified), 'exact' (50 digits) or 'failed' (numerical tier)
    residue_route: Optional[str] = None
    # G3, spatial models: lambda(q) = eig(M + V q + D q^2) (decay rates)
    spatial_mass_min: Optional[float] = None    # min Re eig(M) (q = 0)
    spatial_rate_min: Optional[float] = None    # min over the q probes
    spatial_diffusion_min: Optional[float] = None  # min Re eig(D)
    # G4
    noise_class: Optional[str] = None     # 'none' | 'white' | 'colored-rational' |
    #                                       'colored-markov-embedded' | 'colored-nonrational'
    noise_gaussian: Optional[bool] = None
    max_noise_cumulant_order: Optional[int] = None
    noise_cross_correlated: Optional[bool] = None
    noise_cross_pairs: tuple = ()
    noise_sources: tuple = ()             # response components with noise
    expand_source: Optional[str] = None   # 'expand' | 'cache (order N)'
    analysis_time_s: float = 0.0


@dataclass(frozen=True)
class ModelReport:
    """The analyzer's result."""
    traits: ModelTraits
    findings: tuple = ()
    ledger: tuple = ()
    route: str = ('unrouted: standalone analysis (compute_cumulants '
                  'records the route; api.routing)')
    question: Optional[Question] = None
    policy: Optional[Policy] = None
    #: compute_cumulants only: the routing record (``api.routing.Route.
    #: as_dict``), the stages the gates ran at, and the G6 run diagnostics
    #: (M8 certificate verdicts, error-accounting counters, accuracy estimate).
    route_detail: dict = field(default_factory=dict)
    stages: tuple = ()
    diagnostics: dict = field(default_factory=dict)

    def ledger_dicts(self) -> list:
        """The ledger as JSON-able dicts (saved with each result)."""
        return [{'kind': e.kind, 'name': e.name, 'detail': e.detail,
                 'declared': e.declared, 'satisfied': e.satisfied}
                for e in self.ledger]

    def findings_dicts(self) -> list:
        return [{'code': f.code, 'level': f.level, 'gate': f.gate,
                 'message': f.message, 'remedy': f.remedy}
                for f in self.findings]

    def by_level(self, level: str) -> tuple:
        return tuple(f for f in self.findings if f.level == level)

    @property
    def errors(self) -> tuple:
        return self.by_level('error')

    @property
    def warnings(self) -> tuple:
        return self.by_level('warning')

    @property
    def ok(self) -> bool:
        """``True`` when no finding is an error."""
        return not self.errors

    @property
    def codes(self) -> tuple:
        return tuple(f.code for f in self.findings)

    def has(self, code: str) -> bool:
        return code in self.codes

    def table(self) -> str:
        """The compact report: one plain-text table (at most 100 columns)
        with the traits that matter, the route, the regime advisories
        (``g_eff``), the pole statistics, the cost when known and the
        findings by level (one line each, message truncated; ``summary()``
        has everything).  ``str(report)`` is the same text."""
        return _report_table(self)

    __str__ = table

    def summary(self) -> str:
        t = self.traits
        lines = [f'Model {t.name!r}: '
                 f'{len(self.errors)} error(s), {len(self.warnings)} warning(s)']
        for fld in fields(ModelTraits):
            lines.append(f'  {fld.name:26s} {getattr(t, fld.name)!r}')
        lines.append('Findings:')
        for f in self.findings:
            lines.append(f'  [{f.level}] {f.code}: {f.message}')
            if f.remedy:
                lines.append(f'      remedy: {f.remedy}')
        lines.append('Ledger:')
        for e in self.ledger:
            lines.append(f'  {e.kind:13s} {e.name}: {e.detail} '
                         f'(declared={e.declared}, satisfied={e.satisfied})')
        lines.append(f'Route: {self.route}')
        for rej in (self.route_detail or {}).get('rejected', []):
            lines.append(f'  rejected {rej["backend"]}: '
                         + '; '.join(rej['reasons']))
        if self.diagnostics:
            lines.append('Run diagnostics (G6):')
            for k, v in self.diagnostics.items():
                lines.append(f'  {k:26s} {v!r}')
        return '\n'.join(lines)


#: Width of ``ModelReport.table()`` (plain text, fits a 100-column terminal).
TABLE_WIDTH = 100


def _num(x, nd=3) -> str:
    """Compact number for the report table (``-`` for ``None``)."""
    if x is None:
        return '-'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x == 0:
        return '0'
    if not math.isfinite(x):
        return str(x)
    return f'{x:.{nd}g}'


def _yn(x) -> str:
    return '-' if x is None else ('yes' if x else 'no')


def _wrap_rows(label: str, text: str, width: int = TABLE_WIDTH) -> list:
    """``label`` column (10 wide) + ``text`` wrapped to ``width``."""
    import textwrap
    pad = ' ' * 11
    lines = textwrap.wrap(text, width=width - 11,
                          break_on_hyphens=False) or ['']
    return [f'{label:<10} {lines[0]}'] + [pad + ln for ln in lines[1:]]


def _report_table(rep) -> str:
    """The compact plain-text report (see ``ModelReport.table``)."""
    t = rep.traits
    n_e, n_w = len(rep.errors), len(rep.warnings)
    n_i = len(rep.by_level('info'))
    status = 'OK' if not n_e else f'{n_e} ERROR(S)'
    rule = '-' * TABLE_WIDTH
    stat = f'[{status}; {n_w} warning(s), {n_i} info]'
    name = str(t.name)
    room = TABLE_WIDTH - len(stat) - len('Model report: ') - 1
    if len(name) > room:
        name = name[:max(room - 3, 1)] + '...'
    head = f'Model report: {name}'
    out = [head + stat.rjust(max(1, TABLE_WIDTH - len(head))), rule]
    q = rep.question
    if q is not None and (q.k is not None or q.max_ell is not None):
        out += _wrap_rows('question', f'k={q.k if q.k is not None else "-"}  '
                          f'max_ell={q.max_ell if q.max_ell is not None else "-"}'
                          + (f'  external={list(q.external_fields)}'
                             if q.external_fields else ''))
    route = rep.route
    out += _wrap_rows('route', route.split(':')[0] if route.startswith('unrouted')
                      else route)
    kind = 'spatial (d=%d)' % t.spatial_dim if t.spatial else 'temporal'
    out += _wrap_rows(
        'model', f'{kind}; {t.n_physical} physical / {t.n_response} response '
        f'component(s); {t.n_parameters} parameter(s); Taylor order '
        f'{t.taylor_order}; time-homogeneous {_yn(t.time_homogeneous)}')
    out += _wrap_rows(
        'vertices', f'polynomial {_yn(t.polynomial)}; species '
        f'{list(t.vertex_species) or "none"}; max degree {t.max_vertex_degree}; '
        f'state-dependent noise {_yn(t.state_dependent_noise)}')
    out += _wrap_rows(
        'noise', f'{t.noise_class or "-"}; Gaussian {_yn(t.noise_gaussian)}; '
        f'cross-correlated {_yn(t.noise_cross_correlated)}')
    out += _wrap_rows(
        'saddle', f'{_num(t.n_saddles)} found, {_num(t.n_stable_saddles)} stable; '
        f'selected #{_num(t.saddle_index)} stable {_yn(t.stable)}; marginal '
        f'modes {_num(t.n_marginal_modes)}')
    if t.spatial:
        out += _wrap_rows(
            'spectrum', f'min decay rate over q {_num(t.spatial_rate_min)}; '
            f'min mass {_num(t.spatial_mass_min)}; min diffusion '
            f'{_num(t.spatial_diffusion_min)}')
    else:
        out += _wrap_rows(
            'poles', f'n={_num(t.n_poles)}; rate min/max {_num(t.pole_rate_min)} / '
            f'{_num(t.pole_rate_max)}; fast/slow {_num(t.fast_slow_ratio)}; '
            f'repeated {_yn(t.repeated_poles)}; close pairs '
            f'{_num(t.n_close_pole_pairs)}'
            + (f' ({t.pole_cluster_status}, kappa {_num(t.pole_cluster_kappa)})'
               if t.pole_cluster_status else '')
            + f'; delta part {_yn(t.delta_component)}')
    ge = t.loop_parameter_estimate
    reg = f'g_eff {"" if t.loop_parameter_complete is not False else ">= "}{_num(ge)}'
    if ge is not None:
        reg += ' (' + ('small' if ge <= _GEFF_WARN else 'LARGE')
        reg += f', warn > {_GEFF_WARN:g})'
    elif not t.spatial and t.taylor_order and t.taylor_order < 4:
        reg += ' (tree level: the vertices are not expanded)'
    if t.taylor_radius_ratio is not None:
        reg += f'; Taylor term ratio at 3 std {_num(t.taylor_radius_ratio)}'
    out += _wrap_rows('regime', reg)
    d = rep.diagnostics or {}
    cost = f'analysis {1e3 * t.analysis_time_s:.1f} ms'
    if d.get('n_diagrams') is not None:
        cost += f'; {d["n_diagrams"]} diagrams'
    if d.get('accuracy_estimate') is not None:
        cost += f'; accuracy estimate {_num(d["accuracy_estimate"])}'
    if d.get('nquad_calls'):
        cost += f'; {d["nquad_calls"]} quadrature fallback call(s)'
    out += _wrap_rows('cost', cost)
    out.append(rule)
    if not rep.findings:
        out += _wrap_rows('findings', 'none')
    else:
        counts = ' / '.join(f'{lv} {len(rep.by_level(lv))}'
                            for lv in LEVELS)
        out += _wrap_rows('findings', counts)
        for f in rep.findings:
            line = f'{f.code}: ' + ' '.join(f.message.split())
            if len(line) > TABLE_WIDTH - 11:
                line = line[:TABLE_WIDTH - 14].rstrip() + '...'
            out.append(f'  {f.level[0].upper():<8} {line}')
    return '\n'.join(out)


# ── internal state ────────────────────────────────────────────────────


class _Ctx:
    """Mutable state shared by the gates for one ``analyze`` call."""

    def __init__(self, model, question, policy):
        self.model = model
        self.q = question
        self.p = policy
        self.findings: list[Finding] = []
        self.ledger: list[LedgerEntry] = []
        self.t: dict[str, Any] = {}       # trait values
        self.fundamental = None
        self.ns = None
        self.n_tilde = 0
        self.ring_names: list[str] = []
        self.action = None
        self.ft = None
        self.K_ft = None
        self.omega = None
        self.num_params = None
        self.stop_numeric = False         # set by fatal G0/G1/G2 findings
        self.stability_known = False
        self.sectors = None               # {(n_resp, n_phys): {exponent: coeff}}
        self.use_cache = True
        self.K_coeff = None               # (1,1)-sector coefficient matrix
        self.poly_fns = ()
        self.wired = False                # True inside compute_cumulants
        self.pipeline_ns = None           # the pipeline's namespace (read only)

    def add(self, code, message, remedy='', level=None):
        """``level``: overrides the code's registered level (S3: pole
        findings are informational when the state-space route takes the
        run)."""
        gate, lvl = FINDING_CODES[code]
        self.findings.append(Finding(code=code, level=level or lvl,
                                     message=message, remedy=remedy,
                                     gate=gate))

    def ledger_add(self, kind, name, detail, declared=False, satisfied=None):
        self.ledger.append(LedgerEntry(kind=kind, name=name, detail=detail,
                                       declared=bool(declared),
                                       satisfied=satisfied))


def _coerce(obj, cls, what):
    if obj is None:
        return cls()
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, dict):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(obj) - known)
        if unknown:
            raise TypeError(f'{what}: unknown key(s) {unknown}; '
                            f'known: {sorted(known)}')
        return cls(**obj)
    raise TypeError(f'{what} must be a {cls.__name__}, a dict or None; '
                    f'got {type(obj).__name__}')


@contextlib.contextmanager
def _quiet():
    """Silence the pipeline helpers' prints and warnings."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _exc_text(exc) -> str:
    s = f'{type(exc).__name__}: {exc}'
    return s if len(s) <= 400 else s[:397] + '...'


#: Warnings that ``mode: strict`` never promotes: the regime advisories (G5)
#: are estimates about the quality of the answer, not defects of the model,
#: and must not stop a run (decision of Oct 13 2026).
_NEVER_PROMOTE = frozenset({'G5_LOOP_PARAMETER', 'G5_TAYLOR_RADIUS'})


def _apply_policy(findings, policy, ctx=None):
    """Final levels: ``strict`` promotes warnings to errors (except the G5
    advisories, ``_NEVER_PROMOTE``), then
    ``policy.downgrade`` turns the named errors into warnings (each one a
    ledger consent).  Returns a new list."""
    out = []
    for f in findings:
        if (policy.mode == 'strict' and f.level == 'warning'
                and f.code not in _NEVER_PROMOTE):
            f = replace(f, level='error',
                        remedy=(f.remedy + ' ' if f.remedy else '')
                        + '[strict policy: warning promoted to error]')
        if (f.level == 'error' and f.code in policy.downgrade
                and f.code not in _NEVER_DOWNGRADE):
            f = replace(f, level='warning',
                        remedy=(f.remedy + ' ' if f.remedy else '')
                        + '[policy.downgrade: error accepted as a warning]')
            if ctx is not None:
                ctx.ledger_add('consent', f'downgrade:{f.code}', f.message,
                               declared=True, satisfied=None)
        out.append(f)
    return out


def _sorted_findings(findings):
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    return sorted(findings, key=lambda f: (order.get(f.level, 9), f.gate))


# ── public entry point ────────────────────────────────────────────────


def analyze(model: dict, question=None, policy=None, *,
            use_cache: bool = True) -> ModelReport:
    """Analyze ``model`` for ``question`` under ``policy``; see the module
    docstring.  Never raises for a bad model (a finding names the cause);
    raises ``TypeError`` only for a malformed ``question``/``policy``
    argument.

    ``use_cache``: read the pipeline's expand cache
    (``saved_models/<model>/expand_taylor<N>.sobj``, any ``N >=`` the
    order needed, fingerprint-checked by its loader) instead of expanding
    the action when one exists.  The cache is never written.  The
    expansion is the only expensive step (minutes for the heaviest tracked
    models when cold)."""
    return analyze_with_context(model, question, policy, use_cache=use_cache)[0]


def analyze_with_context(model: dict, question=None, policy=None, *,
                         use_cache: bool = True):
    """``analyze`` that also returns its internal state (``(report, ctx)``):
    ``ctx.K_ft`` / ``ctx.omega`` (the linear operator with the true kernel
    images), ``ctx.num_params`` (parameters and saddle), ``ctx.ring_names``
    and the noise sectors.  Used by ``api.delay`` (the exact tree of a
    delayed model)."""
    t_start = time.perf_counter()
    question = _coerce(question, Question, 'question')
    policy = _coerce(policy, Policy, 'policy')
    ctx = _Ctx(model, question, policy)
    ctx.use_cache = bool(use_cache)

    # Order: cheapest first.  The DAE saddle search and its linearization
    # need only the namespace, so they run before the expansion (and still
    # diagnose a model whose expansion fails at a bad saddle); the
    # iteration solver needs the expanded theory and runs after it.
    for gate in (_g0_declarations, _g1_structure, _g2_dae_gate, _g1_expand,
                 _g2_iteration_gate, _g1_driven, _g3_propagator, _g4_noise,
                 _g5_regime, _ledger_common):
        try:
            gate(ctx)
        except Exception as exc:        # a gate bug must not escape analyze
            ctx.add('INTERNAL_ERROR',
                    f'internal error in {gate.__name__}: {_exc_text(exc)}',
                    'report this model to the maintainers')
            ctx.stop_numeric = True

    findings = _apply_policy(ctx.findings, policy, ctx)
    findings = _sorted_findings(findings)
    ctx.t['analysis_time_s'] = time.perf_counter() - t_start
    known = {f.name for f in fields(ModelTraits)}
    traits = ModelTraits(**{k: v for k, v in ctx.t.items() if k in known})
    return ModelReport(traits=traits, findings=tuple(findings),
                       ledger=tuple(ctx.ledger), question=question,
                       policy=policy), ctx


# ── G0: declarations ──────────────────────────────────────────────────


def _resolve_parameters(ctx):
    """The parameter point ``compute_cumulants`` uses: the declared defaults
    of every parameter the question does not give, the question's values
    winning (a partial dict is completed); a mean-field parameter's default
    is never merged (it is solved, not supplied).  Same error text."""
    model, q = ctx.model, ctx.q
    pspecs = model.get('parameters', []) or []
    declared = {p.get('name') for p in pspecs}
    given = dict(q.parameters) if q.parameters else {}
    unknown = sorted(set(given) - declared)
    if unknown:
        ctx.add('G0_UNKNOWN_PARAMETER',
                f'question.parameters names undeclared parameter(s): '
                f'{", ".join(unknown)}.',
                'check the spelling against the model\'s parameters')
    fundamental = {p['name']: p['default'] for p in pspecs
                   if p.get('default') is not None and not p.get('mean_field')}
    fundamental.update(given)
    missing = sorted(p['name'] for p in pspecs
                     if p['name'] not in fundamental
                     and not p.get('mean_field'))
    if missing:
        ctx.add('G0_PARAMETERS_MISSING',
                'compute_cumulants: '
                + ('no ``parameters=`` given' if not given else
                   'the ``parameters=`` given do not include them')
                + ', and these model parameters declare no default: '
                + ', '.join(missing) + '.',
                'Pass parameters={...} or declare defaults via '
                'ModelBuilder.parameter(name, default=...).')
        return None
    return fundamental


def _flat_values(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_flat_values(x))
        return out
    return [v]


def _check_domain(ctx, name, domain, value):
    try:
        vals = [complex(x) for x in _flat_values(value)]
    except (TypeError, ValueError):
        ctx.add('G0_DOMAIN_VIOLATION',
                f'parameter {name!r} = {value!r} is not numeric.',
                'give a number (or a list of numbers for an indexed parameter)')
        return
    bad, boundary = [], []
    for z in vals:
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            bad.append(z)
        elif domain in ('positive', 'real', 'integer') and abs(z.imag) > 0:
            bad.append(z)
        elif domain == 'positive' and z.real < 0:
            bad.append(z)
        elif domain == 'positive' and z.real == 0:
            boundary.append(z)
        elif domain == 'integer' and z.real != round(z.real):
            bad.append(z)
    if boundary and not bad:
        ctx.add('G0_DOMAIN_BOUNDARY',
                f'parameter {name!r} is declared \'positive\' (> 0) but '
                f'{len(boundary)} of its value(s) are 0; the symbol is created '
                'with a positivity assumption.',
                "declare domain='real' (or 'nonnegative' once supported), or "
                'drop the zero entries')
    if bad:
        shown = ', '.join(f'{b.real:g}' if b.imag == 0 else str(b)
                          for b in bad[:4])
        ctx.add('G0_DOMAIN_VIOLATION',
                f'parameter {name!r} is declared {domain!r} but takes the '
                f'value(s) {shown}.',
                'fix the value or the declared domain')


def _g0_declarations(ctx):
    model, q, p = ctx.model, ctx.q, ctx.p
    if not isinstance(model, dict):
        ctx.add('G0_MODEL_INVALID',
                f'the model must be a dict from a model builder; got '
                f'{type(model).__name__}.', 'call .build() on the builder')
        ctx.stop_numeric = True
        return
    if p.mode not in ('strict', 'permissive'):
        ctx.add('G0_POLICY_INVALID',
                f"policy.mode must be 'strict' or 'permissive'; got {p.mode!r}.",
                "use Policy(mode='strict') or Policy(mode='permissive')")
    if p.interpretation not in (None, 'ito', 'stratonovich'):
        ctx.add('G0_POLICY_INVALID',
                f"policy.interpretation must be 'ito' or 'stratonovich'; got "
                f'{p.interpretation!r}.', "declare 'ito' or 'stratonovich'")
    bad_dg = [c for c in p.downgrade
              if FINDING_CODES.get(c, ('', ''))[1] != 'error'
              or c in _NEVER_DOWNGRADE]
    if bad_dg:
        ctx.add('G0_POLICY_INVALID',
                f'policy.downgrade names {bad_dg}: only error codes of '
                'FINDING_CODES can be downgraded, and not '
                f'{sorted(_NEVER_DOWNGRADE)}.',
                'name error codes (e.g. G4_NEGATIVE_VARIANCE)')

    name = model.get('name')
    ctx.t['name'] = str(name) if name else '<unnamed>'
    if not name:
        ctx.add('G0_NAME_MISSING', 'the model declares no name.',
                'name the model (the caches are keyed by it)')
    phys = model.get('physical_fields') or []
    ctx.t['physical_fields'] = tuple(f.get('name', '?') for f in phys)
    ctx.t['n_populations'] = (len(model.get('populations') or [])
                              or len((model.get('index_sets') or {}).get('pop', [])))
    pspecs = model.get('parameters', []) or []
    ctx.t['n_parameters'] = len(pspecs)
    if not phys:
        ctx.add('G0_MODEL_INVALID', 'the model declares no physical field.',
                'declare at least one physical_field(...)')
        ctx.stop_numeric = True
    if not callable(model.get('action')):
        ctx.add('G0_MODEL_INVALID', 'the model has no action callable.',
                'set the action (set_action_text / set_action) and build()')
        ctx.stop_numeric = True

    # names: missing / duplicated across fields, parameters, functions, kernels
    seen, dups = {}, []
    for kind, specs in (('physical field', phys), ('parameter', pspecs),
                        ('function', model.get('functions') or []),
                        ('kernel', model.get('kernels') or [])):
        for s in specs:
            nm = s.get('name') if isinstance(s, dict) else None
            if not nm:
                ctx.add('G0_NAME_MISSING', f'a {kind} declares no name.',
                        f'name every {kind}')
                continue
            if nm in seen:
                dups.append(f'{nm!r} ({seen[nm]} and {kind})')
            else:
                seen[nm] = kind
    if dups:
        ctx.add('G0_DUPLICATE_NAME', 'name(s) declared twice: ' + ', '.join(dups) + '.',
                'rename one of each pair')

    # parameter point and domains
    ctx.fundamental = _resolve_parameters(ctx)
    no_domain = []
    for ps in pspecs:
        if ps.get('mean_field'):
            continue
        dom = ps.get('domain')
        if not dom:
            no_domain.append(ps['name'])
            continue
        if dom not in _SUPPORTED_DOMAINS:
            ctx.add('G0_DOMAIN_UNKNOWN',
                    f'parameter {ps["name"]!r} declares the unknown domain {dom!r}.',
                    f'use one of {", ".join(_SUPPORTED_DOMAINS)}')
            continue
        if ctx.fundamental is not None and ps['name'] in ctx.fundamental:
            _check_domain(ctx, ps['name'], dom, ctx.fundamental[ps['name']])
    if no_domain:
        ctx.add('G0_DOMAIN_MISSING',
                'parameter(s) without a declared domain: ' + ', '.join(no_domain) + '.',
                "declare domain='positive' or 'real' so values can be checked")
    if ctx.fundamental is None:
        ctx.stop_numeric = True

    # question
    k, ext, ell = q.k, q.external_fields, q.max_ell
    if k is not None and k < 1:
        ctx.add('G0_QUESTION_INVALID', f'k must be >= 1; got {k}', 'ask for k >= 1')
    if ell is not None and ell < 0:
        ctx.add('G0_QUESTION_INVALID', f'max_ell must be >= 0; got {ell}',
                'ask for max_ell >= 0')
    if ext is not None and k is not None and len(ext) != k:
        ctx.add('G0_QUESTION_INVALID',
                f'external_fields has {len(ext)} entries but k={k}',
                'give one external field per leg')
    k_eff = k if (k is not None and k >= 1) else (len(ext) if ext else 2)
    ell_eff = ell if (ell is not None and ell >= 0) else 1
    ctx.t['taylor_order'] = max(k_eff + 2 * ell_eff, 2)
    if getattr(q, 'taylor_order', None):
        ctx.t['taylor_order'] = int(q.taylor_order)

    if ctx.stop_numeric and not phys:
        return
    # the namespace (cheap: symbols only, no expansion)
    try:
        from engine.core.field_theory import FieldTheory
        with _quiet():
            ft = FieldTheory(model, taylor_order=ctx.t['taylor_order'])
            ns, R, n_tilde = ft._build_namespace()
        ft._ns, ft._R, ft._n_tilde = ns, R, n_tilde
        ctx.ft, ctx.ns, ctx.n_tilde = ft, ns, n_tilde
        ctx.ring_names = list(ns._ring_var_names)
        ctx.t['n_response'] = n_tilde
        ctx.t['n_physical'] = len(ctx.ring_names) - n_tilde
    except Exception as exc:
        ctx.add('G0_NAMESPACE',
                f'the model namespace cannot be built: {_exc_text(exc)}',
                'check the field, parameter, kernel and function declarations')
        ctx.stop_numeric = True
        return

    if ext is not None:
        from api.access import normalize_external_fields
        from engine.diagrams.type_assignment import build_field_index_map
        try:
            ext_int = normalize_external_fields(
                list(ext), naming_convention=model.get('naming_convention'))
        except Exception as exc:
            ctx.add('G0_EXTERNAL_FIELD', str(exc),
                    "give (field, population) pairs, e.g. [('x', 1), ('x', 1)]")
            return
        _, phys_idx = build_field_index_map(ctx.ring_names, n_tilde)
        # spatial models: the second entry labels the leg (a slot), not a
        # population (the spatial drivers read the field name only)
        spatial = bool(model.get('spatial'))
        names = {k_[0] for k_ in phys_idx}
        for f in dict.fromkeys(ext_int):
            if (f[0] in names) if spatial else (f in phys_idx):
                continue
            ctx.add('G0_EXTERNAL_FIELD',
                    f'external field {f} not in phys_idx '
                    f'{sorted(phys_idx.keys())}',
                    'use a physical field of the model (fluctuation name '
                    'or declared natural name) and a 1-based population')


# ── G1: structure ─────────────────────────────────────────────────────


def _op_name(op) -> str:
    if op is None:
        return ''
    n = getattr(op, 'name', None)
    if callable(n):
        try:
            return str(n())
        except Exception:
            pass
    return getattr(op, '__name__', str(op))


def _offending_factors(expr, var_set, *, allow_negative_powers, transparent=()):
    """Subexpressions of ``expr`` that make it non-polynomial (or, with
    ``allow_negative_powers``, non-rational) in the variables ``var_set``.

    Sums and products recurse; integer powers are allowed (negative ones
    only when ``allow_negative_powers``); calls named in ``transparent``
    (``Conv``: linear in its field argument) recurse into their operands;
    anything else that contains one of the variables is offending."""
    from sage.all import SR
    from sage.symbolic.operators import add_vararg, mul_vararg
    out = []

    def has_var(e):
        try:
            return any(e.has(v) for v in var_set)
        except Exception:
            return False

    def walk(e):
        e = SR(e)
        if not has_var(e):
            return
        op = e.operator()
        if op is None:
            return
        if op in (add_vararg, mul_vararg, operator.add, operator.mul):
            for o in e.operands():
                walk(o)
            return
        if op is operator.pow:
            base, ex = e.operands()
            if has_var(ex):
                out.append(e)
                return
            try:
                is_int = bool(SR(ex).is_integer())
            except Exception:
                is_int = False
            if is_int and (allow_negative_powers or SR(ex) >= 0):
                walk(base)
            else:
                out.append(e)
            return
        if _op_name(op) in transparent or re.sub(r'(_\d+)+$', '', _op_name(op)) in transparent:
            for o in e.operands():
                walk(o)
            return
        out.append(e)

    walk(expr)
    uniq, seen = [], set()
    for e in out:
        s = str(e)
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _polynomial_functions(model) -> tuple:
    """Names of the declared functions whose expression is polynomial in
    their arguments (``phi = a*v``): the action calls them as formal
    functions (``phi_1(v)``), which the expansion specializes exactly."""
    from sage.all import SR
    out = []
    for f in model.get('functions') or []:
        args = f.get('args_text') or f.get('args') or []
        expr = f.get('expression_text')
        if not args or not expr or not f.get('name'):
            continue
        try:
            txt = re.sub(r'\b(\w+)\[[^\]]*\]', r'\1', str(expr))
            e = SR(txt.replace('**', '^'))
            if all(e.is_polynomial(SR.var(str(a))) for a in args):
                out.append(str(f['name']))
        except Exception:
            continue
    return tuple(out)


def _g1_structure(ctx):
    if ctx.ns is None or ctx.stop_numeric:
        return
    from sage.all import SR
    model, ns = ctx.model, ctx.ns

    spatial = model.get('spatial') or {}
    ctx.t['spatial'] = bool(spatial)
    ctx.t['spatial_dim'] = int(spatial.get('dim', 0)) if spatial else 0
    if spatial:
        ctx.add('G1_SPATIAL',
                f'spatial model (d = {ctx.t["spatial_dim"]}, boundary '
                f'{(model.get("boundary") or {}).get("mode", "infinite")!r}): '
                'the propagator carries the Laplacian; the temporal pole '
                'analysis (G3) is skipped.',
                'compute_cumulants needs spatial_grid= for this model')

    try:
        with _quiet():
            S = SR(model['action'](ns))
        ctx.action = S
    except Exception as exc:
        ctx.add('G1_ACTION_EVAL', f'the action cannot be evaluated: {_exc_text(exc)}',
                'check the action text against the declared names')
        ctx.stop_numeric = True
        return

    # explicit time dependence (G2's first check: cheap, symbolic)
    t_sym = SR.var('t')
    tdp = model.get('time_dependent_parameters') or []
    explicit_t = bool(S.has(t_sym))
    ctx.t['time_homogeneous'] = not (explicit_t or tdp)
    if explicit_t or tdp:
        what = []
        if explicit_t:
            from sage.symbolic.operators import add_vararg
            Se = S.expand()
            ops = Se.operands() if Se.operator() in (add_vararg, operator.add) else [Se]
            terms = [str(o) for o in ops if SR(o).has(t_sym)]
            what.append('the action depends explicitly on t'
                        + (f' (term(s): {", ".join(terms[:3])})' if terms else ''))
        if tdp:
            what.append(f'time-dependent parameters {list(tdp)}')
        ctx.add('G2_TIME_DEPENDENT',
                '; '.join(what) + '. The stationary pipeline (propagator in '
                'omega, time-translation-invariant Phase J) needs a '
                'time-homogeneous model.',
                'remove the explicit time dependence (e.g. embed a periodic '
                'drive as an extra oscillator field) or use a time-dependent '
                'backend when one exists')
        ctx.stop_numeric = True

    # legacy dict key noise_structure: only white noise is implemented on
    # the numeric routes (time-domain Phase J never applies a temporal noise
    # kernel), so a non-white temporal_type would be computed as white noise
    nstruct = model.get('noise_structure') or {}
    ntype = (nstruct.get('temporal_type', 'white')
             if isinstance(nstruct, dict) else nstruct)
    if ntype != 'white':
        ctx.add('G2_NOISE_STRUCTURE',
                f"noise_structure temporal_type {ntype!r}: the numeric "
                "routes implement white noise only and would silently drop "
                "the noise kernel.",
                'write colored noise as a Markov embedding (an extra '
                'OU-driven field, as in models/ou_quartic_colored.model.py) '
                'or a kernel Conv term, and remove noise_structure')
        ctx.stop_numeric = True

    # polynomial or not (in the fields)
    fvars = list(ns._all_field_sr_vars)
    ctx.poly_fns = _polynomial_functions(model)
    bad = _offending_factors(S, fvars, allow_negative_powers=False,
                             transparent=('Conv',) + ctx.poly_fns)
    ctx.t['polynomial'] = not bad
    ctx.t['nonpolynomial_factors'] = tuple(bad[:6])
    N = ctx.t['taylor_order']
    if bad:
        ctx.add('G1_NONPOLYNOMIAL',
                f'the action is not polynomial in the fields: '
                f'{", ".join(bad[:4])}{" ..." if len(bad) > 4 else ""}. '
                f'It is Taylor-expanded to total order {N}.',
                'exact for every diagram up to the requested (k, max_ell)')
        if not ctx.p.allow_taylor_truncation:
            ctx.add('G1_TAYLOR_CONSENT_MISSING',
                    f'the result depends on a Taylor truncation at order {N} '
                    'that the policy does not allow.',
                    'set policy allow_taylor_truncation=True')
        ctx.ledger_add('consent', 'taylor_truncation',
                       f'non-polynomial terms ({", ".join(bad[:3])}) '
                       f'expanded to total order {N} = max(k + 2*max_ell, 2)',
                       declared=ctx.p.allow_taylor_truncation)



def _g1_expand(ctx):
    """The expansion (or its cache), vertex species, state-dependent noise
    and the linear operator K(omega)."""
    if ctx.ns is None or ctx.action is None or ctx.stop_numeric:
        return
    model = ctx.model
    N = ctx.t['taylor_order']
    # expansion: the pipeline's expand cache if one fits (read only, in
    # its dict form), else a fresh expand() (never saved)
    ctx.t['expand_source'] = 'expand'
    try:
        with _quiet():
            sectors = _read_expand_cache(ctx, N) if getattr(ctx, 'use_cache', False) else None
            if sectors is None:
                ctx.ft.expand()
                sectors = {k: dict(p.dict()) for k, p in ctx.ft.sectors().items()}
        ctx.ns = ctx.ft._ns
    except Exception as exc:
        ctx.add('G1_EXPAND_FAILED', f'FieldTheory.expand failed: {_exc_text(exc)}',
                'see the message; the pipeline stops at the same place')
        ctx.stop_numeric = True
        return
    ctx.sectors = {k: d for k, d in sectors.items() if d}
    _species_findings(ctx, ctx.sectors)

    # the linear operator K(omega)
    _build_k_ft(ctx)


def _species_findings(ctx, sector_keys):
    """Vertex species, state-dependent noise and the interpretation
    convention, from the keys of the non-zero sectors."""
    model = ctx.model
    species = sorted(k for k in sector_keys if k[0] + k[1] >= 3 and k[1] >= 1)
    ctx.t['vertex_species'] = tuple(species)
    ctx.t['max_vertex_degree'] = max((a + b for a, b in species), default=0)
    sdn = [k for k in species if k[0] >= 2]
    ctx.t['state_dependent_noise'] = bool(sdn)
    if sdn:
        ctx.add('G1_STATE_DEPENDENT_NOISE',
                f'state-dependent noise: vertex species {sdn} carry >= 2 '
                'response legs.', '')
        interp = ctx.p.interpretation or model.get('interpretation')
        if not interp:
            ctx.add('G0_CONVENTION_MISSING',
                    'the noise is state dependent but no stochastic '
                    'interpretation is declared; the pipeline reads the action '
                    'as Ito.',
                    "declare policy interpretation='ito' (or build the model "
                    "with SDE(...).interpretation('stratonovich'), which "
                    'converts it to Ito)')
        ctx.ledger_add('convention', 'interpretation',
                       f'state-dependent noise read as '
                       f'{interp or "ito (pipeline default)"}; equal-time '
                       'Theta(0) = 0',
                       declared=bool(interp),
                       satisfied=(None if not interp else interp == 'ito'))
        if interp == 'stratonovich' and not model.get('interpretation'):
            ctx.add('G0_CONVENTION_MISSING',
                    'policy declares Stratonovich, but the model dict carries '
                    'no Ito drift correction; the pipeline reads it as Ito.',
                    "build the model with SDE(...).interpretation('stratonovich')")


def _read_expand_cache(ctx, N):
    """The pipeline's expand-cache bundle for ``N`` in dict form
    ``{(n_resp, n_phys): {exponent: coefficient}}``, or ``None``.

    Runs the validity check of ``api._expand_cache.load_expand`` (the
    shared ``bundle_stale_reason``: ring variables, ``n_tilde``, model-text
    fingerprint, operator-IR signature) but skips its coercion back into the
    polynomial ring, which is the
    slow part (about 140 s for dendritic_quad_soma_sigmoid): the gates
    only read the coefficients.  Never writes."""
    import os
    from sage.all import load as sage_load
    from api import _expand_cache as _ec
    model, ft = ctx.model, ctx.ft
    order = _ec.find_best_cached_order(model, N)
    if order is None:
        return None
    path = _ec.expand_cache_path(model, order)
    if not os.path.isfile(path):
        return None
    try:
        bundle = sage_load(path)
    except Exception:
        return None
    _ec.prepare_for_load(ft)
    if _ec.bundle_stale_reason(model, ft, bundle) is not None:
        return None
    by_tp = bundle['by_tp']
    if order > N:
        by_tp = _ec.downgrade_by_tp_dict(by_tp, N)
    ctx.t['expand_source'] = f'cache (order {order})'
    return {tuple(k): dict(v) for k, v in by_tp.items()}


def _build_k_ft(ctx):
    """K_mat from the (1,1) sector and its Fourier image K_ft(omega), as in
    ``api._propagator.build_propagator`` up to (not including) the
    symbolic inverse, which is the slow part."""
    from sage.all import SR, I, matrix, dirac_delta, diff
    from engine.core.field_theory import fourier_transform
    from api._propagator import _to_kernel
    ft, model = ctx.ft, ctx.model
    ns = ft._ns
    K = _k_coeff_matrix(ctx)
    if K is None:
        return
    nf = ft._n_tilde
    Dt, dD, dDp = ns.Dt, ns.delta_D, ns.delta_Dp
    t_var = SR.var('t')
    omega = SR.var('omega')
    subs_t = {dD: dirac_delta(t_var), dDp: diff(dirac_delta(t_var), t_var)}
    K_ft = [[SR(0)] * nf for _ in range(nf)]
    for i in range(nf):
        for j in range(nf):
            c = SR(K[i][j])
            if c.is_zero():
                continue
            if not (c.has(dD) or c.has(dDp)):
                # first order in Dt (checked by expand): c0*delta + c1*delta'
                # transforms to c0 + c1*I*omega (the convention of
                # engine.core.field_theory.fourier_transform)
                K_ft[i][j] = c.subs({Dt: I * omega})
            else:
                c = _to_kernel(c, Dt, dD, dDp)
                K_ft[i][j] = fourier_transform(SR(c).subs(subs_t), t_var, omega)
    K_ft = matrix(SR, K_ft)
    hook = model.get('kernel_ft_image')
    if hook is not None:
        kft = hook(ns, omega)
        K_ft = K_ft.apply_map(lambda e: SR(e).subs(kft))
    ctx.K_ft, ctx.omega = K_ft, omega


def _k_coeff_matrix(ctx):
    """``K[i][j]``: the (1,1)-sector coefficient of ``r_i x_j`` (response
    row, physical column, in ``ctx.ring_names`` order), or ``None`` after a
    ``G1_UNPAIRED_FIELD`` finding.  Kept on ``ctx.K_coeff``."""
    from sage.all import SR
    S_free = ctx.sectors.get((1, 1), {})
    names = list(ctx.ring_names)
    nf = ctx.n_tilde
    resp, phys = names[:nf], names[nf:]
    pos_row = {names.index(nm): i for i, nm in enumerate(resp)}
    pos_col = {names.index(nm): j for j, nm in enumerate(phys)}
    K = [[SR(0)] * len(phys) for _ in range(nf)]
    for ev, coeff in S_free.items():
        row = col = None
        for idx in range(len(names)):
            if ev[idx] > 0:
                row = pos_row.get(idx, row)
                col = pos_col.get(idx, col)
        if row is not None and col is not None:
            K[row][col] += SR(coeff)
    # structural zeros (``is_trivial_zero``: no simplifier call; the
    # coefficients come out of the expansion in canonical form)
    zero_cols = [phys[j] for j in range(len(phys))
                 if all(SR(K[i][j]).is_trivial_zero() for i in range(nf))]
    zero_rows = [resp[i] for i in range(nf)
                 if all(SR(K[i][j]).is_trivial_zero() for j in range(len(phys)))]
    if len(phys) != nf or zero_cols or zero_rows:
        parts = []
        if len(phys) != nf:
            parts.append(f'{nf} response vs {len(phys)} physical components')
        if zero_cols:
            parts.append(f'physical component(s) {zero_cols} have no linear '
                         'operator (zero column of K)')
        if zero_rows:
            parts.append(f'response component(s) {zero_rows} pair with no '
                         'field (zero row of K)')
        ctx.add('G1_UNPAIRED_FIELD',
                'unpaired field(s): ' + '; '.join(parts) + '. The propagator '
                'K(omega)^-1 is singular.',
                'give every physical field a linear dynamics term '
                '(xt*(Dt + ...)*x) in the action')
        ctx.stop_numeric = True
        return None
    ctx.K_coeff = K
    return K


def _numeric_K(ctx):
    """``omega_value -> numpy K`` with the saddle and parameters
    substituted, or ``None`` when symbols remain."""
    from sage.all import SR
    nf = ctx.K_ft.nrows()
    K_num = ctx.K_ft.apply_map(lambda e: SR(e).subs(ctx.num_params))
    free = set()
    for e in K_num.list():
        free.update(SR(e).variables())
    free.discard(ctx.omega)
    if free:
        return None, sorted(str(s) for s in free), K_num
    fns = [[None] * nf for _ in range(nf)]
    for i in range(nf):
        for j in range(nf):
            fns[i][j] = SR(K_num[i, j])

    def K_at(w):
        M = np.zeros((nf, nf), dtype=complex)
        for i in range(nf):
            for j in range(nf):
                e = fns[i][j]
                M[i, j] = complex(e.subs({ctx.omega: w})) if not e.is_zero() else 0j
        return M
    return K_at, [], K_num


def _noise_matrix(ctx):
    """The (2,0) noise matrix at the saddle (``-2 * coefficient`` of
    ``r_a r_b``, symmetrized) and the response components that carry any
    noise cumulant (any (n >= 2, 0) sector)."""
    memo = getattr(ctx, '_noise_memo', None)
    if memo is not None and memo[0] is ctx.num_params:
        return memo[1]
    from sage.all import SR
    nf = ctx.n_tilde
    names = ctx.ring_names
    sectors = ctx.sectors
    Nmat = np.zeros((nf, nf))
    sources, unresolved = set(), set()
    for key, poly in sectors.items():
        if key[0] < 2 or key[1] != 0:
            continue
        for ev, coeff in poly.items():
            idx = [i for i in range(len(names)) if ev[i] > 0]
            c = SR(coeff).subs(ctx.num_params)
            try:
                cv = complex(c)
            except TypeError:
                cv = None
                unresolved.update(str(s) for s in c.variables())
            if cv is not None and abs(cv) == 0:
                continue
            sources.update(i for i in idx if i < nf)
            if key == (2, 0) and cv is not None:
                if len(idx) == 1:          # c * r_a^2  ->  N_aa = -2c
                    Nmat[idx[0], idx[0]] += -2.0 * cv.real
                else:                      # c * r_a r_b -> N_ab = N_ba = -c
                    a, b = idx
                    Nmat[a, b] += -cv.real
                    Nmat[b, a] += -cv.real
    out = (Nmat, sorted(sources), unresolved)
    ctx._noise_memo = (ctx.num_params, out)
    return out


def _g1_driven(ctx):
    """Undriven fields: a physical component j with G[j, a] == 0 for every
    noisy response component a (G = K^-1 at generic omega)."""
    if ctx.stop_numeric or ctx.K_ft is None or ctx.num_params is None:
        return
    if ctx.t.get('spatial'):
        return
    K_at, free, _ = _numeric_K(ctx)
    if K_at is None:
        return                             # G3 reports the unresolved symbols
    nf = ctx.n_tilde
    _, sources, _ = _noise_matrix(ctx)
    phys = ctx.ring_names[nf:]
    Gs = []
    for w in (0.3711, 1.9137, -0.7319):
        M = K_at(w)
        if not np.all(np.isfinite(M)):
            continue
        s = np.linalg.svd(M, compute_uv=False)
        if s[-1] <= 1e-13 * max(s[0], 1e-300):
            continue
        Gs.append(np.linalg.inv(M))
    if not Gs:
        ctx.add('G1_SINGULAR_PROPAGATOR',
                'K(omega) is singular at every probe frequency: the linear '
                'operator does not determine the fields (singular propagator).',
                'check that every field has its own linear dynamics term')
        ctx.stop_numeric = True
        return
    scale = max(float(np.max(np.abs(G))) for G in Gs)
    undriven = []
    for j, nm in enumerate(phys):
        reach = max(float(np.max(np.abs(G[j, sources]))) if sources else 0.0
                    for G in Gs)
        if reach <= 1e-12 * scale:
            undriven.append(nm)
    _undriven_finding(ctx, undriven, sources)


def _undriven_finding(ctx, undriven, sources):
    ctx.t['undriven_fields'] = tuple(undriven)
    if undriven and sources:
        ctx.add('G1_UNDRIVEN_FIELD',
                f'no noise source reaches {", ".join(undriven)} through the '
                'linear operator (G[field, noisy response] = 0): its '
                'fluctuations vanish at tree level and the noise covariance '
                'restricted to it is singular.',
                'add a noise source that drives it, or drop it as an external '
                'field (it is deterministic at the saddle)')


# ── G2: mean field ────────────────────────────────────────────────────


def _classify_eigs(eigs):
    eigs = np.asarray(eigs, dtype=complex)
    if eigs.size == 0:
        return None, 0
    # relative to the largest eigenvalue, no floor (a floor of 1.0 called
    # every mode of a model in slow time units marginal: audit UNIT-03)
    scale = float(np.max(np.abs(eigs)))
    tol = _MARGINAL_RTOL * scale
    re = eigs.real
    n_marg = int(np.sum(np.abs(re) <= tol))
    if np.any(re > tol):
        return 'unstable', n_marg
    if n_marg:
        return 'marginal', n_marg
    return 'stable', 0


def _g2_ready(ctx):
    if ctx.stop_numeric or ctx.ft is None or ctx.fundamental is None:
        if ctx.t.get('time_homogeneous') is False and 'G2_SKIPPED' not in [
                f.code for f in ctx.findings]:
            ctx.add('G2_SKIPPED', 'mean field, propagator and noise gates '
                    'skipped: the model is not time homogeneous.', '')
        return False
    return True


def _g2_dae_gate(ctx):
    """Every saddle of the ``.equation(...)`` rows and its stability."""
    if not _g2_ready(ctx) or not ctx.model.get('equations'):
        return
    fpi = ctx.q.fixed_point_index
    _g2_dae(ctx, fpi, 0 if fpi is None else int(fpi))
    _check_positivity(ctx)


def _g2_iteration_gate(ctx):
    """Every root of the ``.set_mf_equation`` saddle equations (models
    without ``.equation`` rows; the same multi-root solver)."""
    model, fund = ctx.model, ctx.fundamental
    if not _g2_ready(ctx) or model.get('equations'):
        return
    fpi = ctx.q.fixed_point_index
    ctx.t['mf_solver'] = 'dae'
    try:
        from api._mean_field import solve_mean_field
        with _quiet():
            mf = solve_mean_field(ctx.ft, model, fund, verbose=False,
                                  fixed_point_index=(0 if fpi is None
                                                     else int(fpi)))
    except Exception as exc:
        msg = _exc_text(exc)
        code = ('G2_NO_SADDLE' if 'no MF fixed point found' in msg
                else 'G2_MF_FAILED')
        ctx.add(code, f'the mean-field solve failed: {msg}',
                'check the mean-field equations and the parameter point')
        ctx.stop_numeric = True
        return
    _g2_iteration_record(ctx, mf, fpi)
    _check_positivity(ctx)


def _g2_iteration_record(ctx, mf, fpi):
    """The roots of the ``.set_mf_equation`` saddle equations (``mf``: the
    solver result, its selected root)."""
    ctx.t['mf_solver'] = 'dae'
    ctx.num_params = mf['num_params']
    roots = [r['values'] for r in (mf.get('mf_all_roots') or [])]
    ctx.t['n_saddles'] = len(roots) or 1
    ctx.t['saddle_index'] = mf.get('mf_index_used', 0)
    ctx.t['saddle'] = _saddle_tuple(mf.get('saddle_values') or {})
    ctx.add('G2_SINGLE_SADDLE_SOLVER',
            f'the model declares saddle equations (.set_mf_equation, no '
            f'.equation(...)): {len(roots) or 1} root(s) found'
            + (': ' + '; '.join(_fmt_root(r) for r in roots[:4])
               if len(roots) > 1 else '')
            + '. Their stability is not classified here (no Dt terms); it '
            'is read from the propagator poles (G3).',
            'declare .equation(lhs=..., rhs=...) rows for a linear '
            'stability analysis of every saddle')
    if fpi is not None and not (0 <= fpi < max(len(roots), 1)):
        ctx.add('G2_FIXED_POINT_INDEX',
                f'fixed_point_index={fpi} is out of range: {len(roots)} '
                'saddle(s) to choose from (the pipeline clamps it with a '
                'warning).', f'use 0..{max(len(roots), 1) - 1}')


def _saddle_tuple(values):
    return tuple((str(k), tuple(float(x) for x in (v if isinstance(v, (list, tuple))
                                                   else [v])))
                 for k, v in sorted(values.items()))


def _g2_dae(ctx, fpi, fpi_eff):
    from api._mean_field_dae import solve_mean_field_dae_compat
    model, fund = ctx.model, ctx.fundamental
    ctx.t['mf_solver'] = 'dae'
    m_all = dict(model)
    m_all['stability_analysis'] = False      # every root, same sort order
    try:
        with _quiet():
            mf = solve_mean_field_dae_compat(ctx.ft, m_all, fund,
                                             fixed_point_index=0, verbose=False)
    except Exception as exc:
        msg = _exc_text(exc)
        if 'no MF fixed point found' in msg:
            ctx.add('G2_NO_SADDLE', msg,
                    'check the equations, or the parameter point (no real '
                    'saddle); a degenerate (marginal) root, e.g. a cubic with '
                    'zero linear term, also defeats the Newton solver')
        else:
            ctx.add('G2_MF_FAILED', f'the mean-field solve failed: {msg}',
                    'check the mean-field equations and the parameter point')
        ctx.stop_numeric = True
        return
    roots = [r['values'] for r in mf['mf_all_roots']]
    recs, any_dynamic = _g2_dae_records(ctx, roots)
    stab_on = bool(model.get('stability_analysis', False))
    sel = _g2_dae_classify(ctx, recs, any_dynamic, stab_on, fpi, fpi_eff)
    if sel is None:
        return
    rec = recs[sel]
    # num_params for the selected root (compat built root 0)
    num_params = dict(mf['num_params'])
    for sname, vals in rec['values'].items():
        arr = getattr(ctx.ft._ns, sname, None)
        if arr is None:
            continue
        for i, v in enumerate(vals):
            num_params[arr[i]] = float(v)
    ctx.num_params = num_params


def _g2_dae_records(ctx, roots, pipeline_records=None):
    """``(records, any_dynamic)``: each root with its finite linearization
    eigenvalues and class.  ``pipeline_records``: the pipeline's
    ``mf_all_roots`` (wired mode); their eigenvalues are used when the
    pipeline classified the roots (``stability_analysis`` on)."""
    from api._mean_field_dae import linear_stability
    model, fund = ctx.model, ctx.fundamental
    ctx.t['n_saddles'] = len(roots)
    recs = []
    any_dynamic = False
    for i, r in enumerate(roots):
        pr = pipeline_records[i] if pipeline_records is not None else None
        try:
            if pr is not None and pr.get('stable') is not None:
                eigs = np.asarray(pr.get('eigenvalues_finite', []), dtype=complex)
            else:
                with _quiet():
                    st = linear_stability(model, fund, r)
                eigs = np.asarray(st['eigenvalues_finite'], dtype=complex)
            cls, n_marg = _classify_eigs(eigs)
            any_dynamic = any_dynamic or eigs.size > 0
            recs.append({'values': r, 'eigs': eigs, 'class': cls, 'n_marg': n_marg})
        except Exception as exc:
            recs.append({'values': r, 'eigs': np.array([], dtype=complex),
                         'class': None, 'n_marg': 0, 'error': _exc_text(exc)})
    return recs, any_dynamic


def _g2_dae_classify(ctx, recs, any_dynamic, stab_on, fpi, fpi_eff,
                     selected=None, selectable=None):
    """Saddle selection and stability findings; returns the selected
    record's index, or ``None`` (no selectable saddle).  ``selected`` /
    ``selectable`` (wired mode): the pipeline's choice and its selectable
    set, used instead of re-deriving them."""
    if not any_dynamic:
        ctx.add('G2_STABILITY_UNAVAILABLE',
                'the linearization has no finite eigenvalue (algebraic '
                'equations, or the stability evaluation failed'
                + (f': {recs[0]["error"]}' if recs and recs[0].get('error') else '')
                + '); stability is read from the propagator poles (G3).',
                'add the Dt terms to the .equation(...) rows (lhs) for a '
                'linear stability analysis of every saddle')
        if selectable is None:
            selectable = list(range(len(recs)))
    else:
        ctx.stability_known = True
        stable_idx = [i for i, r in enumerate(recs) if r['class'] == 'stable']
        ctx.t['n_stable_saddles'] = len(stable_idx)
        if selectable is None:
            selectable = stable_idx if stab_on else list(range(len(recs)))
        if len(stable_idx) > 1:
            confirmed = bool(ctx.p.assertions.get('saddle_confirmed'))
            msg = (f'{len(stable_idx)} stable saddles: '
                   + '; '.join(_fmt_root(recs[i]['values']) for i in stable_idx)
                   + '. The expansion is around one well; transitions between '
                   'wells are not in any loop order.')
            if confirmed:
                ctx.findings.append(Finding('G2_MULTISTABLE', 'info', msg, '', 'G2'))
            else:
                ctx.add('G2_MULTISTABLE', msg + ' The choice needs confirmation.',
                        "confirm with policy assertions={'saddle_confirmed': "
                        "True} and pick the well with question fixed_point_index")
            ctx.ledger_add('assertion', 'saddle_confirmed',
                           f'expansion around one of {len(stable_idx)} stable '
                           'saddles (metastable; tunnelling neglected)',
                           declared=confirmed)
    if not selectable:
        all_marg = bool(recs) and all(r['class'] == 'marginal' for r in recs)
        ctx.add('G2_UNSTABLE_SADDLE',
                f'no stable saddle among {len(recs)}: '
                + '; '.join(f'{_fmt_root(r["values"])} eig {_fmt_eigs(r["eigs"])}'
                            for r in recs[:4])
                + (f'; ... ({len(recs) - 4} more)' if len(recs) > 4 else '')
                + ('. Every root is marginal (a zero eigenvalue: a continuum of '
                   'homogeneous saddles, e.g. a massless field)' if all_marg else '')
                + '. The pipeline raises (stability_analysis on) and the '
                'diagrammatic expansion is undefined.',
                'change the parameters to a stable regime'
                + (' (give the field a mass)' if all_marg else ''))
        ctx.t['stable'] = False
        ctx.t['stability_source'] = 'linearization'
        ctx.stop_numeric = True
        return None
    if fpi is not None and not (0 <= fpi < len(selectable)):
        ctx.add('G2_FIXED_POINT_INDEX',
                f'fixed_point_index={fpi} is out of range: '
                f'{len(selectable)} {"stable" if stab_on and any_dynamic else ""}'
                f' saddle(s) to choose from (the pipeline clamps it with a '
                'warning).', f'use 0..{len(selectable) - 1}')
        fpi_eff = max(0, min(fpi_eff, len(selectable) - 1))
    sel = (selected if selected is not None
           else selectable[min(fpi_eff, len(selectable) - 1)])
    ctx.t['saddle_index'] = (selectable.index(sel) if sel in selectable
                             else None)
    rec = recs[sel]
    ctx.t['saddle'] = _saddle_tuple(rec['values'])
    if any_dynamic:
        ctx.t['eigenvalues'] = tuple(complex(z) for z in rec['eigs'])
        ctx.t['n_marginal_modes'] = rec['n_marg']
        ctx.t['stable'] = rec['class'] == 'stable'
        ctx.t['stability_source'] = 'linearization'
        if rec['class'] == 'unstable':
            ctx.add('G2_UNSTABLE_SADDLE',
                    f'the selected saddle {_fmt_root(rec["values"])} is '
                    f'linearly unstable: eigenvalues {_fmt_eigs(rec["eigs"])} '
                    'have Re > 0. Perturbations grow; no stationary state.',
                    'select a stable saddle (fixed_point_index), turn on '
                    'stability_analysis, or change the parameters')
            ctx.stop_numeric = True
        elif rec['class'] == 'marginal' and ctx.t.get('spatial'):
            _spatial_massless(ctx, rec['eigs'])
        elif rec['class'] == 'marginal':
            ctx.add('G2_MARGINAL_MODE',
                    f'the selected saddle {_fmt_root(rec["values"])} has '
                    f'{rec["n_marg"]} marginal mode(s): eigenvalues '
                    f'{_fmt_eigs(rec["eigs"])} with |Re| < '
                    f'{_MARGINAL_RTOL:g} (relative). The correlation time '
                    'diverges and the loop expansion is not controlled.',
                    'move away from the bifurcation point, or add a '
                    'restoring term')
            ctx.stop_numeric = True
    return sel


def _fmt_root(values):
    return '{' + ', '.join(f'{k}=' + ','.join(f'{x:.6g}' for x in v)
                           for k, v in values.items()) + '}'


def _fmt_eigs(eigs):
    return '[' + ', '.join(f'{complex(z).real:.4g}{complex(z).imag:+.4g}i'
                           for z in eigs) + ']'


def _check_positivity(ctx):
    """``assertions={'positive_at_saddle': [names]}``: each saddle (or its
    natural name) must be > 0 at the selected saddle."""
    names = ctx.p.assertions.get('positive_at_saddle')
    if not names or ctx.t.get('saddle') is None:
        return
    if isinstance(names, str):
        names = [names]
    saddle = dict(ctx.t['saddle'])
    nat = ((ctx.model.get('naming_convention') or {})
           .get('mean_field_saddles') or {})
    from api.naming import saddle_for_field
    for nm in names:
        key = (nm if nm in saddle
               else nat.get(nm) or saddle_for_field(ctx.model, nm) or nm)
        vals = saddle.get(key)
        if vals is None:
            ctx.ledger_add('assertion', f'positive_at_saddle:{nm}',
                           f'no saddle named {nm!r} (have {sorted(saddle)})',
                           declared=True, satisfied=None)
            continue
        ok = all(v > 0 for v in vals)
        ctx.ledger_add('assertion', f'positive_at_saddle:{nm}',
                       f'{key} = {list(vals)} > 0', declared=True, satisfied=ok)
        if not ok:
            ctx.add('G2_ASSERTION_VIOLATED',
                    f'asserted {key} > 0 at the saddle, but {key} = {list(vals)}.',
                    'check the parameter point or the assertion')


# ── G3: propagator ────────────────────────────────────────────────────


def _has_kernel_dynamics(K_ft, omega):
    """``True`` when an entry of ``K(omega)`` is more than ``c0 + c1 omega``
    (a kernel image: a denominator in omega, or a delay factor)."""
    from sage.all import SR
    for e in K_ft.list():
        e = SR(e)
        if e.is_trivial_zero() or not e.has(omega):
            continue
        try:
            if e.is_polynomial(omega) and e.degree(omega) <= 1:
                continue
        except Exception:
            pass
        return True
    return False


def _g2_kernel(ctx):
    """G2-K: the saddle's stability from ``det K(s)`` with the true kernel
    images (``api._kernel_stability``).  Runs for every model whose
    ``K(omega)`` holds a kernel (G2's linearization collapses each kernel to
    ``g_hat(0)``).  Delay models: an unstable / marginal / neutral saddle is an
    error (``G2_UNSTABLE_SADDLE`` / ``G2_MARGINAL_MODE`` / ``G2_NEUTRAL_DELAY``).
    Rational-kernel models: a warning (``G2_KERNEL_UNSTABLE``); their poles
    are checked again by G3, which then names the instability
    (``G2_UNSTABLE_SADDLE`` instead of ``G3_ACAUSAL_POLE``)."""
    if (ctx.stop_numeric or ctx.K_ft is None or ctx.num_params is None
            or ctx.t.get('spatial') or ctx.t.get('kernel_stability') is not None):
        return
    if ctx.t.get('propagator_class') not in ('rational', 'delay'):
        return
    if ctx.model.get('_delay_pade'):
        # a Pade copy (api.delay): the true model's G2-K decided already, and
        # the copy's poles are checked by G3
        return
    if not _has_kernel_dynamics(ctx.K_ft, ctx.omega):
        return
    from api._kernel_stability import kernel_stability
    try:
        ks = kernel_stability(ctx.K_ft, ctx.omega, ctx.num_params)
    except Exception as exc:
        ks = None
        why = _exc_text(exc)
    delay = ctx.t.get('propagator_class') == 'delay'
    if ks is None:
        ctx.t['kernel_stability'] = 'undetermined'
        # rational-kernel models keep their (pole-based) checks: info only
        ctx.add('G2_STABILITY_UNAVAILABLE' if delay else 'G2_KERNEL_STABILITY',
                f'kernel-aware stability (G2-K) not decided: {why}.',
                'check the saddle by simulation', level=None if delay else 'info')
        return
    ctx.t['kernel_stability'] = ks.status
    ctx.t['kernel_stability_detail'] = ks.as_dict()
    ctx.t['n_unstable_roots'] = ks.n_rhp
    roots = ', '.join(f'{complex(r):.6g}' for r in ks.roots[:4])
    what = ('the delay(s) ' + ', '.join(ctx.t.get('delays') or ())) if delay \
        else 'the kernel(s)'
    if ks.status == 'stable':
        ctx.add('G2_KERNEL_STABILITY',
                f'det K(s) with the true kernel images has no root with Re s >= 0 '
                f'({ks.method}; {ks.detail}).', 'none needed')
    elif ks.status in ('unstable', 'marginal'):
        if delay:
            ctx.t['stable'] = False
            ctx.t['stability_source'] = 'characteristic_function'
            code = 'G2_UNSTABLE_SADDLE' if ks.status == 'unstable' else 'G2_MARGINAL_MODE'
            ctx.add(code,
                    (f'the saddle is unstable through {what}: det K(s) has '
                     f'{ks.n_rhp} root(s) with Re s > 0 ({ks.method}), at {roots}'
                     if ks.status == 'unstable' else
                     f'marginal mode through {what}: det K(s) has a root on the '
                     f'imaginary axis at {roots} ({ks.method})')
                    + ' (the mean-field linearization, which replaces each kernel by '
                    'its integral, cannot see it).',
                    'change the parameters to a stable regime (e.g. a shorter delay '
                    'or a weaker delayed coupling)')
        else:
            ctx.add('G2_KERNEL_UNSTABLE',
                    f'det K(s) with the true kernel images has {ks.n_rhp} root(s) '
                    f'with Re s >= 0 at {roots} ({ks.method}): an instability '
                    f'through {what} that the mean-field linearization (each kernel '
                    'replaced by its integral) does not see.',
                    'check the parameters; the G3 pole check names it as an '
                    'instability if the propagator carries the mode')
    elif ks.status == 'neutral':
        if delay:
            ctx.add('G2_NEUTRAL_DELAY',
                    f'neutral delay system: {ks.detail}. Its stability cannot be '
                    'decided from a bounded contour and the Pade route does not '
                    'apply.', 'write the delayed term without a time derivative')
        else:
            ctx.add('G2_KERNEL_STABILITY',
                    f'kernel-aware stability (G2-K) not decided: {ks.detail}.', '',
                    level='info')
    elif ks.status == 'undetermined':
        ctx.add('G2_STABILITY_UNAVAILABLE' if delay else 'G2_KERNEL_STABILITY',
                f'kernel-aware stability (G2-K) not decided: {ks.detail}.',
                'check the saddle by simulation', level=None if delay else 'info')
    elif ks.status == 'acausal' and delay:
        # (rational models: the G3 pole check names it)
        ctx.add('G3_ACAUSAL_POLE',
                f'acausal kernel image: {ks.detail}; a retarded kernel has '
                'exp(-i omega d) with d > 0 and poles of its image in the left '
                'half s-plane.',
                'check the kernel images (a delay d must be positive)')


def _g3_propagator(ctx):
    if ctx.K_ft is None:
        return
    if ctx.t.get('spatial'):
        ctx.t['propagator_class'] = 'spatial'
        _g3_spatial(ctx)
        return
    rational = _g3_rational(ctx)
    _g2_kernel(ctx)
    if not rational:
        return
    if ctx.stop_numeric or ctx.num_params is None:
        ctx.add('G3_SKIPPED', 'pole analysis skipped (no usable saddle).', '')
        return
    K_at, free, K_num = _numeric_K(ctx)
    if free:
        ctx.add('G3_UNRESOLVED_SYMBOLS',
                f'K(omega) still contains {free} after the parameters and the '
                'saddle are substituted (a kernel without a frequency image, '
                'or an unsolved saddle).',
                'declare the kernel with freq_image= or time_expr=')
        return
    try:
        res = _exact_poles(K_num, ctx.omega, ctx.n_tilde)
    except Exception as exc:
        res = None
        reason = _exc_text(exc)
    else:
        reason = 'exact inverse exceeded its budget' if res is None else ''
    if res is None:
        ctx.add('G3_POLES_UNAVAILABLE',
                f'exact pole analysis unavailable ({reason}); the pipeline '
                'falls back to its numerical cofactor tier.', '')
        ctx.ledger_add('consent', 'numeric_fallback',
                       'poles and residues from the numerical cofactor tier',
                       declared=ctx.p.allow_numeric_fallback)
        return
    ctx.t['pole_source'] = 'exact'
    if 'statespace_covers_run' not in ctx.t:
        _standalone_route_prediction(ctx)
    _g3_pole_findings(ctx, res)


def _standalone_route_prediction(ctx):
    """Item 60: ``analyze()`` outside ``compute_cumulants`` makes the run's
    state-space route prediction itself (``compute._ss_route_prediction`` on
    the same operator, parameter point and vertex types), so the G3 pole
    findings get the levels the in-run analyzer gives them.  Left unset
    (registered levels, as before) when the prediction cannot be made."""
    try:
        from engine.core.vertices import (extract_vertex_types,
                                          extract_source_types)
        from api.compute import _ss_route_prediction
        prop = {'K_ft': ctx.K_ft, 'omega': ctx.omega,
                'nf': int(ctx.n_tilde),
                'ring_gen_names': list(ctx.ring_names)}
        vt = (list(extract_vertex_types(ctx.ft))
              + list(extract_source_types(ctx.ft)))
        pred = _ss_route_prediction(prop, ctx.num_params, vt)
    except Exception:
        return
    ctx.t['phase_j_engine'] = pred.get('engine')
    ctx.t['statespace_covers_run'] = bool(pred.get('covers_run'))


def _pc_kappa_max():
    from api._pole_clusters import KAPPA_MAX
    return KAPPA_MAX


def _g3_rational(ctx):
    """The propagator class: ``False`` (stop G3) when ``K(omega)`` is not
    rational in omega.  Delay factors ``exp(-i omega d)`` (``api.delay``) are
    recognised: the class is then ``'delay'`` (``compute_cumulants``
    approximates them by Pade, the ledger records it; G2-K decides the
    stability with the true exponential) and the pole analysis is skipped
    (the poles of the exact propagator are infinitely many)."""
    omega = ctx.omega
    bad = []
    for e in ctx.K_ft.list():
        bad.extend(_offending_factors(e, [omega], allow_negative_powers=True))
    bad = list(dict.fromkeys(bad))
    if bad:
        from api.delay import delay_factors, pade_substitute
        delays, rest = [], []
        for e in ctx.K_ft.list():
            delays.extend(delay_factors(e, omega))
            rest.extend(_offending_factors(pade_substitute(e, omega, 1), [omega],
                                           allow_negative_powers=True))
        if delays and not rest:
            ds = list(dict.fromkeys(str(x) for x in delays))
            ctx.t['propagator_class'] = 'delay'
            ctx.t['delays'] = tuple(ds)
            ctx.add('G3_DELAY_KERNEL',
                    f'delay kernel(s): K(omega) carries exp(-i omega d) for d in '
                    f'{ds}. compute_cumulants approximates the delay by its Pade '
                    '(n, n) approximant with an automatic order (loops) and takes '
                    'the tree from the exact delayed covariance (k = 2); the '
                    'stability is decided with the true exponential (G2-K).',
                    "none needed (delay={'order': n} fixes the order)")
            ctx.ledger_add('approximation', 'delay',
                           f'Pade (n, n) of exp(-i omega d), d in {ds}, order chosen '
                           'by doubling (result["delay"]); tree exact for k = 2',
                           declared=True)
            return False
    if bad:
        ctx.t['propagator_class'] = 'non-rational'
        ctx.t['nonrational_factors'] = tuple(bad[:6])
        ctx.add('G3_NONRATIONAL_KERNEL',
                f'G(omega) = K(omega)^-1 is not rational in omega: offending '
                f'factor(s) {", ".join(bad[:4])}. The pole/residue machinery '
                '(and Phase J) needs a rational propagator.',
                'replace the kernel by a rational (Pade or sum-of-exponentials) '
                'approximation, or embed it with auxiliary fields')
        return False
    ctx.t['propagator_class'] = 'rational'
    return True


def _g3_residue_route(ctx, pa):
    """How the pipeline's exact tier computed the residues
    (``prop['pole_analysis']['residue_route']``, ``api._propagator``):
    ``'double'`` (certified), ``'exact'`` (the 50-digit route, with the
    reason) or ``'failed'`` (the numerical cofactor tier)."""
    route = pa.get('residue_route')
    ctx.t['residue_route'] = route
    why = pa.get('residue_route_reason') or ''
    if route == 'exact':
        ctx.add('G3_RESIDUES_HIGH_PRECISION',
                f'poles and residues by the exact 50-digit route: {why}.',
                'none needed')
    elif route == 'failed':
        ctx.ledger_add('consent', 'numeric_fallback',
                       'poles and residues from the numerical cofactor tier '
                       '(the exact tier could not compute them)',
                       declared=ctx.p.allow_numeric_fallback)
        ctx.add('G3_RESIDUES_NUMERICAL_TIER',
                f'the exact tier could not compute the residues ({why}); the '
                'numerical cofactor tier did, without an accuracy guarantee.',
                'check G6 accuracy; rational parameter values help the exact '
                'tier')


def _g3_pole_findings(ctx, res):
    """Pole statistics, causality and (legacy / algebraic models) the
    stability from ``res = (deg Q, distinct poles, squarefree, delta
    entries, improper entries)`` (``_exact_poles`` or the pipeline's
    ``prop['pole_analysis']``)."""
    Q_deg, poles, squarefree, delta, improper = res[:5]
    pcr = res[5] if len(res) > 5 else None
    ctx.t['n_poles'] = Q_deg
    ctx.t['poles'] = tuple(poles)
    ctx.t['repeated_poles'] = not squarefree
    ctx.t['delta_component'] = bool(delta)
    if improper:
        ctx.add('G3_IMPROPER',
                f'improper propagator entries {improper} (deg P > deg Q): '
                'G(omega) grows at large omega; K is not causal/proper.',
                'check the Dt terms (one Dt per field, on its own response)')
    status = (pcr or {}).get('status')
    if pcr:
        ctx.t['pole_cluster_status'] = status
        ctx.t['pole_cluster_kappa'] = float(pcr.get('kappa_max') or 0.0)
    # S3: the state-space route takes the whole run -> the pole findings
    # below are informational (the route uses no poles or residues; the
    # run's census in G6 says whether any subset used the pole route)
    # (strict policy: the levels stay as registered, the strict semantics of
    # main; the prediction can miss subsets that decline on their rows,
    # which G6_SS_PARTIAL reports after the run)
    ss_all = bool(ctx.t.get('statespace_covers_run'))
    ss_note = (' [informational: the state-space Phase J route (the default) '
               'is predicted to take every subset of this run and uses no '
               'poles or residues; G6_PHASE_J_ENGINE has the census]'
               if ss_all else '')
    if ss_all and getattr(ctx.p, 'mode', None) == 'strict':
        ss_all = False                  # the note stays, the level does not
    if not squarefree:
        ctx.add('G3_REPEATED_POLE',
                'Q(omega) has a repeated root (Jordan block): the pole-residue '
                'form Phase J sums cannot represent it (no t exp(lambda t) '
                'term); '
                + ('compute_cumulants averages deformed propagators instead '
                   '(api._pole_clusters).' if status == 'resolved' else
                   'it cannot be resolved here (G3_POLE_CLUSTER_UNRESOLVED).')
                + ss_note,
                'none needed when resolved; otherwise perturb the '
                'parameters off the degeneracy')
    close = []
    for i in range(len(poles)):
        for j in range(i + 1, len(poles)):
            a, b = poles[i], poles[j]
            if abs(a - b) <= _CLOSE_POLE_RTOL * max(abs(a), abs(b), 1e-300):
                close.append((a, b))
    ctx.t['n_close_pole_pairs'] = len(close)
    kap = float((pcr or {}).get('kappa_max') or 0.0)
    if close:
        how = {'harmless': f'cancellation factor {kap:.3g} <= '
                           f'{_pc_kappa_max():g}: the pole-residue sums keep '
                           'their digits',
               'resolved': f'cancellation factor {kap:.3g}: compute_cumulants '
                           'averages deformed propagators (api._pole_clusters)',
               }.get(status, 'not resolved by the pole-cluster deformation ('
                     + str((pcr or {}).get('reason', '')) + ')'
                     if status in ('unresolved', 'disabled') else
                     'residues ~ 1/gap; not analysed')
        ctx.add('G3_CLOSE_POLES',
                f'{len(close)} pole pair(s) closer than {_CLOSE_POLE_RTOL:g} '
                f'relative: {[(complex(a), complex(b)) for a, b in close[:3]]}; '
                f'{how}.' + ss_note,
                'none needed when resolved (see result["pole_clusters"]); '
                'otherwise move the parameters off the degeneracy or '
                'check G6 accuracy')
    if status in ('unresolved', 'disabled'):
        from api._pole_clusters import lost_digits, LOST_DIGITS_WARN
        ml = ctx.q.max_ell if getattr(ctx.q, 'max_ell', None) is not None else 1
        digits = lost_digits(kap if squarefree else float('inf'), ml)
    if status in ('unresolved', 'disabled') and digits > LOST_DIGITS_WARN:
        lost = ('the pole-residue form cannot represent a repeated pole'
                if not math.isfinite(digits) else
                f'about {digits:.0f} digits lost at max_ell={ml} '
                f'(cancellation factor {kap:.3g})')
        ctx.add('G3_POLE_CLUSTER_UNRESOLVED',
                f'a cluster of {"repeated" if not squarefree else "near-degenerate"} '
                f'poles needs the pole-cluster deformation, which is {status}: '
                f'{(pcr or {}).get("reason", "")}; {lost}.' + ss_note,
                'give the exact tier rational parameters / kernel images (it '
                'needs an exact rational propagator), unset '
                'DAEDALUS_POLE_CLUSTERS=0, or (a cluster too close to the '
                'stability boundary) perturb the parameters off the '
                'degeneracy', level='info' if ss_all else None)
    rates = [p.imag for p in poles if p.imag > 0]
    if rates:
        ratio = max(rates) / min(rates)
        ctx.t['fast_slow_ratio'] = float(ratio)
        ctx.t['pole_rate_min'] = float(min(rates))
        ctx.t['pole_rate_max'] = float(max(rates))
        if ratio > _STIFF_RATIO:
            ctx.add('G3_STIFF', f'fast/slow decay-rate ratio {ratio:.3g} > '
                    f'{_STIFF_RATIO:g}: the tau grid must resolve both scales.',
                    'space the tau grid to resolve 1/rate_max and extend it '
                    'to several 1/rate_min; the pole-sum integrators are '
                    'exact for any ratio, only a grid or a quadrature '
                    'fallback sees the stiffness')
    if delta:
        ctx.add('G3_DELTA_COMPONENT',
                f'G(omega) has an instantaneous (delta) part in entries {delta}.', '')
    scale = max([0.0] + [abs(p) for p in poles])     # relative, no unit floor
    tol = _MARGINAL_RTOL * scale
    lower = [p for p in poles if p.imag < -tol]
    marginal = [p for p in poles if abs(p.imag) <= tol]
    ctx.t['causal'] = not lower and not marginal
    if ctx.stability_known:
        if (lower or marginal) and ctx.t.get('kernel_stability') in ('unstable', 'marginal'):
            ctx.t['stable'] = False
            ctx.t['stability_source'] = 'characteristic_function'
            ctx.add('G2_UNSTABLE_SADDLE' if lower else 'G2_MARGINAL_MODE',
                    f'the saddle is {"unstable" if lower else "marginal"} through '
                    f'its kernel(s): propagator pole(s) {lower + marginal} '
                    f'{"below" if lower else "on"} the real axis, the zeros of '
                    'det K(s) found by G2-K (the mean-field linearization, which '
                    'replaces each kernel by its integral, cannot see them).',
                    'change the parameters to a stable regime')
            return
        if lower or marginal:
            ctx.add('G3_ACAUSAL_POLE',
                    f'poles on or below the real axis {lower + marginal} '
                    'although the saddle linearization is stable: the kernel '
                    'images make G(omega) non-retarded.',
                    'check the kernel frequency images (causal kernels have '
                    'poles in the upper half plane)')
        return
    # legacy / algebraic models: stability comes from the poles
    ctx.t['stability_source'] = 'poles'
    ctx.t['n_marginal_modes'] = len(marginal)
    ctx.t['stable'] = not lower and not marginal
    if lower:
        ctx.add('G2_UNSTABLE_SADDLE',
                f'the saddle is unstable: propagator pole(s) {lower} lie in '
                'the lower half plane (growing modes).',
                'change the parameters to a stable regime')
    elif marginal:
        ctx.add('G2_MARGINAL_MODE',
                f'marginal mode(s): propagator pole(s) {marginal} on the real '
                f'axis (|Im| < {_MARGINAL_RTOL:g} relative). The correlation '
                'time diverges.',
                'move away from the bifurcation point, or add a restoring term')


#: Wavenumbers at which the spatial decay rates are probed (G3, spatial).
_SPATIAL_Q_PROBES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _spatial_massless(ctx, eigs=()):
    """A spatial model whose homogeneous (q = 0) mode is marginal: a
    massless field (a warning, not G2_MARGINAL_MODE: q > 0 modes decay)."""
    if 'G3_MASSLESS' in [f.code for f in ctx.findings]:
        return
    ctx.add('G3_MASSLESS',
            'the q = 0 mode is massless (zero decay rate'
            + (f', eigenvalues {_fmt_eigs(eigs)}' if len(eigs) else '')
            + '): the correlation length is infinite; loop integrals are '
            'infrared sensitive (d <= 2) and the spatial loop drivers may '
            'fail on it.',
            'give the field a mass (restoring term) for loop orders >= 1')


def _g3_spatial(ctx):
    """G3 for spatial models (D2).  ``K(omega, q) = i omega + M + V q +
    D q^2`` (``reaction_diffusion_matrices``) at the saddle: the decay rates
    are ``Re eig(M + V q + D q^2)``; a negative one at some probed q (or a
    negative diffusion eigenvalue, which makes every large-q mode grow) means
    no stationary state; zero at q = 0 is a massless field."""
    if ctx.stop_numeric or ctx.num_params is None:
        return
    from sage.all import SR
    ns = ctx.ft._ns
    lap = getattr(ns, 'Laplacian', None)
    if lap is None:
        return
    try:
        from engine.integration.spatial.heat_kernel import (
            reaction_diffusion_matrices)
        q = SR.var('_analyze_q')
        with _quiet():
            Ms, Ds, Vs = reaction_diffusion_matrices(
                ctx.K_ft, ctx.omega, q, lap,
                # the operator IR lowers Dx to the global symbol GradX (the
                # namespace has no GradX attribute)
                getattr(ns, 'GradX', None) or SR.var('GradX'))
        sub = ctx.num_params
        num = [np.array([[complex(SR(m[i, j]).subs(sub))
                          for j in range(m.ncols())] for i in range(m.nrows())])
               for m in (Ms, Ds, Vs)]
    except Exception as exc:
        ctx.add('G3_SPATIAL_FORM',
                'K(omega, q) is not of the reaction-diffusion form i*omega + '
                f'M + V q + D q^2 at the saddle ({_exc_text(exc)}); the '
                'spatial rate checks are skipped.', '')
        return
    M, D, V = num
    # each tolerance relative to the matrix it tests (M: a rate, D: a
    # diffusion constant, M + V q + D q^2: the decay rate at q), no unit floor
    tol_M = _MARGINAL_RTOL * float(np.max(np.abs(M)))
    tol_D = _MARGINAL_RTOL * float(np.max(np.abs(D)))

    def tol_q(qq):
        return _MARGINAL_RTOL * max(float(np.max(np.abs(M))),
                                    float(np.max(np.abs(V))) * qq,
                                    float(np.max(np.abs(D))) * qq * qq)
    mass = float(np.min(np.linalg.eigvals(M).real))
    diff = float(np.min(np.linalg.eigvals(D).real))
    rates = [(qq, float(np.min(np.linalg.eigvals(M + V * qq + D * qq * qq).real)))
             for qq in _SPATIAL_Q_PROBES]
    q_min, r_min = min(rates, key=lambda t: t[1])
    ctx.t['spatial_mass_min'] = mass
    ctx.t['spatial_diffusion_min'] = diff
    ctx.t['spatial_rate_min'] = r_min
    if diff < -tol_D:
        ctx.add('G3_SPATIAL_UNSTABLE',
                f'the diffusion matrix has a negative eigenvalue ({diff:.4g}): '
                'anti-diffusive, every mode of large wavenumber grows '
                '(ill-posed, no stationary state).',
                'check the sign of the Laplacian term (+D*Laplacian on the '
                'right-hand side is diffusive)')
    elif any(r < -tol_q(qq) for qq, r in rates if qq > 0):
        ctx.add('G3_SPATIAL_UNSTABLE',
                f'the mode at wavenumber q = {q_min:g} grows (decay rate '
                f'{r_min:.4g} < 0) although the homogeneous mode is stable: a '
                'finite-wavelength (Turing-type) instability; no stationary '
                'state around this saddle.',
                'change the parameters to a regime where every mode decays')
    if abs(mass) <= tol_M:
        _spatial_massless(ctx)


def _to_qq(x):
    """``QQ(x)``; a float that ``QQ`` refuses through ``SR`` (seen for
    Maxima-produced values such as ``2.00000000000000``) by its exact binary
    value."""
    from sage.all import QQ, RR
    try:
        return QQ(x)
    except (TypeError, ValueError):
        return QQ(RR(x))


def _exact_poles(K_num, omega, nf):
    """Exact inverse of K(omega) in Frac(QQ[i][omega]) (the pipeline's
    tier-1 arithmetic).  Returns ``(deg Q, distinct poles, squarefree,
    delta entries, improper entries, pole-cluster report or None)`` or
    ``None`` on budget expiry."""
    from sage.all import (SR, QQ, CDF, PolynomialRing, CyclotomicField,
                          matrix)
    from api._propagator import _run_with_timeout, _PolynomialPathTimeout
    CF = CyclotomicField(4, 'I_')
    iC = CF.gen()
    PR = PolynomialRing(CF, 'om_an')
    F = PR.fraction_field()

    def to_cf(c):
        c = SR(c)
        return CF(_to_qq(c.real_part())) + CF(_to_qq(c.imag_part())) * iC

    def to_F(e):
        e = SR(e)
        if e.is_zero():
            return F(0)
        n = [to_cf(c) for c in SR(e.numerator()).coefficients(omega, sparse=False)]
        d = [to_cf(c) for c in SR(e.denominator()).coefficients(omega, sparse=False)]
        return F(PR(n)) / F(PR(d))

    KF = matrix(F, [[to_F(K_num[i, j]) for j in range(nf)] for i in range(nf)])
    try:
        G = _run_with_timeout(KF.inverse, _EXACT_POLE_BUDGET_SEC)
    except _PolynomialPathTimeout:
        return None
    Q = None
    delta, improper = [], []
    for i in range(nf):
        for j in range(nf):
            if G[i, j] == 0:
                continue
            d = G[i, j].denominator()
            n = G[i, j].numerator()
            Q = d if Q is None else Q.lcm(d)
            if n.degree() > d.degree():
                improper.append((i, j))
            elif n.degree() == d.degree():
                delta.append((i, j))
    if Q is None:
        return 0, [], True, delta, improper, None
    PRc = PolynomialRing(CDF, 'om_c')
    Qc = PRc([complex(c) for c in Q.coefficients(sparse=False)])
    poles = sorted((complex(r) for r, _ in Qc.roots(CDF)),
                   key=lambda p: (p.imag, p.real)) if Q.degree() > 0 else []
    squarefree = bool(Q.is_squarefree())
    clusters = None
    from api import _pole_clusters as _pc
    if not squarefree or _pc.has_close_pair(poles):
        # what compute_cumulants' exact tier will do with them
        try:
            clusters = _pc.analyze(G, Q, nf).report()
        except Exception as exc:
            clusters = dict(status='unresolved', clusters=[],
                            kappa_max=float('inf'),
                            repeated_pole=not squarefree,
                            reason=f'cluster analysis failed ({_exc_text(exc)})')
    return int(Q.degree()), poles, squarefree, delta, improper, clusters


# ── G4: noise ─────────────────────────────────────────────────────────


def _g4_noise(ctx):
    if ctx.ft is None or getattr(ctx, 'sectors', None) is None:
        return
    ns, model = ctx.ft._ns, ctx.model
    sectors = ctx.sectors
    noise_keys = sorted(k for k in sectors if k[0] >= 2 and k[1] == 0)
    nf = ctx.n_tilde
    names = ctx.ring_names
    if not noise_keys:
        ctx.t['noise_class'] = 'none'
        ctx.t['noise_gaussian'] = None
        ctx.t['max_noise_cumulant_order'] = 0
        ctx.add('G4_NO_NOISE', 'the action has no noise sector (no (n >= 2, 0) '
                'term): every connected correlator vanishes at tree level.',
                'add a noise source')
        return
    max_order = max(k[0] for k in noise_keys)
    ctx.t['max_noise_cumulant_order'] = max_order
    # Non-polynomial dependence on a response field = cumulants of every order
    resp_vars = list(ns._all_field_sr_vars)[:nf]
    resp_np = (_offending_factors(ctx.action, resp_vars, allow_negative_powers=False,
                                  transparent=('Conv',) + getattr(ctx, 'poly_fns', ()))
               if ctx.action is not None else [])
    gaussian = max_order == 2 and not resp_np
    ctx.t['noise_gaussian'] = gaussian
    if not gaussian:
        ctx.add('G4_NON_GAUSSIAN',
                f'non-Gaussian noise: cumulants up to order {max_order} in the '
                f'expanded action'
                + (f' (all orders: {", ".join(resp_np[:2])})' if resp_np else '')
                + '.', '')
    # cross-correlations in the (2,0) sector
    pairs = set()
    if (2, 0) in sectors:
        for ev in sectors[(2, 0)]:
            idx = [i for i in range(len(names)) if ev[i] > 0]
            if len(idx) == 2:
                pairs.add((names[idx[0]], names[idx[1]]))
    ctx.t['noise_cross_correlated'] = bool(pairs)
    ctx.t['noise_cross_pairs'] = tuple(sorted(pairs))
    if pairs:
        ctx.add('G4_CROSS_CORRELATED',
                f'cross-correlated noise between {sorted(pairs)}.', '')
    # colored: Markov-embedded rows, or cumulant kernels (z_kappa symbols)
    cn = model.get('correlated_noises') or {}
    embedded = sorted(k for k in cn if str(k).endswith('_markov_aux'))
    src_ns = ns
    pns = getattr(ctx, 'pipeline_ns', None)
    if pns is not None and hasattr(pns, '_cumulant_kernels'):
        src_ns = pns                          # wired: read, never written
    if not hasattr(src_ns, '_cumulant_kernels'):   # an expand-cache load skips it
        from engine.core.field_theory import _build_cumulant_action
        with _quiet():
            _build_cumulant_action(ns, model)   # the analyzer's own namespace
        src_ns = ns
    kernels = getattr(src_ns, '_cumulant_kernels', {}) or {}
    noise_class = 'white'
    if kernels:
        bad = _cumulant_kernel_nonrational(kernels, ns)
        if bad:
            noise_class = 'colored-nonrational'
            ctx.add('G4_NONRATIONAL_NOISE',
                    f'colored noise kernel(s) with a non-rational spectrum: '
                    f'{bad[:3]}.  Phase J integrates the diagrams that carry '
                    'a smooth noise kernel by its quadrature fallback (no '
                    'closed form; per-subset tolerance NQUAD_EPSREL).',
                    'use an exponential (Lorentzian) kernel, which is embedded '
                    'exactly by an auxiliary OU field (Markov embedding)')
        else:
            noise_class = 'colored-rational'
            ctx.add('G4_COLORED_RATIONAL',
                    f'colored noise with a rational spectrum '
                    f'({len(kernels)} kernel entr{"y" if len(kernels) == 1 else "ies"}); '
                    'Phase J integrates the diagrams that carry a smooth noise '
                    'kernel by its quadrature fallback.', '')
        ctx.ledger_add('consent', 'numeric_fallback',
                       'diagrams with a smooth (colored) noise kernel are '
                       'integrated by the quadrature fallback',
                       declared=ctx.p.allow_numeric_fallback)
    elif embedded:
        noise_class = 'colored-markov-embedded'
        aux = sorted({nm for row in embedded
                      for nm in (cn[row].get('response_legs') or {}).get(2, [])})
        if not ctx.p.allow_markov_embedding:
            ctx.add('G4_MARKOV_CONSENT_MISSING',
                    f'colored noise is Markov-embedded with auxiliary field(s) '
                    f'{aux} (exact for an exponential kernel); the policy does '
                    'not allow the embedding.',
                    'set policy allow_markov_embedding=True')
        ctx.ledger_add('consent', 'markov_embedding',
                       f'colored noise rows {embedded} replaced by white noise '
                       f'on auxiliary field(s) {aux}',
                       declared=ctx.p.allow_markov_embedding)
    ctx.t['noise_class'] = noise_class
    # covariance at the saddle
    if ctx.num_params is None:
        return
    Nmat, sources, unresolved = _noise_matrix(ctx)
    ctx.t['noise_sources'] = tuple(names[i] for i in sources)
    if not unresolved:
        w = np.linalg.eigvalsh(Nmat) if nf else np.array([])
        # relative to the covariance itself (a floor of 1.0 let a small
        # noise with a negative variance through)
        scale = float(np.max(np.abs(Nmat))) if nf else 0.0
        if w.size and w.min() < -1e-10 * scale:
            ctx.add('G4_NEGATIVE_VARIANCE',
                    f'the Gaussian noise covariance at the saddle has a '
                    f'negative eigenvalue ({w.min():.4g}): a negative variance '
                    '(e.g. a negative rate or diffusion constant at the saddle).',
                    'check the sign of the noise term (-D*xt^2) and the saddle')


def _cumulant_kernel_nonrational(kernels, ns=None):
    """The registered smooth cumulant kernels (``ns._cumulant_kernels``:
    ``{key: {'kernel_fn', 'legs', ...}}``) whose spectrum is not rational
    in omega.  The smooth part of ``K(tau)`` (the kernel minus its
    ``dirac_delta(tau)`` part) has a rational Fourier image exactly when it
    is a finite sum of ``poly(tau, |tau|) * exp(linear in tau, |tau|) *
    sin/cos/heaviside(linear in tau)`` terms (:func:`_exp_poly_form`); the
    symbolic transform itself is not used (it needs sign assumptions on
    omega for ``exp(-|tau|/c)``).  Returns the offending entries."""
    from sage.all import SR, dirac_delta
    tau = SR.var('tau')
    bad = []
    for key, entry in kernels.items():
        try:
            if isinstance(entry, dict):
                fn, legs = entry['kernel_fn'], tuple(entry.get('legs') or ())
                try:
                    K = SR(fn(ns, *legs, tau))
                except TypeError:
                    K = SR(fn(ns, legs[0], legs[1], tau))
            elif callable(entry):
                K = SR(entry(tau))
            else:
                K = SR(entry)
            K = K.expand()
            c_loc = K.coefficient(dirac_delta(tau))
            smooth = (K - c_loc * dirac_delta(tau)).expand()
            if not _exp_poly_form(smooth, tau):
                bad.append(str(key))
        except Exception:
            bad.append(f'{key} (kernel not evaluable)')
    return bad


def _exp_poly_form(e, tau):
    """``e`` is a finite sum of ``tau^n |tau|^m exp(a tau + b |tau|)
    trig(c tau)^k heaviside(+-tau)`` terms (coefficients free of tau; trig
    in sin, cos, sinh, cosh), each with an ``exp(... |tau| ...)`` or a
    one-sided (heaviside) factor.  The decay itself (the sign of ``b``) is
    not decided for symbolic coefficients."""
    from sage.all import SR
    from sage.symbolic.operators import add_vararg, mul_vararg
    A = SR.var('_analyze_abs_tau')
    e = SR(e).subs({abs(tau): A})
    if not (e.has(tau) or e.has(A)):
        return True

    def linear(arg):
        try:
            arg = SR(arg).expand()
            return all(arg.degree(v) <= 1 for v in (tau, A)) and \
                arg.is_polynomial(tau) and arg.is_polynomial(A)
        except Exception:
            return False

    trig = ('exp', 'sin', 'cos', 'sinh', 'cosh', 'heaviside', 'unit_step')
    damped = []                       # per term: has an exp(... |tau| ...)

    def factor_ok(f):
        f = SR(f)
        if not (f.has(tau) or f.has(A)):
            return True
        if f.is_symbol():
            return True
        op = f.operator()
        if op is operator.pow:
            base, ex = f.operands()
            if base.is_symbol() and SR(ex).is_integer() and SR(ex) >= 0:
                return True
            if _op_name(base.operator()) in trig and SR(ex).is_integer() \
                    and SR(ex) > 0:
                if _op_name(base.operator()) == 'exp' and base.has(A):
                    damped[-1] = True
                return linear(base.operands()[0])
            return False
        name = _op_name(op)
        if name in trig:
            if name == 'exp' and f.has(A) or name in ('heaviside', 'unit_step'):
                damped[-1] = True
            return linear(f.operands()[0])
        return False

    terms = e.operands() if e.operator() in (add_vararg, operator.add) else [e]
    for t in terms:
        t = SR(t)
        facs = t.operands() if t.operator() in (mul_vararg, operator.mul) else [t]
        damped.append(False)
        if not all(factor_ok(f) for f in facs):
            return False
        # a term with no exp(|tau|) / one-sided factor does not decay in both
        # directions: its spectrum has delta functions (constant, cos(w tau))
        # or does not exist (tau^2), so it is not a rational function
        if not damped[-1]:
            return False
    return True


# ── G5: numerical regime advisories (D3) ──────────────────────────────

#: ``g_eff`` above this is flagged (``G5_LOOP_PARAMETER``).  From the
#: validation table in ``scratch/handoffs/reports/24_report.md`` (measured
#: |one-loop / tree| at tau = 0 on zoo models across regimes): every model
#: whose measured ratio is >= 0.2 has g_eff >= 0.2, and no validated model
#: below 0.2 has a measured ratio above 0.07.
_GEFF_WARN = 0.2
#: fluctuation amplitude (in tree standard deviations) at which the Taylor
#: series of a non-polynomial response is judged, and the largest ratio of
#: successive terms accepted there (``G5_TAYLOR_RADIUS``; see
#: ``_taylor_ratio`` and the validation in reports/24_report.md).
_TAYLOR_NSIGMA = 3.0
_TAYLOR_RATIO_WARN = 0.3
#: most terms of the vertex sectors read for ``g_eff`` (each needs one
#: symbolic substitution); more are skipped with ``G5_ADVISORY_SKIPPED``.
_G5_MAX_TERMS = 240


def _g5_wants(key, poly):
    """Wired mode keeps the coefficient dict of a vertex / state-dependent
    noise sector only when it is small enough for ``g_eff`` to read."""
    nr, npy = key
    if npy < 1 or nr < 1 or (nr == 1 and npy < 2):
        return False
    try:
        return poly.number_of_terms() <= _G5_MAX_TERMS
    except (AttributeError, TypeError):
        return False


def _trapezoid(y, x, axis=-1):
    """``np.trapezoid`` (NumPy >= 2) or ``np.trapz``."""
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return fn(y, x, axis=axis)


def _tree_sigmas(ctx, K_eval, poles):
    """Tree-level equal-time standard deviations of the physical
    components, ``sigma_j^2 = int d omega/2pi [G N G^H]_jj`` with
    ``G = K(omega)^-1`` and ``N`` the Gaussian noise matrix at the saddle,
    by a trapezoid rule in ``ln omega`` over ``[r_min/100, 100 |p|_max]``
    (``r`` the decay rates ``Im p``, ``p`` the poles) plus the flat piece
    below and the ``omega^-2`` tail above (white noise through a first-order
    response; exact there to the trapezoid error, ~1e-4).  An underdamped
    pole (``Re p`` > ``Im p`` > 0, a resonance of width ``Im p`` at ``Re p``)
    adds nodes ``Re p + Im p sinh(t)`` (``t`` spaced 0.3: the peak and its
    Lorentzian tails, for the four sharpest) so the peak is resolved.  ``None``
    when ``G`` is singular on the grid; ``NaN`` for a component without a
    finite variance (``G N G^H`` decays slower than ``omega^-1.3``: a delta
    part of ``G``)."""
    nf = ctx.n_tilde
    Nmat = _noise_matrix(ctx)[0]
    rates = [p.imag for p in poles if p.imag > 0]
    lo = min(rates) / 100.0
    hi = max(abs(p) for p in poles) * 100.0
    n = max(24, int(6 * math.log10(hi / lo)) + 1)
    nodes = [np.logspace(math.log10(lo), math.log10(hi), n)]
    res = sorted({(round(p.real, 9), round(p.imag, 9)) for p in poles
                  if p.imag > 0 and p.real > p.imag},
                 key=lambda t: -t[0] / t[1])[:4]
    for re_p, gam in res:
        t_lo = math.asinh((re_p - lo) / gam)
        t_hi = math.asinh((hi - re_p) / gam)
        nodes.append(re_p + gam * np.sinh(np.linspace(
            -t_lo, t_hi, max(9, int((t_lo + t_hi) / 0.3)))))
    w = np.unique(np.concatenate(nodes))
    w = w[(w >= lo * (1 - 1e-12)) & (w <= hi * (1 + 1e-12))]
    lo, hi = float(w[0]), float(w[-1])
    Kp, Km = K_eval(w), K_eval(-w)
    var = np.zeros(nf)
    no_variance = np.zeros(nf, dtype=bool)
    i_s = int(np.searchsorted(w, hi / 30.0))
    for Ks in (Kp, Km):
        if not np.all(np.isfinite(Ks)):
            return None
        try:
            G = np.linalg.inv(Ks)
        except np.linalg.LinAlgError:
            return None
        S = np.einsum('wij,jk,wlk->wil', G, Nmat, G.conj())
        d = np.real(np.einsum('wjj->wj', S))
        integrand = d * w[:, None]
        # + the segment [0, lo] (flat) and the omega^-2 tail beyond hi
        var += _trapezoid(integrand, np.log(w), axis=0) + d[0] * lo + d[-1] * hi
        # a component whose spectrum does not fall off faster than omega^-1.3
        # (a delta part of G: the spike-count fields) has no finite variance;
        # the cutoff would set the number, so it is reported as NaN
        if 0 < i_s < len(w) - 1:
            a, b = d[i_s], d[-1]
            slope = np.where((a > 0) & (b > 0), np.log(np.maximum(a, 1e-300)
                             / np.maximum(b, 1e-300)) / math.log(w[-1] / w[i_s]),
                             np.inf)
            no_variance |= slope < 1.3
    var = var / (2.0 * math.pi)
    if not np.all(np.isfinite(var)) or np.any(var < 0):
        return None
    std = np.sqrt(var)
    std[no_variance] = np.nan
    return std


def _fast_K(ctx):
    """``omega-array -> (n_w, nf, nf) complex K`` with the parameters and
    the saddle substituted (``fast_callable`` per entry), or ``None`` when
    symbols remain."""
    from sage.all import SR, CDF, fast_callable
    nf = ctx.K_ft.nrows()
    fns = []
    for i in range(nf):
        row = []
        for j in range(nf):
            e = SR(ctx.K_ft[i, j])
            if e.is_trivial_zero():          # no simplifier (is_zero can call Maxima)
                row.append(None)
                continue
            e = e.subs(ctx.num_params)
            if set(e.variables()) - {ctx.omega}:
                return None
            row.append(None if e.is_trivial_zero() else
                       fast_callable(e, vars=[ctx.omega], domain=CDF))
        fns.append(row)

    def K_eval(ws):
        ws = np.atleast_1d(np.asarray(ws, float))
        out = np.zeros((len(ws), nf, nf), dtype=complex)
        for i in range(nf):
            for j in range(nf):
                f = fns[i][j]
                if f is not None:
                    out[:, i, j] = [complex(f(float(x))) for x in ws]
        return out
    return K_eval


def _vertex_terms(ctx):
    """``[(sector, response index, {physical index: power}, |c|)]`` for
    every term of the vertex sectors (response degree >= 1, physical degree
    >= 2, plus the state-dependent noise sectors) with the parameters and
    the saddle substituted, or ``None`` when a sector is too large (or was
    not kept)."""
    from sage.all import SR
    names, nf = ctx.ring_names, ctx.n_tilde
    terms, n_seen = [], 0
    for key in sorted(ctx.sectors):
        nr, npy = key
        sec = ctx.sectors[key]
        if nr < 1 or npy < 1 or (nr == 1 and npy < 2):
            continue
        if not isinstance(sec, dict):
            return None
        n_seen += len(sec)
        if n_seen > _G5_MAX_TERMS:
            return None
        for ev, coeff in sec.items():
            try:
                c = complex(SR(coeff).subs(ctx.num_params))
            except (TypeError, ValueError):
                continue
            if c == 0:
                continue
            resp = [i for i in range(nf) if ev[i] > 0]
            phys = {i - nf: int(ev[i]) for i in range(nf, len(names))
                    if ev[i] > 0}
            terms.append((key, resp[0], phys, abs(c)))
    return terms


def _g_eff(ctx, K_eval, sig):
    """The loop-parameter estimate ``g_eff`` and its largest contributions.

    **Definition.**  ``g_eff`` estimates the size of the one-loop
    correction relative to the tree-level value of a low-order cumulant,
    without running a loop.  Let ``sigma_j`` be the tree standard deviation
    of physical component ``j`` (``_tree_sigmas``).  For a vertex term
    ``c * r_a * prod_j phi_j^(e_j)`` (response ``a``, ``m = sum e_j``
    physical legs) let ``V = |c| prod_j sigma_j^(e_j)`` be its size at the
    fluctuation scale and ``s_a = sum_j |K_aj(0)| sigma_j``
    (omega = 0, or the slowest pole rate where ``K(0)`` vanishes) the size of
    the linear term of equation ``a`` (``K`` is the linear operator, ``(1,1)`` sector).  Then

    - ``m >= 3``: the term renormalises the linear operator by contracting
      one pair of legs (``C(m,2)`` ways): contribution ``C(m,2) V / s_a``;
    - ``m = 2`` (quadratic vertex): the first correction is second order in
      the vertex, contribution ``(2 V / s_a)^2``;
    - state-dependent noise ``c r^n prod phi^e`` (``n >= 2``): the same
      with ``V`` compared to the noise strength ``N_aa`` (``m = 1`` is
      second order, ``(V/N_aa)^2``).

    ``g_eff`` is the SUM of the contributions of all terms.  It is an
    order-of-magnitude estimate (monotone in every coupling and in the
    noise amplitude), not a bound: it reads only the tree variance and the
    couplings at the saddle.  Validated against the measured
    ``|C_1(0)| / |C_0(0)|`` in ``reports/24_report.md``.  For
    ``eps x^3`` with OU noise it is ``3 eps D / mu^2``.  It needs the
    vertices of degree 3-4 in the expansion, so it is not estimated at tree
    level (Taylor order < 4, ``max_ell = 0``).

    A term that touches a field with no finite tree variance (a delta part of
    ``G``, e.g. spike-count fields; ``sigma`` is NaN) cannot be estimated and
    is left out: ``g_eff`` is then a LOWER bound (the third return value counts
    the omitted terms)."""
    names, nf = ctx.ring_names, ctx.n_tilde
    terms = _vertex_terms(ctx)
    if terms is None:
        return None, 'a vertex sector is too large to read cheaply', 0
    if not terms:
        return 0.0, '', 0
    rates = [p.imag for p in ctx.t['poles'] if p.imag > 0]
    K0 = K_eval(np.array([0.0, min(rates)]))
    # fields without a finite variance (sigma NaN) add nothing to the linear
    # reference: a smaller reference only makes g_eff larger (conservative)
    s0 = np.nansum(np.abs(K0[0]) * sig[None, :], axis=1)  # omega = 0
    s1 = np.nansum(np.abs(K0[1]) * sig[None, :], axis=1)  # omega = slowest rate
    s0 = np.where(np.isfinite(s0), s0, 0.0)               # K(0) may be infinite
    s_lin = np.where(s0 > 1e-9 * np.maximum(s1, 1e-300), s0, s1)
    Nmat = _noise_matrix(ctx)[0]
    n_ref = float(np.max(np.abs(np.diag(Nmat)))) if nf else 0.0
    contrib, skipped = [], 0
    for (nr, _npy), a, phys, c in terms:
        V = c * float(np.prod([sig[j] ** e for j, e in phys.items()]))
        m = sum(phys.values())
        if nr == 1:
            ref = float(s_lin[a])
        else:
            na = abs(Nmat[a, a])
            ref = float(na if na > 0 else n_ref)
        if not (math.isfinite(V) and math.isfinite(ref)):
            skipped += 1             # touches a field without a finite variance
            continue
        if not ref > 0:
            continue
        r = V / ref
        if nr == 1 and m == 2:
            g = (2.0 * r) ** 2
        elif m == 1:
            g = r * r
        else:
            g = r * m * (m - 1) / 2.0
        lab = ('r_' + names[a] + ' ' if nr == 1 else
               f'r_{names[a]}^{nr} ') + ' '.join(
                   f'{names[nf + j]}^{e}' if e > 1 else names[nf + j]
                   for j, e in sorted(phys.items()))
        contrib.append((g, lab))
    contrib.sort(key=lambda t: -t[0])
    total = float(sum(g for g, _ in contrib))
    detail = '; '.join(f'{lab}: {g:.3g}' for g, lab in contrib[:3])
    return total, detail, skipped


def _taylor_ratio(ctx, sig, lin):
    """Decay of the Taylor series of the response about the saddle at the
    fluctuation amplitude, per physical component.

    The expansion keeps the terms ``a_m r_a phi_j^m`` (``m`` up to the
    Taylor order minus one).  At amplitude ``s = _TAYLOR_NSIGMA * sigma_j``
    the term sizes are ``t_m = |a_m| s^m``; with ``m0`` the lowest non-linear
    order present, ``q = max_(m > m0) (t_m / t_m0)^(1 / (m - m0))`` is the
    geometric-mean ratio of successive terms (scale free, no normalisation
    by the linear term).  ``q`` estimates ``s / R`` for ``R`` the distance
    from the saddle to the nearest singularity of the response (radius of
    convergence), and truncation at order ``M`` leaves a relative error of
    about ``q^(M - m0 + 1)``.  It UNDERESTIMATES ``s / R`` for a response
    whose nearest singularities are a complex-conjugate pair (a sigmoid:
    by up to 2), which is why the threshold is ``_TAYLOR_RATIO_WARN = 0.3``.
    Returns ``[(component, q, s)]``, largest ``q`` first; components with a
    single non-linear order (nothing to compare) are skipped; ``None`` when a
    sector was not kept (too many terms).

    ``lin[a, j] = |K_aj(0)|`` is the linear response.  A component whose
    highest kept term ``t_M`` is below 1 % of the linear term ``|K_aj(0)| s``
    reports ``q = 0``: the ratio of two small terms is not a convergence
    statement (near an inflection point of a sigmoid the even coefficients
    vanish and ``t_3 / t_2`` is large for a harmless series; found on a
    zoo dendrite model with ``sigma = 0.015`` against a radius of about 3)."""
    from sage.all import SR
    names, nf = ctx.ring_names, ctx.n_tilde
    coef = {}                                    # (a, j) -> {m: |a_m|}
    for (nr, npy), sec in ctx.sectors.items():
        if nr != 1 or npy < 2:
            continue
        if not isinstance(sec, dict):
            return None                          # not kept (too large)
        for ev, c in sec.items():
            phys = [i for i in range(nf, len(names)) if ev[i] > 0]
            resp = [i for i in range(nf) if ev[i] > 0]
            if len(phys) != 1 or len(resp) != 1:
                continue
            try:
                v = abs(complex(SR(c).subs(ctx.num_params)))
            except (TypeError, ValueError):
                continue
            d = coef.setdefault((resp[0], phys[0] - nf), {})
            m = int(ev[phys[0]])
            d[m] = d.get(m, 0.0) + v
    best = {}
    for (a, j), d in coef.items():
        ms = sorted(m for m, v in d.items() if v > 0 and m >= 2)
        if len(ms) < 2:
            continue
        sj = _TAYLOR_NSIGMA * float(sig[j])
        if not (math.isfinite(sj) and sj > 0):
            continue                             # no finite variance / no noise
        t = {m: d[m] * sj ** m for m in ms}
        m0 = ms[0]
        q = max((t[m] / t[m0]) ** (1.0 / (m - m0)) for m in ms[1:])
        if t[ms[-1]] < 0.01 * float(lin[a, j]) * sj:
            q = 0.0                              # the nonlinearity is negligible
        if math.isfinite(q) and (j not in best or q > best[j][0]):
            best[j] = (q, sj)
    return sorted(((names[nf + j], q, sj) for j, (q, sj) in best.items()),
                  key=lambda t: -t[1])


def _g5_regime(ctx):
    """G5, the regime advisories that need the tree variance: the loop
    parameter estimate ``g_eff`` (``_g_eff``) and, for a non-polynomial
    action, the Taylor-series check (``_taylor_ratio``).  Skipped quietly
    for spatial models, non-rational propagators and colored noise that is
    not Markov-embedded (the noise spectrum is not the constant matrix this
    estimate reads)."""
    if (ctx.stop_numeric or ctx.num_params is None or ctx.K_ft is None
            or ctx.sectors is None or ctx.t.get('spatial')
            or ctx.t.get('propagator_class') != 'rational'
            or not ctx.t.get('poles')):
        return
    if ctx.t.get('noise_class') in ('colored-rational', 'colored-nonrational'):
        ctx.add('G5_ADVISORY_SKIPPED',
                'g_eff and the Taylor-radius check read the white-noise '
                'matrix; the noise here is colored with a kernel.', '')
        return
    if not any(p.imag > 0 for p in ctx.t['poles']):
        return
    if (ctx.t.get('taylor_order') or 0) < 4:
        return          # tree level (max_ell = 0): the vertices of degree 3-4 are not expanded
    K_eval = _fast_K(ctx)
    if K_eval is None:
        return
    sig = _tree_sigmas(ctx, K_eval, ctx.t['poles'])
    if sig is None:
        ctx.add('G5_ADVISORY_SKIPPED',
                'the tree variance could not be evaluated (G(omega) singular '
                'or non-finite on the frequency grid).', '')
        return
    if not np.any(sig > 0):
        return                          # no noise reaches the fields (G4_NO_NOISE)
    ctx.t['tree_std'] = tuple(float(s) for s in sig)
    g, detail, skipped = _g_eff(ctx, K_eval, sig)
    if g is None:
        ctx.add('G5_ADVISORY_SKIPPED', f'g_eff not estimated: {detail}.', '')
    else:
        ctx.t['loop_parameter_estimate'] = float(g)
        ctx.t['loop_parameter_detail'] = detail
        ctx.t['loop_parameter_complete'] = not skipped
        if skipped:
            ctx.add('G5_ADVISORY_SKIPPED',
                    f'g_eff is a lower bound: {skipped} vertex term(s) touch '
                    'a field with no finite tree variance (a delta part of '
                    'G(omega), e.g. a spike-count field) and are left out.',
                    '')
        ml = ctx.q.max_ell if getattr(ctx.q, 'max_ell', None) is not None else 1
        if g > _GEFF_WARN and ml >= 1:
            ctx.add('G5_LOOP_PARAMETER',
                    f'loop parameter estimate g_eff {">= " if skipped else "= "}'
                    f'{g:.3g} > '
                    f'{_GEFF_WARN:g} (largest: {detail}): the loop series '
                    f'in the couplings is not rapidly convergent, so the '
                    f'one-loop correction can be comparable to the tree.',
                    'compare max_ell = 0, 1, 2 for convergence, reduce the '
                    'nonlinearity or the noise amplitude, or treat the '
                    'result as qualitative')
    if ctx.t.get('polynomial') is False:
        K0 = K_eval(np.array([0.0, min(p.imag for p in ctx.t['poles']
                                       if p.imag > 0)]))
        k0 = np.abs(K0[0])
        k1 = np.abs(K0[1])
        lin = np.where(np.isfinite(k0) & (k0 > 1e-9 * np.maximum(k1, 1e-300)),
                       k0, k1)
        rad = _taylor_ratio(ctx, sig, lin)
        if rad is None:
            ctx.add('G5_ADVISORY_SKIPPED',
                    'Taylor check not made: a vertex sector is too large to '
                    'read cheaply.', '')
        elif rad:
            comp, q, amp = rad[0]
            ctx.t['taylor_radius_ratio'] = float(q)
            if q > _TAYLOR_RATIO_WARN:
                ctx.add('G5_TAYLOR_RADIUS',
                        f'Taylor truncation (order {ctx.t.get("taylor_order")}) '
                        f'of the response in {comp!r} is unreliable at the '
                        f'fluctuation scale: at {_TAYLOR_NSIGMA:g} tree '
                        f'standard deviations ({amp:.3g}) successive terms '
                        f'shrink only by {q:.2g} per order (> '
                        f'{_TAYLOR_RATIO_WARN:g}); the saddle is within about '
                        f'{amp / q:.3g} of a singularity or domain edge of '
                        'the response function.',
                        'raise the Taylor order (larger k or max_ell), choose '
                        'a regime with weaker fluctuations (smaller noise), '
                        'or replace the response by a polynomial')
        else:
            ctx.add('G5_ADVISORY_SKIPPED',
                    'Taylor check not made: fewer than two non-linear orders '
                    'of the response in the expansion.', '')


# ── ledger entries common to every question ───────────────────────────


def _ledger_common(ctx):
    q, model = ctx.q, ctx.model
    if not isinstance(model, dict):
        return
    if q.max_ell is not None:
        ctx.ledger_add('approximation', 'loop_truncation',
                       f'loop expansion truncated at max_ell = {q.max_ell}',
                       declared=True)
    init = (model.get('initial') or {}).get('mode', 'stationary')
    ctx.ledger_add('approximation', 'stationary_state',
                   f'correlators of the {init} state around the selected '
                   'saddle (time-translation invariance)',
                   declared='initial' in model,
                   satisfied=ctx.t.get('time_homogeneous'))
    if ctx.n_tilde >= 6:
        ctx.ledger_add('consent', 'numeric_fallback',
                       f'nf = {ctx.n_tilde} >= 6: the pipeline skips the '
                       'symbolic inverse; poles and residues are numerical',
                       declared=ctx.p.allow_numeric_fallback)
    known = {'saddle_confirmed', 'positive_at_saddle'}
    for k, v in (ctx.p.assertions or {}).items():
        if k not in known:
            ctx.ledger_add('assertion', str(k), f'declared {v!r} (not checked)',
                           declared=True, satisfied=None)


# ══ wired mode: the gates inside compute_cumulants (D2) ═══════════════
#
# ``compute_cumulants`` builds a :class:`Preflight` and calls one method
# per pipeline stage boundary.  The gates are the ones ``analyze`` runs,
# fed with the PIPELINE's own objects (the expanded sectors, the
# propagator, the mean-field roots, the tier-1 pole record), so they cost
# no second expansion, solve or inverse; they only read those objects.
# A stage whose new findings include an ``error`` raises
# :class:`AnalyzerError`; ``Policy.downgrade`` accepts chosen errors as
# warnings, and a bug inside a gate (``INTERNAL_ERROR``) is recorded as a
# warning, never a stop.  Stage boundaries (cheapest first):
#
#   declarations  before the expansion: G0, G1 structure (own namespace;
#                 the action evaluated symbolically, time dependence)
#   expanded      after the expansion: vertex species, state-dependent
#                 noise, unpaired fields (the (1,1) sector)
#   propagator    after build_propagator: G3 propagator class (rational?)
#   mean_field    after the mean-field solve: G2 (every root, stability,
#                 fixed_point_index), G3 spatial rates, G4 noise, G1
#                 undriven fields (structural reachability in K), ledger
#   poles         after compute_poles_and_residues (Phase J backends):
#                 G3 pole statistics, causality, stability from poles
#   (run)         after the backend: G6 run diagnostics, never raises


class AnalyzerError(ValueError):
    """``compute_cumulants`` stopped at an analyzer gate.  ``report``: the
    :class:`ModelReport` so far; ``findings``: the error findings;
    ``stage``: the gate boundary (a ``ValueError``, so callers that caught
    the pipeline's own ``ValueError`` for these cases still do)."""

    def __init__(self, message, report=None, findings=(), stage=''):
        super().__init__(message)
        self.report = report
        self.findings = tuple(findings)
        self.codes = tuple(f.code for f in self.findings)
        self.stage = stage


def _format_stop(errors, stage, original=None):
    lines = [f'compute_cumulants stopped by the model analyzer at the '
             f'{stage!r} stage (before the diagram enumeration):']
    for f in errors:
        lines.append(f'  [{f.code}] {f.message}')
        if f.remedy:
            lines.append(f'      remedy: {f.remedy}')
    lines.append("  (policy={'downgrade': ['<CODE>']} accepts a chosen error "
                 'as a warning; preflight=False turns the analyzer off.)')
    if original is not None:
        lines.append(f'  pipeline error: {_exc_text(original)}')
    return '\n'.join(lines)


class Preflight:
    """The analyzer wired into ``compute_cumulants`` (see the section
    comment above).  ``taylor_order``: the order the pipeline expands at."""

    def __init__(self, model, question=None, policy=None, *,
                 taylor_order=None):
        self.question = _coerce(question, Question, 'question')
        self.policy = _coerce(policy, Policy, 'policy')
        ctx = _Ctx(model, self.question, self.policy)
        ctx.wired = True
        ctx.use_cache = False
        self.ctx = ctx
        self._seen = 0
        self._taylor = taylor_order
        self.findings: list[Finding] = []
        self.stages: list[str] = []
        self.route = None
        self.diagnostics: dict = {}
        self.time_s = 0.0

    # ── bookkeeping ──
    def _stage(self, name, fns):
        t0 = time.perf_counter()
        ctx = self.ctx
        for fn in fns:
            try:
                fn()
            except Exception as exc:
                ctx.add('INTERNAL_ERROR',
                        f'internal error in the {name!r} gate '
                        f'({getattr(fn, "__name__", "?")}): {_exc_text(exc)}',
                        'report this model to the maintainers; '
                        'preflight=False skips the analyzer')
                ctx.stop_numeric = True
        new = ctx.findings[self._seen:]
        self._seen = len(ctx.findings)
        new = _apply_policy(new, self.policy, ctx)
        # a gate bug never stops a run (not even under strict)
        new = [replace(f, level='warning') if f.code == 'INTERNAL_ERROR'
               else f for f in new]
        self.findings.extend(new)
        self.stages.append(name)
        self.time_s += time.perf_counter() - t0
        return [f for f in new if f.level == 'error']

    def _check(self, name, fns, original=None):
        errs = self._stage(name, fns)
        if errs:
            raise AnalyzerError(_format_stop(errs, name, original),
                                self.report(), errs, name)

    def report(self) -> ModelReport:
        ctx = self.ctx
        known = {f.name for f in fields(ModelTraits)}
        t = dict(ctx.t)
        t['analysis_time_s'] = self.time_s
        traits = ModelTraits(**{k: v for k, v in t.items() if k in known})
        route = self.route
        return ModelReport(
            traits=traits, findings=tuple(_sorted_findings(self.findings)),
            ledger=tuple(ctx.ledger),
            route=route.name if route is not None else 'unrouted',
            question=self.question, policy=self.policy,
            route_detail=route.as_dict() if route is not None else {},
            stages=tuple(self.stages), diagnostics=dict(self.diagnostics))

    def traits_dict(self) -> dict:
        return dict(self.ctx.t)

    # ── stage boundaries ──
    def declarations(self):
        ctx = self.ctx

        def g0():
            _g0_declarations(ctx)
            if self._taylor is not None:
                ctx.t['taylor_order'] = int(self._taylor)

        def g1():
            _g1_structure(ctx)

        self._check('declarations', [g0, g1])

    def expanded(self, ft):
        ctx = self.ctx

        def g1():
            if ctx.ns is None or ctx.action is None or ctx.stop_numeric:
                return
            ctx.pipeline_ns = ft._ns
            ctx.ring_names = list(ft._ns._ring_var_names)
            ctx.n_tilde = int(ft._n_tilde)
            keep = {}
            for key, poly in ft.sectors().items():
                # dict form only where a gate reads coefficients
                keep[key] = (dict(poly.dict())
                             if (key == (1, 1) or (key[0] >= 2 and key[1] == 0)
                                 or _g5_wants(key, poly))
                             else True)
            ctx.sectors = keep
            ctx.t['expand_source'] = 'pipeline'
            _species_findings(ctx, keep)
            _k_coeff_matrix(ctx)

        self._check('expanded', [g1])

    def propagator(self, prop):
        ctx = self.ctx

        def g3():
            if ctx.stop_numeric or ctx.K_coeff is None:
                return
            ctx.K_ft, ctx.omega = prop['K_ft'], prop['omega']
            if ctx.t.get('spatial'):
                ctx.t['propagator_class'] = 'spatial'
                return
            _g3_rational(ctx)

        self._check('propagator', [g3])

    def mean_field(self, mf):
        ctx = self.ctx

        def g2():
            if ctx.stop_numeric or ctx.fundamental is None:
                return
            if ctx.model.get('equations'):
                _w_g2_dae(ctx, mf)
            else:
                _g2_iteration_record(ctx, mf, ctx.q.fixed_point_index)
            _check_positivity(ctx)

        def g3s():
            if ctx.t.get('spatial') and ctx.K_ft is not None:
                _g3_spatial(ctx)

        def g2k():
            _g2_kernel(ctx)

        def g4():
            _g4_noise(ctx)

        def g1_driven():
            _w_g1_driven(ctx)

        def ledger():
            _ledger_common(ctx)

        self._check('mean_field', [g2, g3s, g2k, g4, g1_driven, ledger])

    def failed(self, stage, exc):
        """A pipeline stage (``'expanded'``: the expansion and its saddle
        sanity check; ``'mean_field'``: the solve) raised ``exc``: diagnose
        it with the analyzer's own mean-field solve (DAE models; it needs no
        expansion) and raise ``AnalyzerError`` (from ``exc``) when a gate
        names the cause; otherwise return (the caller re-raises ``exc``
        unchanged)."""
        ctx = self.ctx

        def g2():
            if (ctx.stop_numeric or 'G2' in {f.gate for f in ctx.findings}
                    or not ctx.model.get('equations')):
                return
            _g2_dae_gate(ctx)

        errs = self._stage(stage, [g2])
        if errs:
            raise AnalyzerError(_format_stop(errs, stage, exc),
                                self.report(), errs, stage) from exc

    def mean_field_failed(self, exc):
        self.failed('mean_field', exc)

    def poles(self, prop, ss_route=None):
        """G3 + G5 on the propagator.  ``ss_route`` (S3, ``compute.
        _ss_route_prediction``): when the state-space Phase J route covers
        the run, the pole findings (repeated, close, unresolved clusters)
        are informational: the route uses no poles or residues."""
        ctx = self.ctx
        if ss_route is not None:
            ctx.t['phase_j_engine'] = ss_route.get('engine')
            ctx.t['statespace_covers_run'] = bool(ss_route.get('covers_run'))

        def g3():
            if (ctx.stop_numeric or ctx.num_params is None
                    or ctx.t.get('propagator_class') != 'rational'):
                return
            pa = prop.get('pole_analysis')
            if pa:
                ctx.t['pole_source'] = 'pipeline'
                res = (pa['degree'], list(pa['poles']), pa['squarefree'],
                       list(pa['delta']), list(pa['improper']),
                       prop.get('pole_clusters'))
                _g3_residue_route(ctx, pa)
            else:
                # the pipeline's exact tier failed before Q(omega) (it then
                # used the numerical cofactor tier): the analyzer's own
                # exact inverse (budget _EXACT_POLE_BUDGET_SEC)
                ctx.ledger_add('consent', 'numeric_fallback',
                               'poles and residues from the numerical '
                               'cofactor tier (the exact tier failed)',
                               declared=ctx.p.allow_numeric_fallback)
                res, reason = None, 'the exact inverse failed'
                K_at, free, K_num = _numeric_K(ctx)
                if not free:
                    try:
                        res = _exact_poles(K_num, ctx.omega, ctx.n_tilde)
                    except Exception as exc:
                        reason = _exc_text(exc)
                if res is None:
                    ctx.add('G3_POLES_UNAVAILABLE',
                            'the pipeline did not take its exact (tier-1) pole '
                            f'path and the analyzer\'s exact inverse failed '
                            f'({reason}): the pole statistics (repeated / close '
                            '/ acausal poles) are not available.', '')
                    return
                ctx.t['pole_source'] = 'exact'
                pcr = prop.get('pole_clusters') or res[5]
                if pcr and pcr.get('status') == 'resolved' and not prop.get(
                        'pole_deformation'):
                    # the pipeline used its numerical tier: nothing deformed
                    pcr = dict(pcr, status='unresolved',
                               reason='the pipeline\'s exact tier failed, so '
                                      'no deformation was built')
                res = tuple(res[:5]) + (pcr,)
            _g3_pole_findings(ctx, res)

        def g5():
            _g5_regime(ctx)

        self._check('poles', [g3, g5])

    def after_run(self, result, counters, *, complete, evaluated):
        """G6 (never raises): ``counters``: the Phase J counter deltas of
        the run; ``complete``: no evaluation ran in worker processes;
        ``evaluated``: the run evaluated Phase J on a tau grid."""
        ctx = self.ctx
        self._stage('run', [lambda: _g6(ctx, self, result, counters,
                                         complete, evaluated)])


def _w_g2_dae(ctx, mf):
    """G2 from the pipeline's DAE result: every root it found (with its
    stability annotation when ``stability_analysis`` is on), its choice
    and its selectable set."""
    ctx.t['mf_solver'] = 'dae'
    prs = list(mf.get('mf_all_roots') or [])
    roots = [r['values'] for r in prs]
    recs, any_dynamic = _g2_dae_records(ctx, roots, pipeline_records=prs)
    stab_on = bool(ctx.model.get('stability_analysis', False))
    selectable = ([i for i, r in enumerate(prs) if r.get('stable')]
                  if stab_on else list(range(len(prs))))
    chosen = mf.get('mf_values')
    selected = next((i for i, r in enumerate(roots) if r == chosen), None)
    fpi = ctx.q.fixed_point_index
    _g2_dae_classify(ctx, recs, any_dynamic, stab_on, fpi,
                     0 if fpi is None else int(fpi),
                     selected=selected, selectable=selectable)
    ctx.num_params = mf['num_params']


def _w_g1_driven(ctx):
    """Undriven fields by structural reachability: ``x_j`` is driven when a
    chain ``j -> c -> ... -> a`` of non-zero ``K[i][c]`` (the equation of
    ``x_i`` contains ``x_c``) reaches a noisy response ``a``; with every
    diagonal entry non-zero this is the non-zero pattern of ``G = K^-1``
    (Neumann series of ``D^-1 N``).  It never flags a driven field; an
    exact numerical cancellation inside ``G`` (a structurally reachable
    field the numbers leave undriven) is not detected.  Without a non-zero
    diagonal the numerical probe of ``analyze`` runs instead."""
    if (ctx.stop_numeric or ctx.K_coeff is None or ctx.num_params is None
            or ctx.t.get('spatial')):
        return
    from sage.all import SR
    K, nf = ctx.K_coeff, ctx.n_tilde
    nz = [[not SR(K[i][j]).is_trivial_zero() for j in range(nf)]
          for i in range(nf)]
    if not all(nz[i][i] for i in range(nf)):
        if ctx.K_ft is not None:
            _g1_driven(ctx)
        return
    _, sources, _ = _noise_matrix(ctx)
    reach = set(sources)
    changed = True
    while changed:
        changed = False
        for i in range(nf):
            if i not in reach and any(nz[i][c] and c in reach
                                      for c in range(nf)):
                reach.add(i)
                changed = True
    phys = ctx.ring_names[nf:]
    _undriven_finding(ctx, [phys[j] for j in range(nf) if j not in reach],
                      sources)


# ── G6: after the run ─────────────────────────────────────────────────

#: Phase J counters reported by G6 (``final_integral._RUNTIME_COUNTERS``).
G6_COUNTERS = (
    'divergent_subsets', 'divergent_cancelled', 'dd_ill_conditioned',
    'dd_err_ratio_max', 'dbm_error_ratio_max', 'dbm_declined_ill_conditioned',
    'dd_overflow', 'dd_unsupported', 'nquad_calls',
    'nquad_hardened_quad_flags', 'nquad_hardened_uncertified',
    'dd_polygon', 'dd_interval', 'dd_poset', 'dd_grouped_m0', 'dbm_answered',
    'dd_err_abs_sum', 'dbm_err_abs_sum',
    # S3: the state-space route
    'ss_evals', 'ss_estimated', 'ss_err_abs_sum', 'ss_subsets_built',
    'ss_subsets_declined', 'ss_fallback_subsets', 'ss_fallback_evals')


def _g6_engine(ctx, d, result, c):
    """S3: the Phase J engine census (``result['phase_j_engine']``) as a G6
    finding and a ledger entry; a call-time fallback on the pole route is a
    warning (recorded, never raised)."""
    rep = result.get('phase_j_engine') if isinstance(result, dict) else None
    if not rep or rep.get('engine') != 'statespace':
        return              # DAEDALUS_PHASE_J_ENGINE=poles: the report of main
    n_ss = int(rep.get('ss_subsets') or 0)
    n_dec = int(rep.get('ss_declined') or 0)
    share = rep.get('statespace_share')
    d['phase_j_engine'] = rep.get('engine')
    d['statespace_share'] = share
    pdef = rep.get('pole_deformation')
    detail = (f"engine {rep.get('engine')}: {n_ss} subset(s) on the "
              f"state-space route, {n_dec} on the pole routes"
              + (f" (declined: {rep.get('ss_declined_by_reason')})"
                 if n_dec else '')
              + (f"; pole-cluster deformation {pdef}" if pdef else '')
              + (f" (bypass failed: {rep['bypass_failed']})"
                 if rep.get('bypass_failed') else ''))
    ctx.add('G6_PHASE_J_ENGINE', detail + '.', '')
    nf = int(rep.get('fallback_evals') or 0)
    ill = sorted(k for k in (rep.get('ss_declined_by_reason') or {})
                 if 'ill_conditioned' in str(k))
    if ill or 'ill_conditioned' in str(rep.get('reason') or ''):
        ctx.add('G6_SS_ILL_CONDITIONED',
                'the state-space realization of the propagator is exactly '
                'Hurwitz but not numerically once rounded (graded beyond double '
                'precision, e.g. a high-order rational kernel in a multi-field '
                f'model): {n_dec} subset(s) used the pole routes instead.',
                'see result["phase_j_engine"]; the pole route\'s accuracy is '
                'reported by G6_ACCURACY')
        ctx.ledger_add('assertion', 'realization_conditioning',
                       'the state-space realization failed its a posteriori check '
                       '(api._realization: graded beyond double precision); the pole '
                       f'routes took {n_dec} subset(s)', satisfied=False)
    if rep.get('covers_run') and n_dec:
        pc = result.get('pole_clusters') if isinstance(result, dict) else None
        ctx.add('G6_SS_PARTIAL',
                f'the state-space route was predicted to take every subset '
                f'of this run, but {n_dec} subset(s) declined it '
                f'({rep.get("ss_declined_by_reason")}) and used the pole '
                f'routes: the pole findings of G3, reported as informational, '
                f'apply to them'
                + (f' (pole clusters: {pc.get("status")})' if pc else '')
                + '.',
                'see result["phase_j_engine"] and G6_ACCURACY; '
                'DAEDALUS_PHASE_J_ENGINE=poles gives the pole engine\'s own '
                'report')
    if nf:
        ctx.add('G6_SS_FALLBACK',
                f"{rep.get('fallback_subsets')} routed subset(s) fell back on "
                f"the pole route at call time ({nf} evaluation(s); reasons "
                f"{rep.get('fallback_by_reason')}); those values are the pole "
                "route's.",
                'see result["phase_j_engine"]; a stiff segment beyond the '
                "state-space route's range or a segment over the memory "
                'budget (PHASE_J_MEM_BUDGET_GB) are the usual causes')
    ctx.ledger_add('assertion', 'phase_j_engine', detail,
                   satisfied=(nf == 0))


def _g6(ctx, pf, result, c, complete, evaluated):
    """M8 certificate verdicts, error accounting and the total-relative
    accuracy estimate.

    The estimate.  Every closed-form subset integral (the DD kernels and the
    DBM route) carries a rounding-error estimate ``1e-15 * sum|term|``
    (``dbm_integral.COND_ERR_FACTOR`` times the magnitude of its expansion,
    the routes' own convention); the counters ``dd_err_abs_sum`` /
    ``dbm_err_abs_sum`` add them up over every subset evaluation of the
    run.  Their sum ``E`` estimates ``sum_tau |delta C(tau)|`` over the tau
    grid (triangle inequality; the Wick-permutation compensation only
    divides), so ``R = E / sum_tau |C(tau)|`` estimates the grid-L1
    relative error of the cumulant from these routes.  It is an ESTIMATE,
    not a rigorous bound: the 1e-15 factor is the usual order of the
    rounding of a sum, not its worst case for long sums, and errors already
    in the poles and residues are not in it.  No estimated subset
    (``n_estimated_subsets == 0``) gives ``R = None`` (nothing checked); a
    non-finite cumulant gives ``R = inf``.  Threshold: ``Question.accuracy`` when
    declared, else ``_G6_DEFAULT_RTOL = 1e-10``, the tolerance each route
    accepts ONE subset at (``COND_RTOL``; ``NQUAD_EPSREL`` is the same
    value): above it, cancellation between subsets has amplified their
    rounding beyond what any single subset is allowed.  Subsets answered by
    the quadrature fallback (``nquad_calls``) or as exact zeros carry no
    estimate and are reported, not bounded."""
    d = pf.diagnostics
    d['evaluated'] = bool(evaluated)
    try:
        d['n_diagrams'] = len(result.get('diagrams') or ()) or None
    except TypeError:
        d['n_diagrams'] = None
    d['counters_complete'] = bool(complete)
    d['accuracy_estimate'] = None
    _g6_engine(ctx, d, result, c)
    failed = (result.get('phase_j_incomplete') or {}) \
        if isinstance(result, dict) else {}
    d['n_diagrams_skipped'] = sum(r.get('n_diagrams', 1)
                                  for recs in failed.values() for r in recs)
    if failed:
        first = next(iter(failed.values()))[0]
        ctx.add('G6_DIAGRAMS_SKIPPED',
                f'Phase J could not evaluate {d["n_diagrams_skipped"]} '
                f'diagram(s) (loop orders {sorted(failed)}); they are left '
                f'out of the cumulant (result["status"] = "incomplete"). '
                f'First reason: {str(first.get("reason"))[:200]}',
                'see result["phase_j_incomplete"]; the numbers are '
                'incomplete until every diagram evaluates')
    # spatial quadrature accuracy (api.compute._spatial_accuracy): a result
    # the integrator cannot certify is 'approximate', named here
    for rec in ((result.get('spatial_accuracy') or [])
                if isinstance(result, dict) else []):
        sp = rec.get('coarse_spread')
        code = ('G6_SPATIAL_DERIVATIVE_LOOP'
                if rec.get('reason') == 'derivative_vertex_loop'
                else 'G6_SPATIAL_EQUAL_TIME')
        ctx.add(code,
                f"{rec.get('note')}"
                + (f'; 0.7x-coarser grid spread {sp:.2e} of max|C|'
                   if sp is not None else '')
                + ' (result["status"] = "approximate").',
                'refine SPATIAL_GRID_NT / SPATIAL_GRID_NS to watch it converge')
        ctx.ledger_add('assertion', 'spatial_accuracy',
                       f"{rec.get('reason')}: {rec.get('note')}",
                       satisfied=False)
    if not evaluated:
        ctx.add('G6_NOT_EVALUATED',
                'no Phase J tau-grid evaluation ran inside compute_cumulants '
                '(spatial backend, or k >= 3 where the integrals run when the '
                'returned callables are evaluated): no run diagnostics.', '')
        return
    for k in G6_COUNTERS:
        d[k] = c.get(k, 0)
    if not complete:
        ctx.add('G6_COUNTERS_INCOMPLETE',
                'the tau grid was evaluated in worker processes '
                '(parallel=True), whose Phase J counters do not reach this '
                'process: the run diagnostics cover the parent process only.',
                'pass parallel=False for complete run diagnostics')
    n_div = int(c.get('divergent_subsets', 0))
    if n_div:
        ctx.add('G6_DIVERGENT_SUBSETS',
                f'{n_div} subset integral(s) fail the convergence certificate '
                '(an open direction that does not decay); their boxed fallback '
                'values are in the cumulant (divergence policy).',
                "rerun with DAEDALUS_PHASE_J_ON_DIVERGENCE=warn to name them, "
                "or 'raise' to stop; a divergent subset usually means a "
                'marginal or unstable mode')
    if c.get('divergent_cancelled', 0):
        ctx.add('G6_DIVERGENT_CANCELLED',
                f'{c["divergent_cancelled"]} pole-tuple group(s) fail the '
                'certificate with an exactly cancelling coefficient (dropped, '
                'not divergent).', '')
    n_ill = int(c.get('dd_ill_conditioned', 0)) + int(
        c.get('dbm_declined_ill_conditioned', 0))
    if n_ill:
        ctx.add('G6_ILL_CONDITIONED',
                f'{c.get("dd_ill_conditioned", 0)} closed-form subset sum(s) '
                f'above their own acceptance (worst ratio '
                f'{float(c.get("dd_err_ratio_max", 0.0)):.3g}) and '
                f'{c.get("dbm_declined_ill_conditioned", 0)} DBM call(s) '
                'declined as ill-conditioned; whether they matter for the '
                'cumulant is the accuracy estimate (G6_ACCURACY).', '')
    n_q = int(c.get('nquad_calls', 0))
    if n_q:
        ctx.add('G6_NQUAD_FALLBACK',
                f'{n_q} subset evaluation(s) went to the quadrature fallback '
                f'(per-subset tolerance NQUAD_EPSREL; '
                f'{c.get("nquad_hardened_quad_flags", 0)} flagged quad '
                'call(s)); they are not in the accuracy estimate.', '')
    C = result.get('C_tau')
    # S3: the state-space route's a-posteriori estimates (``statespace``
    # module docstring: refinement residuals + the Taylor tolerance per
    # step, on the final contraction's magnitude) join the closed-form
    # routes' sums
    E = (float(c.get('dd_err_abs_sum', 0.0)) + float(c.get('dbm_err_abs_sum', 0.0))
         + float(c.get('ss_err_abs_sum', 0.0)))
    # item 60: the per-subset estimates above assume the ROUNDED realization
    # represents G; a graded / not numerically Hurwitz one was checked a
    # posteriori (api._realization: refused above 1e-2) and its measured
    # relative error enters the estimate, so a realization that passes the
    # refusal but is off by more than the threshold does not pass G6
    rc = (result.get('phase_j_engine') or {}).get('realization_check') \
        if isinstance(result, dict) else None
    real_rel = 0.0
    if (rc and rc.get('checked') and rc.get('err_rel') is not None
            and int(c.get('ss_estimated', 0))):
        d['realization_err_rel'] = float(rc['err_rel'])
        # below the floor the measurement is the frequency quadrature's own
        # noise on strongly graded spectra (~1e-6 at a 1e6 spread, the
        # Lyapunov solution measured exact to 5e-13), not the realization's
        if float(rc['err_rel']) > _G6_REALIZATION_FLOOR:
            real_rel = float(rc['err_rel'])
    pc = result.get('pole_clusters') if isinstance(result, dict) else None
    if pc and pc.get('status') == 'resolved' and pc.get('node_runs'):
        # the counters add up every node run of the pole-cluster deformation
        # while C is their weighted mean (weights summing to 1)
        E /= float(pc['node_runs'])
    n_est = sum(int(c.get(k, 0)) for k in ('dd_polygon', 'dd_interval',
                                           'dd_poset', 'dbm_answered',
                                           'ss_estimated'))
    Ca = np.asarray(C, dtype=complex) if C is not None else np.zeros(0, complex)
    S = float(np.sum(np.abs(Ca)))
    if not complete:
        R = None                              # the estimates stayed in the workers
    elif Ca.size and not np.all(np.isfinite(Ca)):
        R = float('inf')                      # a NaN / inf cumulant
    elif n_est == 0 and E == 0:
        R = None                              # nothing was estimated
    elif S > 0:
        R = E / S
    else:
        R = 0.0 if E == 0 else float('inf')
    if real_rel and complete and R is not None:
        R = R + real_rel
    # Near-degenerate / repeated poles (api._pole_clusters): a resolved
    # cluster adds the deformation's own estimate (grid-L1, Fourier tail of
    # the node values); an unresolved one its lost-digit estimate.
    if pc and complete:
        if pc.get('status') == 'resolved' and pc.get('accuracy_l1') is not None:
            R = (R or 0.0) + float(pc['accuracy_l1'])
        elif pc.get('status') in ('unresolved', 'disabled'):
            lost = pc.get('lost_digits')
            if lost is not None:
                R = max(R or 0.0, float('inf') if not math.isfinite(lost)
                        else 10.0 ** (lost - 16))
        d['pole_cluster_accuracy'] = pc.get('accuracy')
    thr = ctx.q.accuracy if ctx.q.accuracy is not None else _G6_DEFAULT_RTOL
    d.update(error_estimate_sum=E, cumulant_l1=S, accuracy_estimate=R,
             accuracy_threshold=float(thr), n_estimated_subsets=n_est)
    src = 'question.accuracy' if ctx.q.accuracy is not None else \
        'per-subset acceptance COND_RTOL'
    Rs = 'n/a' if R is None else f'{R:.3g}'
    if R is not None:
        why = ''
    elif not complete:
        why = '; the grid was evaluated in worker processes, nothing was checked'
    else:
        why = '; no subset carried an estimate, nothing was checked'
    ctx.add('G6_ACCURACY',
            f'accuracy estimate R = {Rs}: summed error estimates {E:.3g} of '
            f'{n_est} closed-form subset evaluation(s) against '
            f'sum_tau |C(tau)| = {S:.6g} over the tau grid'
            + (f' plus the realization\'s a posteriori relative error '
               f'{real_rel:.3g}' if real_rel else '')
            + f' (threshold {thr:g}, {src}){why}.',
            'pass parallel=False for the accuracy estimate'
            if (R is None and not complete) else '')
    if R is not None and R > thr:
        ctx.add('G6_ACCURACY_LOW',
                f'the accuracy estimate R = {Rs} exceeds the threshold {thr:g} '
                f'({src}): cancellation between subsets has amplified their '
                'rounding errors (or the cumulant is not finite); the '
                'cumulant may have fewer correct digits.',
                'check G6_ILL_CONDITIONED and G3_CLOSE_POLES (close or '
                'near-degenerate poles); a looser question.accuracy accepts '
                'the result')
    complete_note = '' if complete else ' (parent process only)'
    ctx.ledger_add('approximation', 'phase_j_accuracy',
                   f'grid-L1 relative error estimate R = {Rs}{complete_note} '
                   f'against threshold {thr:g}',
                   declared=ctx.q.accuracy is not None,
                   satisfied=(R <= thr) if (complete and R is not None)
                   else None)

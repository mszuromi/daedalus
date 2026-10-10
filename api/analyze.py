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

**Detected** by gates G0-G4, cheapest first.  Each returns
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

The assumptions ledger lists every consent, convention, assertion and
approximation a result for this question would depend on.  ``route`` is
a placeholder until the backend registry exists.  Not wired into
``compute_cumulants``.
"""
from __future__ import annotations

import contextlib
import io
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
    'G2_NO_SADDLE':              ('G2', 'error'),
    'G2_MF_FAILED':              ('G2', 'error'),
    'G2_UNSTABLE_SADDLE':        ('G2', 'error'),
    'G2_MARGINAL_MODE':          ('G2', 'error'),
    'G2_FIXED_POINT_INDEX':      ('G2', 'error'),
    'G2_MULTISTABLE':            ('G2', 'warning'),
    'G2_SINGLE_SADDLE_SOLVER':   ('G2', 'info'),
    'G2_STABILITY_UNAVAILABLE':  ('G2', 'warning'),
    'G2_ASSERTION_VIOLATED':     ('G2', 'error'),
    'G2_SKIPPED':                ('G2', 'info'),
    # G3 propagator
    'G3_UNRESOLVED_SYMBOLS':     ('G3', 'error'),
    'G3_NONRATIONAL_KERNEL':     ('G3', 'error'),
    'G3_POLES_UNAVAILABLE':      ('G3', 'warning'),
    'G3_REPEATED_POLE':          ('G3', 'warning'),
    'G3_CLOSE_POLES':            ('G3', 'warning'),
    'G3_STIFF':                  ('G3', 'info'),
    'G3_DELTA_COMPONENT':        ('G3', 'info'),
    'G3_IMPROPER':               ('G3', 'error'),
    'G3_ACAUSAL_POLE':           ('G3', 'error'),
    'G3_SKIPPED':                ('G3', 'info'),
    # G4 noise
    'G4_NO_NOISE':               ('G4', 'warning'),
    'G4_NON_GAUSSIAN':           ('G4', 'info'),
    'G4_CROSS_CORRELATED':       ('G4', 'info'),
    'G4_COLORED_RATIONAL':       ('G4', 'info'),
    'G4_NONRATIONAL_NOISE':      ('G4', 'error'),
    'G4_MARKOV_CONSENT_MISSING': ('G4', 'warning'),
    'G4_NEGATIVE_VARIANCE':      ('G4', 'error'),
    # a bug in a gate (caught, never raised)
    'INTERNAL_ERROR':            ('G?', 'error'),
}

# Relative tolerances (decided in the D1 brief).
_MARGINAL_RTOL = 1e-8      # |Re lambda| (or |Im omega_pole|) / scale
_CLOSE_POLE_RTOL = 1e-3    # |p_i - p_j| / max(|p_i|, |p_j|)
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


@dataclass(frozen=True)
class Policy:
    """How permissive the user is (declared, never inferred).

    ``mode``: ``'permissive'`` (default) or ``'strict'`` (warnings become
    errors).  Consents: ``allow_taylor_truncation``,
    ``allow_markov_embedding``, ``allow_numeric_fallback``.
    ``interpretation``: the stochastic convention the action is written in
    (``'ito'`` or ``'stratonovich'``); needed when the noise is state
    dependent.  ``assertions``: facts the user vouches for, e.g.
    ``{'saddle_confirmed': True, 'positive_at_saddle': ['nstar']}``;
    the checkable ones are checked, the rest are echoed in the ledger.
    """
    mode: str = 'permissive'
    allow_taylor_truncation: bool = False
    allow_markov_embedding: bool = False
    allow_numeric_fallback: bool = False
    interpretation: Optional[str] = None
    assertions: dict = field(default_factory=dict)


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
    gate was skipped or failed; a finding says why)."""
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
    mf_solver: Optional[str] = None       # 'dae' | 'iteration'
    n_saddles: Optional[int] = None
    n_stable_saddles: Optional[int] = None
    saddle_index: Optional[int] = None    # index used (pipeline semantics)
    saddle: Optional[tuple] = None        # ((saddle_name, (values...)), ...)
    stable: Optional[bool] = None         # stability of the selected saddle
    stability_source: Optional[str] = None  # 'linearization' | 'poles'
    eigenvalues: tuple = ()               # finite linearization eigenvalues
    n_marginal_modes: Optional[int] = None
    # G3
    propagator_class: Optional[str] = None  # 'rational' | 'non-rational' | 'spatial'
    nonrational_factors: tuple = ()
    n_poles: Optional[int] = None         # with multiplicity (degree of Q(omega))
    poles: tuple = ()                     # distinct roots of Q(omega)
    repeated_poles: Optional[bool] = None
    n_close_pole_pairs: Optional[int] = None
    fast_slow_ratio: Optional[float] = None
    delta_component: Optional[bool] = None
    causal: Optional[bool] = None
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
    route: str = 'unrouted: backend registry not implemented yet'
    question: Optional[Question] = None
    policy: Optional[Policy] = None

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
        return '\n'.join(lines)


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

    def add(self, code, message, remedy=''):
        gate, level = FINDING_CODES[code]
        self.findings.append(Finding(code=code, level=level, message=message,
                                     remedy=remedy, gate=gate))

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
                 _ledger_common):
        try:
            gate(ctx)
        except Exception as exc:        # a gate bug must not escape analyze
            ctx.add('INTERNAL_ERROR',
                    f'internal error in {gate.__name__}: {_exc_text(exc)}',
                    'report this model to the maintainers')
            ctx.stop_numeric = True

    findings = ctx.findings
    if policy.mode == 'strict':
        findings = [replace(f, level='error',
                            remedy=(f.remedy + ' ' if f.remedy else '')
                            + '[strict policy: warning promoted to error]')
                    if f.level == 'warning' else f for f in findings]
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    findings = sorted(findings, key=lambda f: (order.get(f.level, 9), f.gate))
    ctx.t['analysis_time_s'] = time.perf_counter() - t_start
    known = {f.name for f in fields(ModelTraits)}
    traits = ModelTraits(**{k: v for k, v in ctx.t.items() if k in known})
    return ModelReport(traits=traits, findings=tuple(findings),
                       ledger=tuple(ctx.ledger), question=question,
                       policy=policy)


# ── G0: declarations ──────────────────────────────────────────────────


def _resolve_parameters(ctx):
    """The parameter point: the question's, else the declared defaults
    (``compute_cumulants``' rule, with its error text)."""
    model, q = ctx.model, ctx.q
    pspecs = model.get('parameters', []) or []
    declared = {p.get('name') for p in pspecs}
    if q.parameters:
        unknown = sorted(set(q.parameters) - declared)
        if unknown:
            ctx.add('G0_UNKNOWN_PARAMETER',
                    f'question.parameters names undeclared parameter(s): '
                    f'{", ".join(unknown)}.',
                    'check the spelling against the model\'s parameters')
        fundamental = {p['name']: p['default'] for p in pspecs
                       if p.get('default') is not None}
        fundamental.update(q.parameters)
        missing = sorted(p['name'] for p in pspecs
                         if p['name'] not in fundamental
                         and not p.get('mean_field'))
    else:
        fundamental = {p['name']: p['default'] for p in pspecs
                       if p.get('default') is not None}
        missing = sorted(p['name'] for p in pspecs
                         if p.get('default') is None
                         and not p.get('mean_field'))
    if missing:
        ctx.add('G0_PARAMETERS_MISSING',
                'compute_cumulants: no ``parameters=`` given, and these model '
                'parameters declare no default: ' + ', '.join(missing) + '.',
                'Pass parameters={...} or declare defaults via '
                'ModelBuilder.parameter(name, default=...).')
        return None
    return fundamental


def _flat_values(v):
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
        for f in dict.fromkeys(ext_int):
            if f not in phys_idx:
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
    sectors = ctx.sectors
    species = sorted(k for k in sectors if k[0] + k[1] >= 3 and k[1] >= 1)
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

    # the linear operator K(omega)
    _build_k_ft(ctx)


def _read_expand_cache(ctx, N):
    """The pipeline's expand-cache bundle for ``N`` in dict form
    ``{(n_resp, n_phys): {exponent: coefficient}}``, or ``None``.

    Runs the validity checks of ``api._expand_cache.load_expand`` (ring
    variables, ``n_tilde``, model-text fingerprint, operator-IR signature)
    but skips its coercion back into the polynomial ring, which is the
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
    if list(bundle.get('ring_var_names', [])) != _ec._get_ring_var_names(ft):
        return None
    if int(bundle.get('n_tilde', -1)) != int(ft._n_tilde):
        return None
    spec = model.get('spec_signature')
    if spec is not None and bundle.get('spec_signature') != spec:
        return None
    if _ec.vertex_form_factor_signature(ft._ns) != bundle.get('vertex_signature'):
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
    S_free = ctx.sectors.get((1, 1), {})
    names = list(ctx.ring_names)
    nf = ft._n_tilde
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
    zero_cols = [phys[j] for j in range(len(phys))
                 if all(SR(K[i][j]).is_zero() for i in range(nf))]
    zero_rows = [resp[i] for i in range(nf)
                 if all(SR(K[i][j]).is_zero() for j in range(len(phys)))]
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
        return
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
    return Nmat, sorted(sources), unresolved


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
    scale = max(1.0, float(np.max(np.abs(eigs))))
    tol = _MARGINAL_RTOL * scale
    re = eigs.real
    n_marg = int(np.sum(np.abs(re) < tol))
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
    if not ctx.model.get('equations') or not _g2_ready(ctx):
        return
    fpi = ctx.q.fixed_point_index
    _g2_dae(ctx, fpi, 0 if fpi is None else int(fpi))
    _check_positivity(ctx)


def _g2_iteration_gate(ctx):
    """The single saddle of the mean-field iteration rows (legacy models)."""
    model, fund = ctx.model, ctx.fundamental
    if model.get('equations') or not _g2_ready(ctx):
        return
    fpi = ctx.q.fixed_point_index
    ctx.t['mf_solver'] = 'iteration'
    try:
        from api._mean_field import solve_mean_field
        with _quiet():
            mf = solve_mean_field(ctx.ft, model, fund, verbose=False)
    except Exception as exc:
        ctx.add('G2_MF_FAILED', f'the mean-field solve failed: {_exc_text(exc)}',
                'check the mean-field equations and the parameter point')
        ctx.stop_numeric = True
        return
    ctx.num_params = mf['num_params']
    ctx.t['n_saddles'] = 1
    ctx.t['saddle_index'] = 0
    sv = mf.get('saddle_values') or {}
    if not sv:
        sv = {k: v for k, v in (('nstar', mf.get('nstar_vals')),
                                ('vstar', mf.get('vstar_vals'))) if v}
    ctx.t['saddle'] = _saddle_tuple(sv)
    ctx.add('G2_SINGLE_SADDLE_SOLVER',
            'the model declares mean-field iteration rows (no '
            '.equation(...)): the iteration solver returns one saddle; '
            'other saddles are not searched. Stability is read from the '
            'propagator poles (G3).',
            'declare .equation(lhs=..., rhs=...) rows to search every saddle')
    if fpi not in (None, 0):
        ctx.add('G2_FIXED_POINT_INDEX',
                f'fixed_point_index={fpi} is ignored by the iteration '
                'solver (one saddle).', 'drop fixed_point_index')
    _check_positivity(ctx)


def _saddle_tuple(values):
    return tuple((str(k), tuple(float(x) for x in (v if isinstance(v, (list, tuple))
                                                   else [v])))
                 for k, v in sorted(values.items()))


def _g2_dae(ctx, fpi, fpi_eff):
    from api._mean_field_dae import (solve_mean_field_dae_compat,
                                     linear_stability)
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
    ctx.t['n_saddles'] = len(roots)
    recs = []
    any_dynamic = False
    for r in roots:
        try:
            with _quiet():
                st = linear_stability(model, fund, r)
            eigs = np.asarray(st['eigenvalues_finite'], dtype=complex)
            cls, n_marg = _classify_eigs(eigs)
            any_dynamic = any_dynamic or eigs.size > 0
            recs.append({'values': r, 'eigs': eigs, 'class': cls, 'n_marg': n_marg})
        except Exception as exc:
            recs.append({'values': r, 'eigs': np.array([], dtype=complex),
                         'class': None, 'n_marg': 0, 'error': _exc_text(exc)})
    stab_on = bool(model.get('stability_analysis', False))
    if not any_dynamic:
        ctx.add('G2_STABILITY_UNAVAILABLE',
                'the linearization has no finite eigenvalue (algebraic '
                'equations, or the stability evaluation failed'
                + (f': {recs[0]["error"]}' if recs and recs[0].get('error') else '')
                + '); stability is read from the propagator poles (G3).', '')
        selectable = list(range(len(recs)))
    else:
        ctx.stability_known = True
        stable_idx = [i for i, r in enumerate(recs) if r['class'] == 'stable']
        ctx.t['n_stable_saddles'] = len(stable_idx)
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
        ctx.add('G2_UNSTABLE_SADDLE',
                f'no stable saddle among {len(recs)}: '
                + '; '.join(f'{_fmt_root(r["values"])} eig {_fmt_eigs(r["eigs"])}'
                            for r in recs)
                + '. The pipeline raises (stability_analysis on) and the '
                'diagrammatic expansion is undefined.',
                'change the parameters to a stable regime')
        ctx.t['stable'] = False
        ctx.t['stability_source'] = 'linearization'
        ctx.stop_numeric = True
        return
    if fpi is not None and not (0 <= fpi < len(selectable)):
        ctx.add('G2_FIXED_POINT_INDEX',
                f'fixed_point_index={fpi} is out of range: '
                f'{len(selectable)} {"stable" if stab_on and any_dynamic else ""}'
                f' saddle(s) to choose from (the pipeline clamps it with a '
                'warning).', f'use 0..{len(selectable) - 1}')
        fpi_eff = max(0, min(fpi_eff, len(selectable) - 1))
    sel = selectable[min(fpi_eff, len(selectable) - 1)]
    ctx.t['saddle_index'] = selectable.index(sel)
    rec = recs[sel]
    ctx.t['saddle'] = _saddle_tuple(rec['values'])
    # num_params for the selected root (compat built root 0)
    num_params = dict(mf['num_params'])
    for sname, vals in rec['values'].items():
        arr = getattr(ctx.ft._ns, sname, None)
        if arr is None:
            continue
        for i, v in enumerate(vals):
            num_params[arr[i]] = float(v)
    ctx.num_params = num_params
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
    for nm in names:
        key = nm if nm in saddle else nat.get(nm, f'{nm}star')
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


def _g3_propagator(ctx):
    if ctx.K_ft is None:
        return
    if ctx.t.get('spatial'):
        ctx.t['propagator_class'] = 'spatial'
        return
    from sage.all import SR
    omega = ctx.omega
    bad = []
    for e in ctx.K_ft.list():
        bad.extend(_offending_factors(e, [omega], allow_negative_powers=True))
    bad = list(dict.fromkeys(bad))
    if bad:
        ctx.t['propagator_class'] = 'non-rational'
        ctx.t['nonrational_factors'] = tuple(bad[:6])
        ctx.add('G3_NONRATIONAL_KERNEL',
                f'G(omega) = K(omega)^-1 is not rational in omega: offending '
                f'factor(s) {", ".join(bad[:4])}. The pole/residue machinery '
                '(and Phase J) needs a rational propagator.',
                'replace the kernel by a rational (Pade or sum-of-exponentials) '
                'approximation, or embed it with auxiliary fields')
        return
    ctx.t['propagator_class'] = 'rational'
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
        res = _exact_poles(K_num, omega, ctx.n_tilde)
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
    Q_deg, poles, squarefree, delta, improper = res
    ctx.t['n_poles'] = Q_deg
    ctx.t['poles'] = tuple(poles)
    ctx.t['repeated_poles'] = not squarefree
    ctx.t['delta_component'] = bool(delta)
    if improper:
        ctx.add('G3_IMPROPER',
                f'improper propagator entries {improper} (deg P > deg Q): '
                'G(omega) grows at large omega; K is not causal/proper.',
                'check the Dt terms (one Dt per field, on its own response)')
    if not squarefree:
        ctx.add('G3_REPEATED_POLE',
                'Q(omega) has a repeated root (Jordan block): the single-pole '
                'residue formula does not apply; the pipeline falls back to '
                'its numerical tier.',
                'perturb the parameters off the degeneracy (e.g. mu != 1/tauc)')
    close = []
    for i in range(len(poles)):
        for j in range(i + 1, len(poles)):
            a, b = poles[i], poles[j]
            if abs(a - b) <= _CLOSE_POLE_RTOL * max(abs(a), abs(b), 1e-300):
                close.append((a, b))
    ctx.t['n_close_pole_pairs'] = len(close)
    if close:
        ctx.add('G3_CLOSE_POLES',
                f'{len(close)} pole pair(s) closer than {_CLOSE_POLE_RTOL:g} '
                f'relative: {[(complex(a), complex(b)) for a, b in close[:3]]}; '
                'residues ~ 1/gap lose precision.',
                'perturb the parameters, or treat the pair as a double pole')
    rates = [p.imag for p in poles if p.imag > 0]
    if rates:
        ratio = max(rates) / min(rates)
        ctx.t['fast_slow_ratio'] = float(ratio)
        if ratio > _STIFF_RATIO:
            ctx.add('G3_STIFF', f'fast/slow decay-rate ratio {ratio:.3g} > '
                    f'{_STIFF_RATIO:g}: the tau grid must resolve both scales.', '')
    if delta:
        ctx.add('G3_DELTA_COMPONENT',
                f'G(omega) has an instantaneous (delta) part in entries {delta}.', '')
    scale = max([1.0] + [abs(p) for p in poles])
    tol = _MARGINAL_RTOL * scale
    lower = [p for p in poles if p.imag < -tol]
    marginal = [p for p in poles if abs(p.imag) <= tol]
    ctx.t['causal'] = not lower and not marginal
    if ctx.stability_known:
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


def _exact_poles(K_num, omega, nf):
    """Exact inverse of K(omega) in Frac(QQ[i][omega]) (the pipeline's
    tier-1 arithmetic).  Returns ``(deg Q, distinct poles, squarefree,
    delta entries, improper entries)`` or ``None`` on budget expiry."""
    from sage.all import (SR, QQ, CDF, PolynomialRing, CyclotomicField,
                          matrix)
    from api._propagator import _run_with_timeout, _PolynomialPathTimeout
    CF = CyclotomicField(4, 'I_')
    iC = CF.gen()
    PR = PolynomialRing(CF, 'om_an')
    F = PR.fraction_field()

    def to_cf(c):
        c = SR(c)
        return CF(QQ(c.real_part())) + CF(QQ(c.imag_part())) * iC

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
        return 0, [], True, delta, improper
    PRc = PolynomialRing(CDF, 'om_c')
    Qc = PRc([complex(c) for c in Q.coefficients(sparse=False)])
    poles = sorted((complex(r) for r, _ in Qc.roots(CDF)),
                   key=lambda p: (p.imag, p.real)) if Q.degree() > 0 else []
    return int(Q.degree()), poles, bool(Q.is_squarefree()), delta, improper


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
    if not hasattr(ns, '_cumulant_kernels'):   # an expand-cache load skips it
        from engine.core.field_theory import _build_cumulant_action
        with _quiet():
            _build_cumulant_action(ns, model)
    kernels = getattr(ns, '_cumulant_kernels', {}) or {}
    noise_class = 'white'
    if kernels:
        bad = _cumulant_kernel_nonrational(kernels)
        if bad:
            noise_class = 'colored-nonrational'
            ctx.add('G4_NONRATIONAL_NOISE',
                    f'colored noise kernel(s) with a non-rational spectrum: '
                    f'{bad[:3]}.',
                    'use an exponential (Lorentzian) kernel, which is embedded '
                    'exactly by an auxiliary OU field')
        else:
            noise_class = 'colored-rational'
            ctx.add('G4_COLORED_RATIONAL',
                    f'colored noise with a rational spectrum '
                    f'({len(kernels)} kernel entr{"y" if len(kernels) == 1 else "ies"}).', '')
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
        scale = max(1.0, float(np.max(np.abs(Nmat)))) if nf else 1.0
        if w.size and w.min() < -1e-10 * scale:
            ctx.add('G4_NEGATIVE_VARIANCE',
                    f'the Gaussian noise covariance at the saddle has a '
                    f'negative eigenvalue ({w.min():.4g}): a negative variance '
                    '(e.g. a negative rate or diffusion constant at the saddle).',
                    'check the sign of the noise term (-D*xt^2) and the saddle')


def _cumulant_kernel_nonrational(kernels):
    """Fourier image of each registered smooth cumulant kernel K(tau);
    returns the entries whose image is not rational in omega."""
    from sage.all import SR
    from engine.core.field_theory import fourier_transform
    tau = SR.var('tau')
    omega = SR.var('omega')
    bad = []
    for key, fn in kernels.items():
        try:
            K = SR(fn(tau)) if callable(fn) else SR(fn)
            img = fourier_transform(K, tau, omega)
            if img.has(SR.var('tau')) or _offending_factors(
                    img, [omega], allow_negative_powers=True):
                bad.append(str(key))
        except Exception:
            bad.append(f'{key} (Fourier image unavailable)')
    return bad


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

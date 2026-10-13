"""
api.routing — the backend registry: which integrator computes a cumulant.

Each backend declares ``applies(request) -> (bool, reasons)`` and a
``priority`` (lower is tried first); :func:`select` takes the first
applicable one and returns a :class:`Route` that records the choice and
every rejected backend with its reasons.  ``compute_cumulants`` calls
``route.backend.run(state)`` (``state``: the pipeline objects after the
mean-field stage, see ``api.compute._PipelineState``), so a new backend is
one :func:`register` call and touches no existing backend::

    from api.routing import Backend, Reason, register
    register(Backend(
        name='my_backend', priority=5,
        applies=lambda req: ((True, ()) if req.traits.get('my_trait')
                             else (False, (Reason('my_trait', 'no my_trait'),))),
        run=my_runner, description='...'))

The built-in backends reproduce the dispatch of ``compute_cumulants`` as of
0.2.0 exactly (``tests/test_analyzer_routing.py`` prints the table):

=====================  ========  ==============================================
backend                priority  applies when
=====================  ========  ==============================================
spatial_integrator     10        spatial model and ``spatial_grid`` (k = 2,
                                 ``max_ell <= 2``, stationary initial state) or
                                 ``spatial_points`` (k != 2)
phase_j_grouped        20        temporal model, ``use_grouped_phase_j=True``,
                                 propagator rational in omega (or unknown)
phase_j_per_diagram    30        temporal model, propagator rational (or unknown)
phase_h_symbolic       90        never automatically: registered so that it is
                                 listed; it has no ``compute_cumulants`` runner
=====================  ========  ==============================================

Honest failure: when no backend applies, :class:`NoBackendError` names the
first failing trait (of the first backend whose rejection is not a plain
domain mismatch, e.g. "the model is temporal" for the spatial backend) and
lists every rejected backend with its reasons.  The exception is raised as
the trait's own exception type (``ValueError`` / ``NotImplementedError``,
the types ``compute_cumulants`` raised for these cases before the registry
existed: :class:`NoBackendValueError` / :class:`NoBackendNotImplementedError`)
so existing callers keep working.  The message starts with the text 0.2.0
raised and adds the list of rejected backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = ['Backend', 'Reason', 'RouteRequest', 'Route', 'NoBackendError',
           'NoBackendValueError', 'NoBackendNotImplementedError',
           'register', 'unregister', 'backends', 'select', 'routing_table']


@dataclass(frozen=True)
class Reason:
    """Why a backend does not apply.  ``trait``: the failing trait;
    ``domain``: the rejection is a plain domain mismatch (spatial vs
    temporal), not a failing property of a model the backend is for;
    ``exc``: the exception type to raise when this is the reason reported
    for an honest failure."""
    trait: str
    message: str
    domain: bool = False
    exc: type = ValueError


@dataclass(frozen=True)
class RouteRequest:
    """What a backend's ``applies`` sees: the question (declared) and the
    traits the analyzer detected so far (``traits``: a dict of
    ``api.analyze.ModelTraits`` field values; empty when the analyzer is
    off, so every check here must also work from ``spatial``)."""
    model_name: str
    spatial: bool
    k: int
    max_ell: int
    has_spatial_grid: bool = False
    has_spatial_points: bool = False
    grouped: bool = False
    initial_mode: str = 'stationary'
    requested: Optional[str] = None      # an explicit backend name
    traits: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Backend:
    """An integrator backend.  ``run(state) -> result dict``;
    ``standard_tail``: ``compute_cumulants`` saves ``output_npz`` /
    ``output_csv`` and prints its summary after ``run`` (the temporal
    backends; the spatial one returns early, as before the registry)."""
    name: str
    priority: int
    applies: Callable[[RouteRequest], tuple]
    run: Callable[[Any], dict]
    description: str = ''
    standard_tail: bool = False


@dataclass(frozen=True)
class Route:
    """The routing decision: the chosen backend and every rejected one
    (``(name, reasons)`` in priority order)."""
    backend: Backend
    rejected: tuple = ()
    request: Optional[RouteRequest] = None

    @property
    def name(self) -> str:
        return self.backend.name

    def as_dict(self) -> dict:
        """JSON-able record (saved with each result)."""
        return {'backend': self.backend.name,
                'priority': self.backend.priority,
                'rejected': [{'backend': n,
                              'reasons': [r.message for r in rs]}
                             for n, rs in self.rejected]}


class NoBackendError(Exception):
    """No registered backend applies.  Raised through :func:`_raise_none`
    as a subclass of the reported trait's exception type."""

    def __init__(self, message, rejected=(), request=None):
        super().__init__(message)
        self.rejected = tuple(rejected)
        self.request = request

    def __reduce__(self):                 # picklable (process pools)
        return (type(self), (str(self), self.rejected, self.request))


class NoBackendValueError(NoBackendError, ValueError):
    """No backend applies; the failing trait is a ``ValueError`` case."""


class NoBackendNotImplementedError(NoBackendError, NotImplementedError):
    """No backend applies; the failing trait is a ``NotImplementedError``
    case (a capability the integrators do not have yet)."""


_NO_BACKEND_BY_TYPE = {ValueError: NoBackendValueError,
                       NotImplementedError: NoBackendNotImplementedError}


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend, *, replace: bool = False) -> Backend:
    """Add ``backend`` to the registry (``replace=True`` to overwrite a
    backend of the same name)."""
    if backend.name in _REGISTRY and not replace:
        raise ValueError(f'backend {backend.name!r} is already registered')
    _REGISTRY[backend.name] = backend
    return backend


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def backends() -> list[Backend]:
    """The registered backends in priority order (name breaks ties)."""
    return sorted(_REGISTRY.values(), key=lambda b: (b.priority, b.name))


def _call_applies(b, request):
    try:
        ok, reasons = b.applies(request)
    except Exception as exc:          # a broken applies must not pick it
        return False, (Reason('applies', f'applies() raised '
                                         f'{type(exc).__name__}: {exc}'),)
    reasons = tuple(r if isinstance(r, Reason) else Reason('?', str(r))
                    for r in (reasons or ()))
    return bool(ok), reasons


def select(request: RouteRequest) -> Route:
    """The first applicable backend (priority order); ``request.requested``
    restricts the choice to that backend.  Raises (see the module
    docstring) when none applies."""
    rejected = []
    cands = backends()
    if request.requested is not None:
        named = [b for b in cands if b.name == request.requested]
        if not named:
            raise ValueError(
                f'backend={request.requested!r} is not registered; known: '
                f'{[b.name for b in cands]}')
        cands = named
    for b in cands:
        ok, reasons = _call_applies(b, request)
        if ok:
            return Route(backend=b, rejected=tuple(rejected), request=request)
        rejected.append((b.name, reasons))
    _raise_none(rejected, request)


def _raise_none(rejected, request):
    first = None
    for _, reasons in rejected:
        specific = [r for r in reasons if not r.domain]
        if specific:
            first = specific[0]
            break
    if first is None:
        first = next((rs[0] for _, rs in rejected if rs),
                     Reason('?', 'no backend is registered'))
    lines = [first.message, '',
             f'No integrator backend applies to model '
             f'{request.model_name!r} (k={request.k}, '
             f'max_ell={request.max_ell}); first failing trait: '
             f'{first.trait}.  Rejected backends:']
    for name, reasons in rejected:
        lines.append(f'  - {name}: '
                     + ('; '.join(r.message for r in reasons) or '(no reason)'))
    exc_type = first.exc if isinstance(first.exc, type) else ValueError
    cls = _NO_BACKEND_BY_TYPE.get(exc_type, NoBackendValueError)
    raise cls('\n'.join(lines), rejected, request)


def routing_table(requests) -> list[tuple[str, str]]:
    """``[(label, backend name or 'NONE: <first reason>'), ...]`` for the
    ``(label, RouteRequest)`` pairs (the zoo table of the tests)."""
    out = []
    for label, req in requests:
        try:
            out.append((label, select(req).name))
        except Exception as exc:          # NoBackendError and subclasses
            out.append((label, 'NONE: ' + str(exc).splitlines()[0]))
    return out


# ── the built-in backends ─────────────────────────────────────────────


def _rational_reason(req):
    pc = req.traits.get('propagator_class')
    if pc not in (None, 'rational'):
        factors = req.traits.get('nonrational_factors') or ()
        return Reason('propagator_class',
                      f'propagator is not rational in omega (class {pc!r}'
                      + (f', factor {factors[0]}' if factors else '')
                      + '): the pole/residue machinery of Phase J needs a '
                        'rational G(omega)')
    return None


def _temporal_domain(req):
    if req.spatial:
        return Reason(
            'spatial',
            f"model {req.model_name!r} is spatial (a field declares "
            f"spatial_dim>=1): the temporal pole-finder/Phase-J path cannot "
            f"consume the Laplacian operator", domain=True)
    return None


def _applies_phase_j_per_diagram(req):
    reasons = [r for r in (_temporal_domain(req), _rational_reason(req)) if r]
    return (not reasons), tuple(reasons)


def _applies_phase_j_grouped(req):
    reasons = [r for r in (_temporal_domain(req), _rational_reason(req)) if r]
    if not req.grouped:
        reasons.append(Reason('use_grouped_phase_j',
                              'use_grouped_phase_j=False (the grouped path '
                              'is opt-in)', domain=True))
    return (not reasons), tuple(reasons)


def _applies_spatial(req):
    """The spatial short-circuit of ``compute_cumulants`` (0.2.0), its
    checks in their original order and with their original messages."""
    if not req.spatial:
        return False, (Reason('spatial', 'the model is temporal (no field '
                                         'declares spatial_dim>=1)',
                              domain=True),)
    if not (req.has_spatial_grid or req.has_spatial_points):
        return False, (Reason(
            'spatial_grid',
            f"model {req.model_name!r} is spatial (a field declares "
            f"spatial_dim>=1): compute_cumulants requires spatial_grid="
            f"... to route to the spatial integrator.  The temporal pole-finder/"
            f"Phase-J path cannot consume the Laplacian operator.  Pass "
            f"spatial_grid (e.g. the model's METADATA['spatial_grid'], or a "
            f"numpy array like np.linspace(0.0, 6.0, 25))."),)
    k = req.k
    if k != 2 and not req.has_spatial_points:
        return False, (Reason(
            'spatial_points',
            f'spatial k={k}: pass spatial_points (an (n_pts, {k-1}, 2) '
            'array of (x_j, tau_j) offsets per non-anchor external '
            'slot) to evaluate the k-point cumulant at explicit '
            'events.  Grid output (spatial_grid/tau_max) is k=2 only.',
            exc=NotImplementedError),)
    if k == 2 and not req.has_spatial_grid:
        return False, (Reason(
            'spatial_grid',
            'spatial k=2 returns C(x,τ) on a grid — pass spatial_grid '
            '(spatial_points is the k>=3 event API).'),)
    if k != 2:
        return True, ()
    if req.max_ell > 2:
        return False, (Reason(
            'max_ell',
            f'spatial v1 implements tree (max_ell=0), 1-loop (max_ell=1) and '
            f'2-loop (max_ell=2) via the full-diagram integrator; got '
            f'max_ell={req.max_ell}.  Higher loops work by construction but are '
            f'increasingly expensive (more diagrams, higher-dim integrals).',
            exc=NotImplementedError),)
    if req.initial_mode != 'stationary':
        return False, (Reason(
            'initial_mode',
            f"spatial v1 supports the 'stationary' initial condition "
            f"only; got {req.initial_mode!r}.", exc=NotImplementedError),)
    return True, ()


def _applies_phase_h(req):
    return False, (Reason(
        'runner', 'the symbolic frequency-domain integrator '
        '(engine.integration.symbolic, Phase H) is a library without a '
        'compute_cumulants runner; it is never selected', domain=True),)


def _run_lazy(name):
    def run(state):
        from api import compute as _compute
        return getattr(_compute, name)(state)
    run.__name__ = f'run_{name}'
    return run


def _run_phase_h(state):              # pragma: no cover - never selected
    raise NotImplementedError('phase_h_symbolic has no compute_cumulants runner')


register(Backend(
    name='spatial_integrator', priority=10, applies=_applies_spatial,
    run=_run_lazy('_run_spatial'),
    description='full-diagram momentum integrator (Symanzik, causal '
                'chambers, q->x transform); k=2 grid or k>=3 events'))
register(Backend(
    name='phase_j_grouped', priority=20, applies=_applies_phase_j_grouped,
    run=_run_lazy('_run_phase_j_grouped'), standard_tail=True,
    description='Phase J, typed diagrams grouped by parent prediagram '
                '(api._grouped_phase_j)'))
register(Backend(
    name='phase_j_per_diagram', priority=30,
    applies=_applies_phase_j_per_diagram,
    run=_run_lazy('_run_phase_j_per_diagram'), standard_tail=True,
    description='Phase J, one time-domain integral per typed diagram '
                '(engine.integration.time_domain.pipeline)'))
register(Backend(
    name='phase_h_symbolic', priority=90, applies=_applies_phase_h,
    run=_run_phase_h,
    description='symbolic residue integration in omega (library only)'))

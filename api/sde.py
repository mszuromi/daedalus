"""SDE front-end: declare fields, drift and noise SOURCES; derive the MSR
action and the mean-field equations; emit the same model dict
:class:`api.model.TemporalModelBuilder` produces (for an SPDE, the one
:class:`api.model.SpatialModelBuilder` produces; see *Declared operators and
SPDEs* below).

Each field ``x`` obeys one equation in the repository's operator form,

    lhs_x = rhs_x + sum_a c_xa * eta_a'

where ``lhs_x`` is the linear differential operator on ``x`` (contains
``Dt``: ``'Dt*x'``, ``'(tau*Dt + 1)*v'``), ``rhs_x`` is everything
deterministic (fields, parameters, defined functions, kernels), and the
noise sources ``eta_a`` (:mod:`api.noise`) couple in through
``into={field: coefficient}`` (the coefficient ``c_xa``, not a population; ``couples=``
is its former name).  The
derived action is

    S = sum_x xt*(lhs_x - rhs_x) - sum_a K_a( sum_x c_xa*xt )

summed over each population's index (``sum(... for i in pop)``), and the
mean-field equations are ``lhs_x|_{Dt=0} = rhs_x + sum_a c_xa*K_a'(0)``.

A source may be *exposed* as a field (``as_field='n'``): the field ``n``
(e.g. a spike train) and its response ``nt`` are declared, the pairing
``nt*n - K(nt)`` enters the action, the equations use ``n`` like any
field, and the mean-field system gains ``n = K'(0)``.

Example (OU quartic)::

    m = (SDE('OU quartic')
         .parameter('mu', default=1.0, domain='positive')
         .parameter('eps', default=0.02, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .field('x')
         .drift('x', '-mu*x - eps*x^3')
         .noise('xi', Gaussian(var='2*D'), into={'x': 1})
         .build())

Expressions may use bare field names (``x``) or indexed ones
(``x[i]``); bare field names, bare parameters indexed by one population
and bare calls of declared functions are indexed by the population index
``i`` automatically.  Population sums
(``sum(w[i,j]*g*n[j] for j in E)``) are written as in the tracked
models.

Conventions: Ito by default (``.interpretation('stratonovich')``
converts Gaussian noise to Ito with the drift correction
``f_i += 1/2 sum_a sum_j [V_a b_ja d_j b_ia + 1/2 b_ja b_ia d_j V_a]``,
``b_ia = c_ia / a_i`` with ``a_i`` the ``Dt`` coefficient of ``lhs_i``
and ``V_a`` the variance rate; it raises for jump and exposed sources).
Time-independent sources only.

``.interpretation('marcus')`` extends Stratonovich to jump sources: a jump
``z`` of a source coupled through the vector field ``G = sum_i b_i(x) d/dx_i``
moves the state along the flow of ``G`` for time ``z``,
``x -> exp(z G) x``.  The action stays Ito-ordered (it is the generator's):
the noise term of a source with cumulant rates ``kappa_k`` is
``sum_k kappa_k/k! D_k``, ``D_k = e^{-xt.x} G^k e^{xt.x}``, in place of
``K(sum_i c_i xt_i)``, so enumeration and Phase J are unchanged.  Gaussian
sources (and Gaussian parts of a sum) take the Stratonovich conversion above
(a Gaussian-only model gives exactly the Stratonovich action); a jump source
with a constant coupling is unchanged (the Ito action).  A jump source with an
affine coupling whose flow is closed form (each coupling affine in its own
field, ``x -> e^{bz} x + (a/b)(e^{bz} - 1)``) gets its noise term exactly,
from the jump law's mgf, kept to ``SDE.series_order`` response legs:
``lam E[(e^{bz} - 1)^n]/n!`` per ``(xt (x + a/b))^n``, with no moment series in
``z`` (that series diverges for exponential-tailed jumps at degree
``n > 1/(|b| scale)``).  Other state-dependent couplings carry the series to
``SDE.series_order`` (``marcus_route('series')`` forces it).
``compute_cumulants`` reports, over the vertices ``(k, max_ell)`` can use,
the size of the next series term, or the rounding of the exact coefficients,
at the saddle (``result['marcus_truncation']``, G6), and refuses an exact
route whose jump law has no mgf where a usable vertex needs it.  A
state-dependent rate stays outside
the flow (the pre-jump rate).  Refused under Marcus: exposed sources, a
point process whose spikes enter with a state-dependent coefficient,
couplings that read other units of a population (joint jumps), and
state-dependent ``Cumulants``/``CGF`` sources.

Mean-field equation: ``lhs|_{Dt=0} = rhs`` must define ``x*``, so a lhs such
as ``Dt*x`` is completed.  The linear term of ``rhs`` (field-independent
coefficient) moves onto the lhs; where there is none (``-x/(1 + eps*x^2)``,
``-x*exp(-eps*x^2)``, a linear coefficient that holds a field) ``c*x`` with
``c = 1`` is added to both sides.  The residual, the saddle and the linear
stability do not depend on ``c`` and the action is untouched; ``SDE.notes``
records the step.  Writing ``lhs='(Dt + c)*x'``, ``rhs='drift + c*x'`` by hand
gives the same model.

Higher-order equations and pure integrators
-------------------------------------------
An lhs of order ``N > 1`` in the time derivative, a polynomial acting on the
field alone (``(m*Dt^2 + gamma*Dt + k)*x``, ``m*D_t^2*x + ...`` with a declared ``D_t``), is embedded as
a first-order system, which is what the engine takes (the MSR action is first
order in ``Dt``): the velocities ``xdot`` (``xddot``, ...; or
``equation(..., velocity=...)``) are declared as fields, ``Dt*x = xdot``, and
the top equation is ``m*Dt*xdot + gamma*xdot = rhs - k*x``.  A noise coupled
into ``x`` is the force on the top equation.  It builds the model a
hand-written first-order system builds (the same action, equations and dict).

A pure integrator, ``drift('x', 'v')`` (an rhs without ``x``), is a dynamical
equation whose mean-field equation ``0 = v`` constrains the other fields; the
saddle of ``x`` is fixed by the system solved together (``Dt*x = v`` with
``(m*Dt + gamma)*v = F - k*x`` gives ``x* = F/k``).  An rhs without any field
(``drift('x', '0')``, a random walk) has no stationary mean field and is
refused.

Coloured and vector Gaussian noise
----------------------------------
``Gaussian(var, kernel=h)`` (``h`` an :class:`api.delay.Kernel` with a
rational image: ``from api.delay import Kernel``, then
``Gaussian(var, kernel=Kernel.exponential('tau'))``) is the white noise of
intensity ``var`` filtered by ``h``, ``eta = h * xi``.  It is Markov-embedded,
exactly: with ``H(I*omega) = N/D`` the state field ``z`` (the source's name,
or ``as_field=``) obeys ``D(Dt) z = xi`` (velocities for ``deg D > 1``),
``eta = N(Dt) z``, and ``c*eta`` enters the rhs of each field in ``into=``
(``c`` may depend on the fields).  ``Kernel.exponential(tau)`` gives the
Ornstein-Uhlenbeck noise ``<eta eta>(s) = var/(2 tau) exp(-|s|/tau)``; a
proper image adds its direct part to the fields as white noise.

``GaussianVector(cov=[[...]])`` is correlated white noise of ``n``
components, coupled with ``into={field: [c_1, ..., c_n]}``; its action term
is ``-(1/2) theta^T cov theta``, and under the Stratonovich rule the drift
gains ``1/2 sum_kl cov_kl sum_j b_jl d_j b_ik``.  The simulator draws it
through ``cov = L L^T`` at the run's parameters.

Declared operators and SPDEs
----------------------------
The operators an equation uses are named with :meth:`SDE.operator` and the
objects of :mod:`api.operators`.  ``Dt`` is a ``TimeDerivative`` declared by
default, so a model that never declares an operator is unchanged; declaring
the time derivative under another name renames it (``Dt`` is then refused);
``∂t`` (or ``∂_t``) in an equation is the time derivative too.  Each operator
is written in product style (``(tau*Dt + 1)*x``, ``D*lap*u``) or call style
(``Dt(x)``, ``lap(u)``): ``Dt(x)`` is ``Dt*x``, byte for byte.  An operator
name may not collide with a field (or its ``xt`` / ``dx`` / ``xstar``),
parameter, function, kernel or population, nor with a built-in name, nor
with an increment of the paper-style syntax (``dt``, ``dW``, ``dN``,
``dW_<name>``, ``d<X>_<name>``; a field's ``dx`` already counts).

A ``Laplacian`` (or ``SDE(name, space=d)``) makes the model an SPDE in ``d``
dimensions, built for the spatial pipeline as by hand with
:class:`api.model.SpatialModelBuilder`: one scalar action, one field per
species, space-time white noise ``Gaussian(var=..., space='white')``
(``<eta(x,t) eta(x',t')> = var delta(x - x') delta(t - t')``), the boundary
of :meth:`SDE.boundary` (default infinite) and a stationary initial
condition::

    m = (SDE('Allen-Cahn', space=1)
         .operator('lap', Laplacian())
         .parameter('mu', default=1.0, domain='positive')
         .parameter('D', default=1.0, domain='positive')
         .parameter('lam', default=0.1, domain='positive')
         .parameter('T', default=1.0, domain='positive')
         .field('phi')
         .equation('phi', lhs='Dt(phi)', rhs='D*lap(phi) - mu*phi - lam*phi^3')
         .noise('eta', Gaussian(var='2*T', space='white'), into={'phi': 1})
         .build())

The Laplacian of a field is the plain symbol ``Laplacian`` (``D*lap(u)`` ->
``D*Laplacian*u``); a Laplacian of an expression (``g*lap(u^2)``, a conserved
derivative vertex) writes the action in the spatial operator IR
(``Lap(u^2)``, ``.operator_ir()``), the mean-field equations keep the plain
form (both vanish at the homogeneous saddle).  A product-style operator
multiplies exactly one field (``lap*u^2`` is refused: write ``lap(u^2)``);
the time derivative acts on fields linearly.  An operator acts on the
model's fields (``lap(mu)`` is refused) and is written in the equations (or
:meth:`extra_action`), not inside a :meth:`define_function` body or a noise
expression.  Temporal-only features are
refused with a Laplacian, never dropped: populations, kernels and delays
(``Conv``, ``x(t - d)``), point processes and exposed sources, non-Gaussian
or time-only noise, the Stratonovich and Marcus rules, and the SDE simulator.
Everything the spatial pipeline refuses (UV-divergent subgraphs, periodic
loops, switch-on transients, ...) it refuses for these models as for the
hand-written ones.  Conserved noise (``div`` of a white vector noise) is not
supported yet.

Paper-style equations and per-source interpretations
----------------------------------------------------
:meth:`SDE.sde` reads equations written as in a paper, one per line
(``dx = (-mu*x) dt + sqrt(2*D) dW + c ∘ dN(lam)``; :mod:`api.sde_paper`), into
the declarations above, so the model is the one those calls build.  Each
noise term carries its interpretation: a plain ``dW`` / ``dN`` term is Ito, a
term marked with a circle (``∘``, ASCII ``o``) is Stratonovich for a Gaussian
source and Marcus for a jump source.  The same is available on any source as
``noise(..., interpretation=)``, which overrides :meth:`SDE.interpretation`
for that source.  The rules act per source (the Stratonovich drift of a
source involves that source alone, the Marcus flow of a jump source its own
coupling), so a model may mix them; it then builds as the explicit model
whose sources carry the same interpretations, and is stamped Marcus when any
source takes the Marcus rule.  Refused, as under the model-wide modes:
Stratonovich on a jump source, Stratonovich or Marcus on an exposed source or
an SPDE, and Marcus on a coupling that reads other units (a joint jump).
:meth:`SDE.sde_text` writes the equations back in that form.

Gaussian-jittered exogenous inputs
----------------------------------
``define_kernel('g', kernel=Kernel.gaussian('sigma', tol=1e-6))``
(:meth:`api.delay.Kernel.gaussian`) declares the Gaussian density, which is
acausal, for the jitter of an EXOGENOUS input: ``Conv(kX*g, n)`` with ``n`` a
point process (or a source declared with ``as_field=``) whose rate holds no
field.  Shifting a stationary exogenous input in time changes no stationary
statistic of the network, so the density is shifted by ``T`` and replaced by a
rational fit (``api._gaussian_fit``), and the shift is dropped.  The build
(:meth:`_check_jitter`) refuses a jitter on anything else, a source whose rate
depends on a field (feedback into the source), a use of the input without the
jitter (``Conv(kX, n)`` next to ``Conv(kX*g, n)``: the shift must be common to
every use) or with a different one, and an indexed jitter or sigma; the model
dict carries ``exogenous_jitter``, from which ``compute_cumulants`` refuses the
input as an external leg and switch-on runs, and records the fit
(``result['input_jitter']``, the ledger's ``jitter:<input>``).  The simulator
runs the fitted kernel: the recorded input leads the network by ``T``.

Kernels of nonlinear functions of the fields
--------------------------------------------
``Conv(K, X)`` with ``X`` nonlinear in the fields (``Conv(k, v^2/a)``,
``Conv(k, phi(v))``, in an equation or a point-process rate) is embedded as
auxiliary linear fields (:meth:`_embed_nonlinear_convs`): one first-order stage
``(b*Dt + a)*u = prev`` per linear factor ``a + b*(I*omega)`` of the kernel
image's denominator, a companion block per irreducible factor of higher degree,
fed by ``X``, the numerator read off the stages; ``Conv(K, X)`` becomes the
output.  The fields are named ``<kernel>_emb<n>_<stage>`` and listed in
``notes``.  This is exact (the kernel route cannot take such a term: it delays
one copy of a fluctuation leg of ``X`` where every leg of the product sits at
the same earlier time).  Refused by name: a delayed kernel, a kernel given only
by ``time_expr``, a non-rational image, and a kernel indexed by both the unit
and the summed population (one filter per pair).  An affine argument keeps the
kernel route.

Diagnostics: ``check``, ``latex`` and parse errors
--------------------------------------------------
Presentation only: none of these changes the action, the equations or the built
model.

:meth:`SDE.check` ``(k=2, max_ell=1, params=None)`` returns an
:class:`SDEReport` (a dict, printed unless ``verbose=False``): the declarations
(fields with their response fields, parameters, functions, kernels,
populations, operators); the derived action, split into one ``dynamics`` term
per field, one ``noise`` term per source (with the source's type and
parameters) and the ``extra_action`` terms; the mean-field equations; the
interpretation corrections, each labelled with its source (the Stratonovich ->
Ito drift terms, and the Marcus noise terms with their route and order); the
vertex table of the action expanded at the saddle (response and field leg
counts, coefficient, degree, and the loop orders the vertex can enter for
``k`` external legs: the degree count ``ell >= (degree - k)/2``, to which the
enumerator's causality rules add); the propagator structure (class, poles,
stability, saddle, linear operator and noise matrix, from the analyzer); and
warnings (a Marcus series or CGF truncation, terms that enter the mean field
only, the analyzer's warnings and errors).  No integration is done: the cost
is the build and the expansion of the action.  It reads
:meth:`action_text`, :meth:`equations` and the analyzer (:mod:`api.analyze`)
rather than deriving anything a second time.

:meth:`SDE.latex` returns ``{'sde', 'noise', 'action', 'mean_field'}``, LaTeX
strings from Sage's ``latex()`` (:mod:`api.sde_latex`): the equations as written
(``dx = (...) dt + c sqrt(var) dW`` with ``dN`` for jump sources, the operator
form ``(tau dt + 1) v = ...`` when the left side is not ``Dt x``, ``Dt`` typeset
as ``\\partial_t`` and the Laplacian as ``\\nabla^2``), the noise sources, the
derived action, and the stationary mean-field equations.  The output depends on
the declarations only and is the same on every run.  Nothing is compiled.

Parse errors: every expression the front end reads (equation sides, noise
coefficients and parameters, function bodies, kernels, point-process rates,
``extra_action``) is checked when it is declared.  A malformed one raises
:class:`SDEError` with the string, a caret (``^``) under the failing character,
the field or noise it belongs to, and a hint; an undeclared name suggests the
closest declared one, and an operator written inside a function argument or a
noise coefficient says so.  The exception types are unchanged (``SDEError`` is
a :class:`~api.noise.NoiseError` and a ``ValueError``).
"""
from __future__ import annotations

import keyword
import re
import warnings
from collections import OrderedDict
from typing import Optional

from sage.all import SR, integrate, oo, assume, assumptions, forget, function

from api.noise import (NoiseError, NoiseSource, THETA, parse_expr, Gaussian, GaussianVector,
                       expr_to_text, expr_names, _MATH_FUNCS, _SAGE_PRINTED_CALLS,
                       expression_syntax_problem, format_caret_error, parse_failure_message)
from api.operators import Laplacian, Operator, TimeDerivative
from api.sde_symbols import Symbol, as_name, as_text

__all__ = ['SDE', 'SDEError', 'SDEReport']


class SDEError(NoiseError):
    """Raised for an invalid SDE declaration."""


class _PaperText(str):
    """The text of :meth:`SDE.sde_text` (``noises``: the sources it names)."""
    noises: dict = {}


class SDEReport(dict):
    """The result of :meth:`SDE.check`: a dict with the keys documented there;
    ``str(report)`` / :meth:`text` is the printed form."""

    def text(self) -> str:
        return _report_text(self)

    __str__ = text


_MAX_VERTEX_ROWS = 24


def _clip(text, n: int = 160) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n - 1].rstrip() + f' ... [{len(text)} characters]'


def _report_text(r) -> str:
    L = []
    decl = r['declarations']
    L.append(f"SDE {r['name']!r}: {r['interpretation']}"
             + (f", space dimension {r['space']}" if r['space'] else ', time only'))
    L.append('')
    L.append('Declarations')
    L.append('  fields:     ' + ', '.join(
        f"{f['name']} (response {f['response']}"
        + (f", spike train of {f['exposed_source']}" if f['exposed_source'] else '') + ')'
        for f in decl['fields']))
    if decl['parameters']:
        L.append('  parameters: ' + ', '.join(
            p['name'] + ('' if p['default'] is None else f"={p['default']}")
            + (f" [{p['domain']}]" if p['domain'] else '') for p in decl['parameters']))
    for fn in decl['functions']:
        L.append(f"  function:   {fn['name']}({', '.join(fn['args'])}) = {fn['expression']}")
    for kn in decl['kernels']:
        if 'gaussian_jitter' in kn:
            # the fit's exact fraction runs to thousands of characters
            j = kn['gaussian_jitter']
            L.append(f"  kernel:     {kn['name']} "
                     + (f"= {' * '.join(j['factors'])}, " if j.get('factors') else '')
                     + f"Gaussian jitter (sigma={j['sigma']}, tol={j['tol']:g}): "
                     f"rational fit of {j['states']} states, shift {j['shift']} dropped"
                     + (f", delay={kn['delay']}" if 'delay' in kn else ''))
            continue
        L.append('  kernel:     ' + kn['name'] + ' ' + ', '.join(
            f'{q}={kn[q]}' for q in ('time_expr', 'freq_image', 'delay') if q in kn))
    L.append('  operators:  ' + ', '.join(f'{n}: {t}' for n, t in decl['operators'].items()))
    L.append('')
    L.append('Action  S = sum of')
    for t in r['action']['terms']:
        if t['kind'] == 'dynamics':
            L.append(f"  [dynamics {t['field']}]  {_clip(t['text'])}")
        elif t['kind'] == 'noise':
            st = t.get('source_type', {})
            ps = st.get('params')
            desc = st.get('type', '') + (' ' + ', '.join(f'{q}={v}' for q, v in ps.items())
                                         if ps else '')
            if st.get('jump'):
                desc += (f", jump={st['jump']['type']}("
                         + ', '.join(f'{q}={v}' for q, v in st['jump']['params'].items()) + ')')
            L.append(f"  [noise {t['source']}: {desc.strip()}]  {_clip(t['text'])}")
        else:
            L.append(f"  [extra]  {_clip(t['text'])}")
    L.append('')
    L.append('Mean-field equations (lhs at Dt = 0 equals rhs)')
    for e in r['mean_field']:
        L.append(f"  {e['field']}:  {_clip(e['lhs'])}  =  {_clip(e['rhs'])}")
    if r['corrections']:
        L.append('')
        L.append('Interpretation corrections')
        for c in r['corrections']:
            L.append(f"  [{c['source']}]  {c['meaning']}")
    v = r.get('vertices')
    L.append('')
    if v is None:
        L.append('Vertices: not available (' + str(r.get('analysis_error')) + ')')
    else:
        L.append(f"Vertices (Taylor order {v['taylor_order']}; loops listed for k={v['k']}, "
                 f"max_ell={v['max_ell']})")
        L.append('  resp  field  degree  kind              loops  monomial -> coefficient')
        for row in v['rows'][:_MAX_VERTEX_ROWS]:
            loops = ','.join(map(str, row['loops'])) or '-'
            L.append(f"  {row['response_legs']:>4}  {row['field_legs']:>5}  {row['degree']:>6}  "
                     f"{row['kind']:<16}  {loops:>5}  {row['monomial']} -> "
                     f"{_clip(row['coefficient'], 100)}")
        if len(v['rows']) > _MAX_VERTEX_ROWS:
            L.append(f"  ... {len(v['rows']) - _MAX_VERTEX_ROWS} more monomials "
                     f"(report['vertices']['rows'])")
    p = r.get('propagator')
    if p:
        L.append('')
        L.append('Propagator')
        L.append(f"  class {p['class']}, poles {p['n_poles']}"
                 + (f", stable {p['stable']}" if p['stable'] is not None else '')
                 + f", noise {p['noise_class']}")
        if p['saddle']:
            L.append('  saddle: ' + ', '.join(f'{n}={v_}' for n, v_ in p['saddle'].items()))
        if p['linear_operator']:
            L.append('  K(Dt) = ' + _clip(' + '.join(f'({c})*{m}'
                                                 for m, c in p['linear_operator'].items())))
        if p['noise_matrix']:
            L.append('  noise = ' + _clip(' + '.join(f'({c})*{m}'
                                               for m, c in p['noise_matrix'].items())))
    if r.get('warnings'):
        L.append('')
        L.append('Warnings')
        for w in r['warnings']:
            L.append(f"  {w['level']:<7} {w['code']}: {_clip(w['message'], 220)}")
    return '\n'.join(L)


_KEYWORDS = {'sum', 'for', 'in', 'Dt', 'Conv', 'I', 'pi', 'e',
             'if', 'else', 'range'} | set(_MATH_FUNCS)
_DT = SR.var('Dt')
_LAP = SR.var('Laplacian')
# the engine's own operator spellings: accepted in a model's text only under a
# declared operator of that name (``Dt`` is declared by default)
_CANONICAL_OPS = {'Dt', 'Laplacian', 'Lap'}
# names an operator may not take (besides the model's own declarations)
_NO_OPERATOR = ((_KEYWORDS | {'theta', 't', 'i', 'j', 'k', 'l', 'pop', 'pop_all', 'Dx', 'omega'}
                 | set(_SAGE_PRINTED_CALLS)) - _CANONICAL_OPS)
_NO_DECLARE = _KEYWORDS | {'theta', 't'}
_UNICODE_DT = ('∂t', '∂_t')
_NO_POPULATION = {'Dt', 'Conv', 'theta', 't'}


def _check_declarable(name, kind):
    """Refuse a declared name that is a built-in constant, keyword or math
    function.  ``e``, ``pi`` and ``I`` are the Sage constants inside noise
    parameters, so a declared ``e`` was read as Euler's number there but as
    the parameter elsewhere (a silently wrong action and saddle)."""
    bad = (name in _NO_POPULATION or keyword.iskeyword(name)) if kind == 'population' \
        else (name in _NO_DECLARE or keyword.iskeyword(name))
    if bad:
        raise SDEError(
            f'{kind} name {name!r} is reserved (built-in constant, keyword or '
            f'function); choose another name (e.g. {name}0).')


def _split_top_level(text: str, sep: str = ','):
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def _norm_at(text, index: int):
    """``text`` with its whitespace collapsed, and where character ``index`` of
    ``text`` falls in it."""
    text = str(text)
    head = text[:index]
    return ' '.join(text.split()), len(' '.join(head.split())) + (
        1 if head.strip() and head[-1:].isspace() else 0)


def _wrap(text) -> str:
    t = str(text).strip()
    return t if re.fullmatch(r'[\w\[\],.\s]*', t) else f'({t})'


def _strip_index(text: str) -> str:
    """``u[i]`` -> ``u``, ``phi[i](u[i])`` -> ``phi(u)``: the scalar text of a
    spatial model (one field per species, no population index)."""
    return re.sub(r'\b(\w+)\[i\]', r'\1', str(text))


def _compositions(n: int, parts: int):
    """The tuples of ``parts`` non-negative integers that sum to ``n``."""
    if parts == 1:
        yield (n,)
        return
    for first in range(n, -1, -1):
        for rest in _compositions(n - first, parts - 1):
            yield (first,) + rest


def _is_rational_in(e, t) -> bool:
    num, den = SR(e).numerator_denominator()
    return num.is_polynomial(t) and den.is_polynomial(t)


def _cgf_shapes(K):
    """``K(theta)`` as ``sum coefficient * shape(theta)`` for the exact Marcus
    route: ``([(kind, key, coefficient)], bases)``.  ``kind`` is ``'exp'``
    (``key = alpha``: ``e^{alpha theta}``), ``'poly'`` (``key = p``:
    ``theta^p``) or ``'generic'`` (``key`` the shape); constants are dropped
    (every difference kills them).  ``bases`` are the subexpressions of the
    generic shapes at which the closed form stops being the mgf (bases of
    negative or non-integer powers, log arguments): the mgf exists at ``s``
    while none of them changes sign or vanishes on ``[0, s]``."""
    import operator
    from sage.all import exp as _exp
    from sage.symbolic.operators import add_vararg, mul_vararg
    K0, K = SR(K), SR(K).expand()
    terms = K.operands() if K.operator() is add_vararg else [K]

    def shape(g):
        op = g.operator()
        if g is THETA or (op is None and g.has(THETA)):
            return 'poly', 1, SR(1)
        if op is operator.pow:
            base, ex = g.operands()
            if base.operator() is not None and str(base.operator()) == 'exp' and not ex.has(THETA):
                return shape(_exp(base.operands()[0] * ex))
            if (base - THETA).is_zero() and ex.is_integer() and bool(ex > 0):
                return 'poly', int(ex), SR(1)
        if op is not None and str(op) == 'exp':
            arg = g.operands()[0]
            if arg.is_polynomial(THETA) and arg.degree(THETA) == 1:
                return 'exp', arg.coefficient(THETA, 1), _exp(arg.subs({THETA: 0}))
        return 'generic', g, SR(1)

    groups = OrderedDict()
    for t in terms:
        if not t.has(THETA):
            continue
        facs = t.operands() if t.operator() is mul_vararg else [t]
        cf, g = SR(1), SR(1)
        for q in facs:
            if q.has(THETA):
                g *= q
            else:
                cf *= q
        kind, key, extra = shape(g)
        slot = groups.setdefault((kind, str(key)), [kind, key, SR(0)])
        slot[2] += cf * extra
    shapes = [(kind, key, cf) for kind, key, cf in groups.values() if not SR(cf).is_zero()]

    bases = []

    def walk(e):
        op = e.operator()
        if op is None:
            return
        ops = e.operands()
        if op is operator.pow:
            base, ex = ops
            if base.has(THETA) and not (ex.is_integer() and bool(ex >= 0)):
                if not any((base - b).is_zero() for b in bases):
                    bases.append(base)
        elif str(op) == 'log' and ops[0].has(THETA):
            if not any((ops[0] - b).is_zero() for b in bases):
                bases.append(ops[0])
        for q in ops:
            walk(q)
    if any(kind == 'generic' for kind, _, _ in shapes):
        # the unexpanded K keeps (1 - c theta)^(-2) as such: expanded, its base
        # is a square that touches 0 at the pole without changing sign
        walk(K0)
    for kind, key, _ in shapes:
        if kind == 'generic':
            walk(key)
    return shapes, bases


class SDE:
    """Builder for an SDE model; see the module docstring."""

    def __init__(self, name: str, n_populations: Optional[int] = None, *,
                 space: Optional[int] = None):
        self.name = name
        self._n_pop_default = int(n_populations) if n_populations else 1
        if space is not None:
            if isinstance(space, bool) or not str(space).isdigit() or int(space) < 1:
                raise SDEError(f'SDE({name!r}, space={space!r}): the spatial dimension must be '
                               f'an integer d >= 1 (omit space= for a time-only SDE).')
            space = int(space)
        self._space = space
        # declared operators, name -> Operator; ``Dt`` is declared by default
        # (``_dt_implicit``) and dropped when the time derivative is renamed
        self._operators: 'OrderedDict[str, Operator]' = OrderedDict([('Dt', TimeDerivative())])
        self._dt_implicit = True
        self._boundary: Optional[dict] = None
        self._dyson: Optional[dict] = None
        self._pops: list[dict] = []
        self._fields: 'OrderedDict[str, dict]' = OrderedDict()
        self._params: list[tuple[tuple, dict]] = []
        self._param_names: list[str] = []
        self._functions: list[tuple[tuple, dict]] = []
        self._function_names: list[str] = []
        self._kernels: 'OrderedDict[str, dict]' = OrderedDict()
        # kernels carrying a Gaussian jitter (Kernel.gaussian), name -> its
        # jitter record; composite kernels inherit it (_compose_kernels)
        self._jitter: dict[str, dict] = {}
        self._jitter_uses: dict[str, dict] = {}
        self._eqs: dict[str, dict] = {}
        self._sources: 'OrderedDict[str, dict]' = OrderedDict()
        self._interp = 'ito'
        self._stability: Optional[bool] = None
        self._extra: list[str] = []
        self._series_order = 8
        self._marcus_route = 'auto'
        self._marcus_terms: 'OrderedDict[str, dict]' = OrderedDict()
        # higher-order equations: field -> its velocity embedding (see equation())
        self._embedded: 'OrderedDict[str, dict]' = OrderedDict()
        self._ito_drift: dict = {}
        # what the last derivation emitted, by origin (read by check() and
        # latex(); the action text itself is unchanged by it)
        self._action_log: list[dict] = []
        self._drift_log: list[dict] = []
        self._mf_only_log: list[dict] = []
        self.notes: list[str] = []

    # ── declarations (passed through to TemporalModelBuilder) ─────────
    def population(self, name: str, *, size: int = 1, description: str = ''):
        """Declare a population (an index set of ``size`` units); returns its
        :class:`~api.sde_symbols.Symbol`."""
        _check_declarable(name, 'population')
        self._check_not_operator(name, 'population')
        self._pops.append({'name': name, 'size': max(int(size), 1),
                           'description': description})
        return Symbol(self, name, 'population')

    def field(self, name: str, *, population: Optional[str] = None,
              description: str = '', domain: Optional[str] = None,
              latex: str = ''):
        """Declare a physical field (its response field is ``<name>t``);
        returns its :class:`~api.sde_symbols.Symbol`."""
        self._check_new_name(name)
        _check_declarable(name, 'field')
        self._fields[name] = {'population': as_name(population), 'description': description,
                              'domain': domain, 'latex': latex, 'exposed': None}
        return Symbol(self, name, 'field')

    def parameter(self, name: str, default=None, indexed=False,
                  domain: Optional[str] = None, description: str = '',
                  indexed_by: Optional[list] = None):
        """Declare a parameter; returns its :class:`~api.sde_symbols.Symbol`."""
        if indexed_by is not None:
            indexed_by = [as_name(p) for p in indexed_by]
        _check_declarable(name, 'parameter')
        self._check_not_operator(name, 'parameter')
        if name in self._param_names:
            raise SDEError(f'parameter {name!r} is declared twice.')
        self._params.append(((name,), dict(default=default, indexed=indexed,
                                           domain=domain, description=description,
                                           indexed_by=indexed_by)))
        self._param_names.append(name)
        return Symbol(self, name, 'parameter')

    def define_function(self, name: str, args: list, expression: str,
                        latex: Optional[str] = None, description: str = '',
                        population: Optional[str] = None):
        """Declare the function ``name(args) = expression``; returns its
        :class:`~api.sde_symbols.Symbol` (``phi(v)`` is its call)."""
        args, expression, population = [as_name(a) for a in args], as_text(expression), \
            as_name(population)
        _check_declarable(name, 'function')
        self._check_not_operator(name, 'function')
        self._syntax_check(expression, f'define_function({name!r}) expression')
        if name in _SAGE_PRINTED_CALLS:
            # A call of it means Sage's own function in noise expressions and the saddle solve.
            raise SDEError(f'function name {name!r} is reserved (Sage\'s {name}); choose another '
                           f'name (e.g. {name}0).')
        self._functions.append(((name, list(args), expression),
                                dict(latex=latex, description=description,
                                     population=population)))
        self._function_names.append(name)
        return Symbol(self, name, 'function')

    def define_kernel(self, name: str, *, time_expr: Optional[str] = None,
                      freq_image: Optional[str] = None, latex_name: str = '',
                      indexed=False, indexed_by: Optional[list] = None,
                      delay: Optional[str] = None, kernel=None):
        """A convolution kernel ``Conv(name, x)`` (see
        :meth:`api.model.TemporalModelBuilder.define_kernel`): ``time_expr`` or
        ``freq_image``, optionally delayed by ``delay`` (``g(t - d)``; the hard
        delay ``delta(t - d)`` alone), or ``kernel=Kernel.delay('d') *
        Kernel.exponential('tau')`` (:class:`api.delay.Kernel`).  Returns its
        :class:`~api.sde_symbols.Symbol`."""
        time_expr, freq_image, delay = as_text(time_expr), as_text(freq_image), as_text(delay)
        _check_declarable(name, 'kernel')
        self._check_not_operator(name, 'kernel')
        for what, expr in (('time_expr', time_expr), ('freq_image', freq_image), ('delay', delay)):
            if isinstance(expr, str):
                self._syntax_check(expr, f'define_kernel({name!r}) {what}')
        if kernel is not None:
            if time_expr is not None or freq_image is not None or delay is not None:
                raise SDEError(f'define_kernel({name!r}): give kernel= alone.')
            freq_image, delay = kernel.freq_image, kernel.delay
            if getattr(kernel, 'jitter', None):
                if indexed or indexed_by:
                    raise SDEError(f'define_kernel({name!r}): a Gaussian jitter kernel is '
                                   f'not indexed (one common time shift of the input); '
                                   f'index the synaptic kernel it multiplies in Conv '
                                   f'instead (Conv(kX[i]*{name}, n)).')
                if '[' in kernel.jitter['sigma']:
                    raise SDEError(f"define_kernel({name!r}): Kernel.gaussian sigma="
                                   f"{kernel.jitter['sigma']!r} is indexed; the jitter of "
                                   f"an input takes one sigma (one common time shift).")
                self._jitter[name] = dict(kernel.jitter)
        elif name in self._jitter:
            del self._jitter[name]
        if time_expr is not None and delay is None:
            from api.delay import split_shifted_time_expr
            sh = split_shifted_time_expr(str(time_expr))
            if sh is not None:
                delay, time_expr = sh
        if delay is not None and time_expr is None and freq_image is None:
            freq_image = '1'
        if time_expr is None and freq_image is None:
            raise SDEError(f'define_kernel({name!r}): give time_expr=, freq_image= '
                           f'or delay=.')
        self._kernels[name] = dict(time_expr=time_expr, freq_image=freq_image,
                                   latex_name=latex_name, indexed=indexed,
                                   indexed_by=indexed_by,
                                   delay=None if delay is None else str(delay))
        return Symbol(self, name, 'kernel')

    def stability_analysis(self, enabled: bool):
        self._stability = bool(enabled)
        return self

    def series_order(self, order: int):
        """Taylor order used in the action for CGFs with a removable
        singularity at 0 (the uniform jump mgf), and the order of the Marcus
        series of a jump source with a state-dependent coupling
        (``interpretation('marcus')``; for an affine coupling, the number of
        response legs its exact noise term is kept to).  Default 8."""
        self._series_order = int(order)
        return self

    def marcus_route(self, route: str):
        """How ``interpretation('marcus')`` builds the noise term of a jump
        source with a state-dependent coupling.  ``'auto'`` (default): exact
        for an affine coupling whose flow is closed form (each coupling affine
        in its own field), from the jump law's mgf, else the moment series of
        ``sum_k kappa_k/k! D_k``.  ``'series'``: the moment series always
        (it converges fast for a small slope times jump size, where the exact
        coefficients can lose digits to cancellation)."""
        route = str(route).lower()
        if route not in ('auto', 'series'):
            raise SDEError(f"marcus_route({route!r}): use 'auto' or 'series'.")
        self._marcus_route = route
        return self

    # ── operators and space ───────────────────────────────────────────
    def operator(self, name: str, op):
        """Declare the operator ``op`` (:mod:`api.operators`) under ``name``.

        ``m.operator('D_t', TimeDerivative())`` names the time derivative ``D_t``
        (it replaces the default ``Dt``, which is then no longer accepted);
        ``m.operator('lap', Laplacian())`` declares the spatial Laplacian, which
        makes the model an SPDE (``SDE(name, space=d)``).  An operator is
        written in product style (``(tau*Dt + 1)*x``, ``D*lap*u``) or call style
        (``Dt(x)``, ``lap(u)``, ``lap(u^2)``).  Declare an operator before the
        equations that use it.  Its name must be a Python identifier that no
        field, response field, parameter, function, kernel or population of the
        model takes; ``Dt`` is reserved for the time derivative (``∂t`` names it
        too), ``Laplacian`` and ``Lap`` for the Laplacian, and the paper-style
        increments (``dt``, ``dW``, ``dN``, ``dW_<name>``, ``d<X>_<name>``)
        cannot be operators.  Returns its :class:`~api.sde_symbols.Symbol`
        (``lap(u)`` and ``(tau*D_t + 1)*x`` are then Sage expressions)."""
        if isinstance(op, type) and issubclass(op, Operator):
            op = op()
        if str(name) in _UNICODE_DT and isinstance(op, TimeDerivative):
            name = 'Dt'                 # ∂t: the default time derivative's own spelling
        if not isinstance(op, Operator):
            raise SDEError(f'operator({name!r}): give an operator from api.operators '
                           f'(TimeDerivative(), Laplacian()); got {type(op).__name__}.')
        name = str(name)
        if name == 'dt' or re.fullmatch(r'd[WN]|d[A-Z]_\w+', name):
            raise SDEError(
                f'operator({name!r}): {name!r} is an increment of the paper-style syntax '
                f'(dx = f dt + g dW + c dN(rate), SDE.sde), so it cannot name an operator; '
                f'the time derivative is Dt (the default, also spelled ∂t in equations), and '
                f'another operator takes another name (e.g. lap, D_t).')
        if not name.isidentifier() or keyword.iskeyword(name) or '__' in name:
            raise SDEError(f'operator({name!r}): an operator name must be a Python identifier '
                           f'(e.g. dt, lap, Δ) without a double underscore.')
        for cls in (TimeDerivative, Laplacian):
            if name in cls.canonical_names and not isinstance(op, cls):
                raise SDEError(f'operator({name!r}): {name!r} is reserved for the '
                               f'{cls.__name__}; choose another name.')
        if name in self._operators:
            if name == 'Dt' and self._dt_implicit and isinstance(op, TimeDerivative):
                self._dt_implicit = False          # the default Dt, now explicit
                return Symbol(self, name, 'operator')
            raise SDEError(f'operator({name!r}): an operator with this name is declared.')
        if name in _NO_OPERATOR:
            raise SDEError(f'operator({name!r}): {name!r} is reserved (built-in constant, '
                           f'keyword, index or function); choose another name.')
        taken = self._name_owner(name)
        if taken:
            raise SDEError(f'operator({name!r}): {name!r} is already the {taken}; an operator '
                           f'name must not collide with fields, parameters or functions.')
        if isinstance(op, TimeDerivative) and self._dt_implicit and 'Dt' in self._operators:
            del self._operators['Dt']              # the time derivative is renamed
        self._dt_implicit = False if isinstance(op, TimeDerivative) else self._dt_implicit
        self._operators[name] = op
        return Symbol(self, name, 'operator')

    def boundary(self, mode: str, **params):
        """The spatial boundary condition of an SPDE (``'infinite'``, the
        default, or ``'periodic'`` with ``length=``); see
        :meth:`api.model.SpatialModelBuilder.boundary`."""
        self._boundary = {'mode': mode, **params}
        return self

    def dyson(self, mode: str = 'fixed', order: Optional[int] = None):
        """The Dyson truncation of a coupled SPDE with unequal diffusion; see
        :meth:`api.model.SpatialModelBuilder.dyson`."""
        self._dyson = {'mode': mode, 'order': order}
        return self

    def _name_owner(self, name: str) -> str:
        """What ``name`` already is in this model (``''`` if free): a field,
        a field's response / fluctuation / saddle, a parameter, a function, a
        kernel, a population or an exposed source."""
        for f in self._fields:
            for n, what in ((f, 'field'), (f + 't', 'response field'),
                            ('d' + f, 'fluctuation'), (f + 'star', 'mean-field saddle')):
                if n == name:
                    return f'{what} of {f!r}' if n != f else f'field {f!r}'
        if name in self._param_names:
            return f'parameter {name!r}'
        if name in self._function_names:
            return f'function {name!r}'
        if name in self._kernels:
            return f'kernel {name!r}'
        if name in {p['name'] for p in self._pops}:
            return f'population {name!r}'
        return ''

    def _check_not_operator(self, name: str, kind: str):
        """Refuse a declaration named like a declared operator (any order of
        declaration is caught: :meth:`operator` checks the other way)."""
        if name in self._operators:
            raise SDEError(f'{kind} name {name!r} is a declared operator '
                           f'({type(self._operators[name]).__name__}); choose another name.')
        for f in ([name] if kind in ('field', 'exposed field', 'point process') else []):
            for n in (f + 't', 'd' + f, f + 'star'):
                if n in self._operators:
                    raise SDEError(f'{kind} {f!r}: its derived name {n!r} is a declared '
                                   f'operator; rename the operator or the {kind}.')
                if n in _CANONICAL_OPS:
                    # (a field D after renaming Dt: its response field would be
                    # the engine's time-derivative symbol)
                    raise SDEError(f'{kind} {f!r}: its derived name {n!r} is the '
                                   f"engine's operator symbol; rename the {kind}.")

    def _check_no_operator(self, text: str, where: str):
        """Refuse an operator in a text that is not an equation (a function
        body, a noise parameter or coefficient): a declared name, or an engine
        spelling (``Dt`` renamed, ``Laplacian`` / ``Lap``) the model does not
        declare.  The default ``Dt`` is left to the checks it always had."""
        self._undeclared_canonical(text, where)
        for n in self._operators:
            if not (n == 'Dt' and self._dt_implicit) and self._has_token(text, n):
                shown = ' '.join(str(text).split())
                raise SDEError(format_caret_error(
                    shown, max(self._token_at(shown, n), 0),
                    f'{where} uses the operator {n!r}; operators act in the equations '
                    f'only.',
                    f'move the operator to the equation\'s lhs/rhs, e.g. '
                    f'{n}(x) or {n}*x, not inside a function argument or a coefficient'))

    # ── dynamics ──────────────────────────────────────────────────────
    def equation(self, field: str, lhs: str, rhs: str = '0', *, velocity=None):
        """``lhs = rhs + noise`` for ``field``; ``lhs`` holds the linear
        differential operator (contains the time derivative, ``Dt`` unless it
        is renamed with :meth:`operator`), ``rhs`` no noise.  Declared operators
        are written in product style (``(tau*Dt + 1)*x``, ``D*lap*u``) or call
        style (``Dt(x)``, ``lap(u)``).  ``field`` may be its symbol, ``lhs`` and
        ``rhs`` Sage expressions.

        A higher-order lhs, a polynomial in the time derivative acting on the
        field (``(m*Dt^2 + gamma*Dt + k)*x``), is embedded as a first-order
        system: the velocities ``xdot`` (``xddot``, ...; or the names given in
        ``velocity=``) are declared as fields, ``Dt*x = xdot``, and the top
        equation ``m*Dt*xdot + gamma*xdot = rhs - k*x`` takes the noise coupled
        into ``x`` (``noise(..., into={'x': c})`` is the force ``c*eta``)."""
        field = as_name(field)
        lhs, rhs = (v if isinstance(v, str) else str(as_text(v)) for v in (lhs, rhs))
        lhs, rhs = self._unicode_dt(lhs), self._unicode_dt(rhs)
        if field not in self._fields:
            raise SDEError(f'equation({field!r}): declare the field first with .field().')
        if field in self._eqs:
            raise SDEError(f'equation({field!r}): the field already has an equation.')
        self._syntax_check(lhs, f'equation({field!r}) lhs')
        self._syntax_check(rhs, f'equation({field!r}) rhs')
        tnames = self._time_names()
        self._undeclared_canonical(lhs, f'equation({field!r}) lhs')
        self._undeclared_canonical(rhs, f'equation({field!r}) rhs')
        if not any(self._has_token(lhs, n) for n in tnames):
            raise SDEError(f'equation({field!r}): lhs={lhs!r} must contain the time '
                           f'derivative {" or ".join(tnames) or "Dt"} (a dynamical equation).')
        for n in tnames:
            if self._has_token(rhs, n):
                shown = ' '.join(str(rhs).split())
                at = max(self._token_at(shown, n), 0)
                in_call = re.search(r'\b(?!' + re.escape(n) + r'\b)[A-Za-z_]\w*\s*\([^()]*$',
                                    shown[:at]) is not None
                raise SDEError(format_caret_error(
                    shown, at, f'equation({field!r}): {n} found in rhs; it belongs on the lhs.',
                    (f'{n} is an operator and cannot be the argument of a function; operators '
                     f'act on fields, e.g. {n}(x)' if in_call else
                     f'the time derivative belongs on the lhs: lhs={n}({field}) or '
                     f"lhs='{n}*{field}'")))
        if (velocity is not None or re.search(r'\^|\*\*', lhs)
                or sum(len(re.findall(rf'(?<!\w){re.escape(n)}(?!\w)', lhs)) for n in tnames) > 1):
            order = self._lhs_orders(field, lhs)
            if order is not None:
                return self._embed(field, lhs, rhs, order, velocity)
            if velocity is not None:
                raise SDEError(f'equation({field!r}): velocity= names the embedded velocities of a '
                               f'higher-order lhs; {lhs!r} is first order.')
        rhs = self._delayed_fields(rhs, field)
        rhs = self._compose_kernels(rhs, f'equation({field!r})')
        if re.search(r'\bt\b', rhs) or re.search(r'\bt\b', lhs):
            raise SDEError(f'equation({field!r}): explicit time t is not supported '
                           f'(time-independent dynamics only; a delayed field is '
                           f'written x(t - d)).')
        # ``lhs`` / ``rhs``: the engine's spelling (the time derivative as
        # ``Dt``; refreshed by _derive); ``*_raw``: as declared
        self._eqs[field] = {'lhs': self._translate(lhs, 'v1', where=f'equation({field!r}) lhs'),
                            'rhs': self._translate(rhs, 'v1', where=f'equation({field!r}) rhs'),
                            'lhs_raw': lhs, 'rhs_raw': rhs}
        return self

    def _unicode_dt(self, text: str) -> str:
        """``∂t`` / ``∂_t`` -> the model's time derivative (``Dt`` unless renamed)."""
        if '∂' not in text:
            return text
        t0 = (self._time_names() or ['Dt'])[0]
        return re.sub(r'∂_?t(?!\w)', t0, text)

    def _lhs_orders(self, field: str, lhs: str):
        """The coefficients ``[c_0, ..., c_N]`` of ``lhs = sum_n c_n Dt^n x``
        when its order ``N`` in the time derivative is above 1 (``None`` for a
        first-order lhs).  A higher-order lhs must be such a polynomial, with
        coefficients free of the fields."""
        L = self._parse(self._translate(lhs, 'v1', where=f'equation({field!r}) lhs'),
                        f'lhs of {field}').expand()
        try:
            N = int(L.degree(_DT))
        except (TypeError, ValueError):
            N = 0
        if N <= 1:
            return None
        x = self._sym(field)
        cs, rest = [], L
        for n in range(N + 1):
            an = L.coefficient(_DT, n) if n else L.subs({_DT: 0})
            c = an.coefficient(x, 1)
            cs.append(c)
            rest = rest - c * _DT ** n * x
        if not rest.expand().is_zero() or any(self._field_dependent(c) for c in cs):
            raise SDEError(f'equation({field!r}): the order-{N} lhs {lhs!r} must be a polynomial in '
                           f'the time derivative acting on {field} alone, with coefficients free of '
                           f'the fields (e.g. (m*Dt^2 + gamma*Dt + k)*{field}); move other terms to '
                           f'the rhs.')
        if cs[-1].is_zero():
            raise SDEError(f'equation({field!r}): cannot read the leading coefficient of {lhs!r}.')
        return cs

    def _embed(self, field, lhs, rhs, cs, velocity):
        """Declare the first-order system of a higher-order equation (see
        :meth:`equation`)."""
        N = len(cs) - 1
        if velocity is None:
            vel = [field + 'd' * j + 'ot' for j in range(1, N)]
        else:
            vel = [as_name(v) for v in ([velocity] if isinstance(velocity, (str, Symbol))
                                        else list(velocity))]
            if len(vel) != N - 1:
                raise SDEError(f'equation({field!r}): an order-{N} lhs needs {N - 1} velocity '
                               f'name(s); got velocity={velocity!r}.')
        for v in vel:
            if v in self._fields or self._name_owner(v):
                raise SDEError(f'equation({field!r}): the velocity name {v!r} is taken '
                               f'({self._name_owner(v) or "a field"}); give velocity=<free name>.')
        names = [field] + vel
        info = self._fields[field]
        for k, v in enumerate(vel):
            self.field(v, population=info['population'],
                       description=f'd{"^" + str(k + 1) if k else ""}{field}/dt'
                                   f'{"^" + str(k + 1) if k else ""} (embedded velocity)')
            self._fields[v]['embedded_of'] = field
        t0 = (self._time_names() or ['Dt'])[0]
        self._embedded[field] = {'lhs_raw': lhs, 'rhs_raw': rhs, 'fields': vel, 'top': vel[-1],
                                 'order': N}
        for j in range(N - 1):
            self.equation(names[j], f'{t0}*{names[j]}', names[j + 1])
        top, cN, cdamp = vel[-1], cs[N], cs[N - 1]
        lead = f'{t0}*{top}' if (cN - 1).is_zero() else f'{_wrap(expr_to_text(cN))}*{t0}*{top}'
        if not cdamp.is_zero():
            lead += ' + ' + (top if (cdamp - 1).is_zero() else f'{_wrap(expr_to_text(cdamp))}*{top}')
        rest = sum((cs[n] * SR.var(names[n]) for n in range(N - 1)), SR(0))
        if rest.is_zero():
            rhs_top = rhs
        elif str(rhs).strip() in ('', '0'):
            rhs_top = expr_to_text(-rest)
        else:
            rt = expr_to_text(rest)
            from sage.symbolic.operators import add_vararg
            import operator as _op
            summed = rest.operator() in (_op.add, add_vararg) or rt.startswith('-')
            rhs_top = f'{rhs} - ({rt})' if summed else f'{rhs} - {rt}'
        self.equation(top, lead, rhs_top)
        # the noise coupled into the field forces its top derivative
        for s in self._sources.values():
            if field in s['couples']:
                s['couples'] = {top if f == field else f: c for f, c in s['couples'].items()}
        return self

    def _delayed_fields(self, text: str, where: str) -> str:
        """``x(t - d)`` (``x`` a declared field, ``d`` a parameter expression)
        -> ``Conv(delay_<d>, x)`` with the hard-delay kernel ``delay_<d>``
        declared on the fly (:mod:`api.delay`)."""
        if not self._fields:
            return text
        names = '|'.join(re.escape(f) for f in sorted(self._fields, key=len, reverse=True))
        ind = re.search(rf'\b({names})\s*\[[^\]]*\]\s*\(\s*t\b', text)
        if ind:
            raise SDEError(f'equation({where!r}): an indexed delayed field '
                           f'{text[ind.start():ind.end()]}...) is not supported; write '
                           f"Conv(K, {ind.group(1)}[...]) with define_kernel('K', delay='d').")
        # Sage prints x(t - d) as x(-d + t) (the call of a field symbol)
        text = re.sub(rf'\b({names})\s*\(\s*-\s*([^()]*?)\s*\+\s*t\s*\)', r'\1(t - \2)', text)
        pat = re.compile(rf'\b({names})\s*\(\s*t\s*-\s*')
        out, pos = '', 0
        while True:
            m = pat.search(text, pos)
            if not m:
                break
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            if depth:
                raise SDEError(f'equation({where!r}): unbalanced parentheses in '
                               f'{text[m.start():]!r}.')
            dexpr = text[m.end():k - 1].strip()
            if not dexpr or re.search(r'\bt\b', dexpr):
                raise SDEError(f'equation({where!r}): cannot read the delay in '
                               f'{text[m.start():k]!r} (write x(t - d), d free of t).')
            self._check_text(dexpr, f'equation({where!r}) delay {dexpr!r}')
            kname = 'delay_' + (re.sub(r'\W+', '_', dexpr).strip('_') or 'd')
            have = self._kernels.get(kname)
            if have is None:
                if kname in self._allowed_names():
                    raise SDEError(f'equation({where!r}): the delay kernel name '
                                   f'{kname!r} is taken; declare the kernel with '
                                   f"define_kernel(..., delay={dexpr!r}).")
                self.define_kernel(kname, delay=dexpr)
            elif have.get('delay') != dexpr or have.get('freq_image') != '1':
                raise SDEError(f'equation({where!r}): kernel {kname!r} exists and is '
                               f'not the delay {dexpr!r}.')
            out += text[pos:m.start()] + f'Conv({kname}, {m.group(1)})'
            pos = k
        return out + text[pos:]

    def drift(self, field: str, f: str):
        """Shorthand for ``equation(field, lhs='Dt*field', rhs=f)`` (with the
        declared name of the time derivative)."""
        field = as_name(field)
        tnames = self._time_names()
        return self.equation(field, lhs=f'{tnames[0] if tnames else "Dt"}*{field}', rhs=f)

    def sde(self, text: str, *, noises: Optional[dict] = None):
        """Declare equations written as in a paper, one per line::

            m.sde('''
                dx = (-mu*x - eps*x^3) dt + sqrt(2*D) dW
                tau*dv = (E - v) dt + sigma ∘ dW_a + J dN(r*exp(v))
            ''')

        The left side is ``d<field>`` (with an optional constant factor); each
        term on the right ends in ``dt`` (the drift), ``dW`` (a standard Wiener
        process), ``dN(rate)`` (a Poisson counting process) or ``d<X>_<name>``
        for a source passed in ``noises={name: source}``; the factor in front
        is the coupling.  A subscript names a process, and one name in several
        equations is one source.  A circle before the differential (``∘ dW``,
        ASCII ``o dW``) marks a Stratonovich term (Gaussian) or a Marcus term
        (jump); a plain term is Ito.  The operator form puts a linear operator
        on the left and the right side per unit time, with noise terms
        ``c dW/dt``: ``(tau*Dt + 1)*v = E + s dW/dt``,
        ``(m*Dt^2 + gamma*Dt + k)*x = sqrt(2*gamma*T) dW/dt``.  Each line
        becomes the existing declarations (:meth:`field` if new, :meth:`drift` or
        :meth:`equation`, :meth:`noise` with ``into=`` and
        ``interpretation=``), so the model is the one those calls build; see
        :mod:`api.sde_paper`."""
        from api.sde_paper import apply_paper
        return apply_paper(self, text, noises)

    def sde_text(self) -> str:
        """The equations written as :meth:`sde` reads them (a ``str``; its
        ``noises`` attribute holds the sources to pass as ``noises=``).
        ``SDE(name)`` with the same parameters, functions and kernels, then
        ``.sde(text, noises=text.noises)``, rebuilds the same model.  Refused
        for point processes and exposed sources, and for velocities named with
        ``velocity=`` (a left side other than ``a*Dt*x`` is written in the
        operator form)."""
        from api.sde_paper import paper_text
        text, noises = paper_text(self)
        out = _PaperText(text)
        out.noises = noises
        return out

    def noise(self, name: str, source: NoiseSource, couples: Optional[dict] = None,
              *, expose: Optional[str] = None, domain: Optional[str] = None,
              population: Optional[str] = None, into: Optional[dict] = None,
              interpretation: Optional[str] = None, as_field: Optional[str] = None):
        """Declare a noise source ``name`` feeding ``into={field: c}``: ``c`` is
        the coupling COEFFICIENT of the noise in that field's equation
        (``dx_field += c * d eta``), not a population index; fields and
        coefficients may be symbols and Sage expressions.  ``as_field='n'`` exposes it as the field ``n``
        (use ``n`` in the equations' rhs) and returns the symbol of ``n``;
        otherwise the builder is returned.  An SPDE takes space-time white
        Gaussian noise, ``Gaussian(var=..., space='white')``.
        ``interpretation=`` (``'ito'``, ``'stratonovich'`` or ``'marcus'``)
        sets the interpretation of this source alone, in place of the model's
        :meth:`interpretation`.  ``couples=`` and ``expose=`` are the former
        names of ``into=`` and ``as_field=`` (deprecated, same meaning)."""
        if couples is not None:
            warnings.warn("noise(couples=...) is deprecated: use into=... (the same "
                          "meaning).", DeprecationWarning, stacklevel=2)
        if expose is not None:
            warnings.warn("noise(expose=...) is deprecated: use as_field=... (the same "
                          "meaning).", DeprecationWarning, stacklevel=2)
            if as_field is not None:
                raise SDEError(f'noise({name!r}): give as_field= or expose= (the same), not both.')
        if as_field is not None:
            expose = as_field
        expose, population = as_name(expose), as_name(population)
        if interpretation is not None:
            interpretation = str(interpretation).lower()
            if interpretation not in ('ito', 'stratonovich', 'marcus'):
                raise SDEError(f"noise({name!r}, interpretation={interpretation!r}): use "
                               f"'ito', 'stratonovich' or 'marcus'.")
        if into is not None:
            if couples is not None:
                raise SDEError(f'noise({name!r}): give into= or couples= (the same), not both.')
            couples = into
        if not isinstance(source, NoiseSource):
            raise SDEError(
                f'noise({name!r}): source must be a noise source from api.noise '
                f'(Gaussian, Poisson, CompoundPoisson, GammaProcess, '
                f'InverseGaussianProcess, Cumulants, CGF); got {type(source).__name__}. '
                f'Per-step distributions are not processes.')
        if name in self._sources:
            raise SDEError(f'noise({name!r}): a source with this name exists.')
        couples = {as_name(f): ([as_text(e) for e in c] if isinstance(c, (list, tuple))
                                else as_text(c)) for f, c in dict(couples or {}).items()}
        # (a field with a higher-order equation is forced through its top derivative)
        couples = {self._embedded[f]['top'] if f in self._embedded else f: c
                   for f, c in couples.items()}
        for f, c in couples.items():
            if isinstance(c, str):
                self._syntax_check(c, f'noise({name!r}) coefficient for {f!r}')
        if not couples and expose is None:
            raise SDEError(f'noise({name!r}): give into={{field: coefficient}} '
                           f'or as_field=<field name>.')
        if getattr(source, 'kernel', None) is not None:
            return self._colored_noise(name, source, couples, expose, population, interpretation)
        if isinstance(source, GaussianVector):
            if expose is not None:
                raise SDEError(f'noise({name!r}): a GaussianVector is coupled with into=, not exposed.')
            for f, c in couples.items():
                if not isinstance(c, (list, tuple)) or len(c) != source.n:
                    raise SDEError(f'noise({name!r}): GaussianVector has {source.n} components; '
                                   f'give into={{{f!r}: [c_1, ..., c_{source.n}]}} (one coefficient '
                                   f'per component); got {c!r}.')
                for k, e in enumerate(c):
                    if isinstance(e, str):
                        self._syntax_check(e, f'noise({name!r}) coefficient {k + 1} for {f!r}')
        else:
            for f, c in couples.items():
                if isinstance(c, (list, tuple)):
                    raise SDEError(f'noise({name!r}): a list of coefficients for {f!r} needs a '
                                   f'GaussianVector source (one coefficient per component).')
        if expose is not None:
            self._check_new_name(expose)
            _check_declarable(expose, 'exposed field')
            self._fields[expose] = {'population': population, 'description':
                                    f'exposed noise source {name}', 'domain': domain,
                                    'latex': '', 'exposed': name}
        self._sources[name] = {'source': source, 'couples': couples,
                               'expose': expose, 'population': population}
        if interpretation is not None:
            self._sources[name]['interp'] = interpretation
        return self if expose is None else Symbol(self, expose, 'field')

    def _colored_noise(self, name, source, couples, as_field, population, interpretation):
        """Coloured Gaussian noise ``eta = h * xi`` (``Gaussian(var, kernel=h)``):
        the Markov embedding of the filter.  With ``H(s) = N(s)/D(s)``
        (``s = I*omega``), the field ``z`` (``as_field=``, default the source
        name) obeys ``D(Dt) z = xi`` (a higher-order ``D`` is embedded with
        velocities), ``eta = N(Dt) z`` is a combination of ``z`` and its
        velocities, and ``c*eta`` is added to the rhs of every field in
        ``into=``.  The white source ``xi`` keeps the name ``name``; a proper
        (not strictly proper) image adds its direct part ``N_n/D_n * xi`` to
        the fields as an ordinary white coupling."""
        from sage.all import I as _I
        kern = source.kernel
        z = as_field or name
        where = f'noise({name!r}, Gaussian(kernel=...))'
        if self._field_dependent_text(expr_to_text(source.params['var'])):
            raise SDEError(f'{where}: var must be free of the fields (the variance of the white '
                           f'noise the kernel filters); put a state-dependent coupling in into=.')
        H = parse_expr(kern.freq_image, what=f'{where} image')
        sv = SR.var('s__k')
        Hs = H.subs({SR.var('omega'): -_I * sv}).rational_simplify()
        Np, Dp = Hs.numerator().expand(), Hs.denominator().expand()
        if not (Np.is_polynomial(sv) and Dp.is_polynomial(sv)) or \
                set(map(str, Hs.variables())) & {'omega', 't'}:
            raise SDEError(f'{where}: the image {kern.freq_image!r} is not rational in I*omega; only '
                           f'rational kernels have a finite Markov embedding.')
        n, m = int(Dp.degree(sv)), int(Np.degree(sv))
        if m > n:
            raise SDEError(f'{where}: the image {kern.freq_image!r} grows with omega (an improper '
                           f'kernel, a derivative of white noise); the numerator degree must not '
                           f'exceed the denominator degree.')
        if n < 1:
            raise SDEError(f'{where}: the image {kern.freq_image!r} is a constant, which is white '
                           f'noise; drop kernel=.')
        dn = Dp.coefficient(sv, n)
        direct = (Np.coefficient(sv, n) / dn).simplify_full() if m == n else SR(0)
        Nr = (Np - direct * Dp).expand()
        t0 = (self._time_names() or ['Dt'])[0]
        Dop = expr_to_text(Dp.subs({sv: SR.var(t0)}))
        pop = population
        if pop is None:
            pops = {self._fields[f]['population'] for f in couples if f in self._fields}
            pop = pops.pop() if len(pops) == 1 else None
        self.field(z, population=pop, description=f'coloured noise state of {name}')
        self._fields[z]['colored_of'] = name
        self.equation(z, f'({Dop})*{z}', '0')
        states = [z] + (self._embedded[z]['fields'] if z in self._embedded else [])
        out = sum((Nr.coefficient(sv, k) * SR.var(states[k]) for k in range(n)
                   if not Nr.coefficient(sv, k).is_zero()), SR(0))
        white = Gaussian(var=source.params['var'], mean=source.params['mean'])
        self.noise(name, white, into={z: 1}, population=pop, interpretation=interpretation)
        s = self._sources[name]
        if not direct.is_zero():
            for f, c in couples.items():
                s['couples'][f] = expr_to_text(parse_expr(c) * direct)
        s['colored'] = {'into': dict(couples), 'field': z, 'output': expr_to_text(out),
                        'kernel': kern, 'states': states}
        return Symbol(self, z, 'field')

    def _field_dependent_text(self, text) -> bool:
        names = expr_names(parse_expr(str(text)))
        return bool(names & set(self._fields))

    def point_process(self, name: str, rate: str, *, population: Optional[str] = None,
                      size='1', domain: Optional[str] = None, description: str = ''):
        """Declare the spike train (counting process) ``name`` with
        conditional intensity ``rate``: ``P(spike of name[i] in dt) =
        rate_i dt``.  The field ``name`` (and its response ``<name>t``) is
        declared, the action gains ``nt*n - (exp(size*nt) - 1)*rate`` and the
        mean field ``n* = size*rate*`` (kernels at omega = 0).  ``rate`` is an
        expression like a drift: fields, parameters, declared functions
        (``phi[i](v[i])``), population sums (``sum(w[i,j]*Conv(g[i,j], n[j])
        for j in E)``) and convolutions ``Conv(K, x)`` of ANY field, the spike
        train ``name`` itself included (a spike-only Hawkes process); a
        product of declared kernels inside ``Conv`` (``Conv(h[i]*g[i,j],
        n[j])``) is their convolution.  ``size`` is the jump of ``name`` at a
        spike (a deterministic mark, default 1).  A spike train enters a rate
        only through ``Conv`` (its instantaneous value is a delta).  Returns
        the symbol of the spike train ``name``."""
        rate, size, population = as_text(rate), as_text(size), as_name(population)
        self._check_new_name(name)
        _check_declarable(name, 'point process')
        if name in self._sources:
            raise SDEError(f'point_process({name!r}): a source with this name exists.')
        rate = ' '.join(str(rate).split())
        if not rate:
            raise SDEError(f'point_process({name!r}): empty rate.')
        self._syntax_check(rate, f'point_process({name!r}) rate')
        self._syntax_check(size, f'point_process({name!r}) size')
        if re.search(r'\bDt\b', rate) or re.search(r'\bt\b', rate):
            raise SDEError(f'point_process({name!r}): the rate may not contain Dt or '
                           f'the time t (a delayed field is Conv(K, x) with '
                           f"define_kernel('K', delay='d')).")
        for n in self._operators:
            if self._has_token(rate, n):
                raise SDEError(f'point_process({name!r}): the rate may not contain the '
                               f'operator {n!r}.')
        self._fields[name] = {'population': population, 'description':
                              description or f'point process {name}',
                              'domain': domain, 'latex': '', 'exposed': name}
        self._sources[name] = {'source': None, 'couples': {}, 'expose': name,
                               'population': population,
                               'pp': {'rate': rate, 'size': str(size).strip() or '1'}}
        return Symbol(self, name, 'field')

    def interpretation(self, mode: str):
        """``'ito'`` (default), ``'stratonovich'`` (Gaussian sources) or
        ``'marcus'`` (Gaussian and jump sources; see the module docstring).
        It applies to the sources that do not set their own
        (``noise(..., interpretation=)``); the terms of :meth:`sde` carry
        their own, so it is refused for a model with paper-style terms."""
        mode = str(mode).lower()
        if mode not in ('ito', 'stratonovich', 'marcus'):
            raise SDEError(f"interpretation({mode!r}): use 'ito', 'stratonovich' or "
                           f"'marcus'.")
        paper = [n for n, s in self._sources.items() if s.get('paper')]
        if paper and mode != 'ito':
            raise SDEError(f"interpretation({mode!r}): the noise terms written with .sde() "
                           f"({', '.join(paper)}) carry their own interpretation: a plain "
                           f"dW / dN term is Ito, a term marked with a circle (∘dW, or 'o dW') "
                           f"is Stratonovich (Gaussian) or Marcus (jump).")
        self._interp = mode
        return self

    def _src_interp(self, name: str) -> str:
        """The interpretation of source ``name``: its own
        (``noise(..., interpretation=)``, a paper-style term) or the model's."""
        return self._sources[name].get('interp') or self._interp

    def _interp_mode(self) -> str:
        """The model's interpretation as one word: the model's own when no
        source sets one; else the model-wide mode the sources' interpretations
        amount to (Gaussian sources Stratonovich and jump sources Ito is
        ``'stratonovich'`` only without jumps), or ``'mixed'``."""
        own = [n for n, s in self._sources.items() if s.get('interp')]
        if not own:
            return self._interp
        for mode in ('ito', 'stratonovich', 'marcus'):
            if all(self._effective(n, mode) == self._effective(n, self._src_interp(n))
                   for n, s in self._sources.items()):
                return mode
        return 'mixed'

    def _effective(self, name: str, mode: str) -> str:
        """What ``mode`` does to source ``name`` (the rule its terms get):
        ``'ito'``, ``'stratonovich'`` (a Gaussian source, the Stratonovich
        conversion) or ``'marcus'`` (a jump source, the Marcus flow); a mode
        the source cannot take is returned as is (refused at derivation)."""
        s = self._sources[name]
        if mode == 'ito':
            return 'ito'
        src = s.get('source')
        gaussian = src is not None and src.is_gaussian
        if mode == 'marcus':
            return 'stratonovich' if gaussian else 'marcus'
        return 'stratonovich' if gaussian else 'stratonovich-jump'

    def _marcus_sources(self) -> list:
        """The sources the Marcus rule applies to (their own or the model's)."""
        return [n for n in self._sources if self._src_interp(n) == 'marcus']

    def extra_action(self, text: str):
        """Append ``text`` to the derived action (written with its own
        ``sum(... for i in pop)`` wrapper if it is indexed)."""
        text = as_text(text)
        self._syntax_check(text, 'extra_action')
        self._extra.append(text)
        return self

    def set_action_text(self, text: str):
        raise SDEError(
            'SDE derives the full action from the declared equations and noise '
            'sources; append terms with extra_action(...), or replace the action '
            'on the builder returned by to_builder().')

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _syntax_check(text, where: str):
        """Refuse an expression string that is not an expression, naming
        ``where`` it was declared and marking the failing character."""
        if isinstance(text, str) and expression_syntax_problem(' '.join(text.split())):
            raise SDEError(parse_failure_message(text, where))

    def _check_new_name(self, name):
        if not re.fullmatch(r'[A-Za-z]\w*', name) or '__' in name:
            raise SDEError(f'{name!r} is not a valid field name.')
        if name in self._fields:
            raise SDEError(f'field {name!r} is declared twice.')
        if name in ('Dt', 't', 'theta', 'i', 'j'):
            raise SDEError(f'{name!r} is a reserved name.')
        self._check_not_operator(name, 'field')

    def _populations(self) -> list[dict]:
        if self._pops:
            return self._pops
        return [{'name': self._implicit_pop(), 'size': self._n_pop_default,
                 'description': ''}]

    def _implicit_pop(self) -> str:
        # The compiler binds ``pop`` to range(number of populations), which is
        # range(1) once a population is declared, so an implicit population of
        # size N > 1 gets its own name and ``pop`` in the texts is renamed.
        return 'pop' if self._n_pop_default == 1 else 'pop_all'

    def _rename_pop(self, text: str) -> str:
        if self._pops or self._implicit_pop() == 'pop':
            return text
        return re.sub(r'\bpop\b', self._implicit_pop(), text)

    def _pop_of_field(self, f: str) -> str:
        p = self._fields[f]['population']
        pops = self._populations()
        if p is None:
            if len(pops) > 1:
                raise SDEError(f'field {f!r}: give population= (several populations '
                               f'are declared).')
            return pops[0]['name']
        if p not in [q['name'] for q in pops]:
            raise SDEError(f'field {f!r}: unknown population {p!r}.')
        return p

    def _pop_of_source(self, name: str) -> str:
        s = self._sources[name]
        pops = set()
        if s['expose']:
            pops.add(self._pop_of_field(s['expose']))
        for f in s['couples']:
            if f not in self._fields:
                raise SDEError(f'noise({name!r}): couples to undeclared field {f!r}.')
            pops.add(self._pop_of_field(f))
        if s['population'] is not None:
            pops.add(s['population'])
        if len(pops) != 1:
            raise SDEError(f'noise({name!r}): a source must couple to fields of ONE '
                           f'population (got {sorted(pops)}).')
        return pops.pop()

    def _allowed_names(self) -> set:
        names = set(self._fields) | {f + 't' for f in self._fields}
        names |= set(self._param_names) | set(self._function_names) | set(self._kernels)
        names |= {p['name'] for p in self._populations()} | {'pop'}
        names |= {'i', 'j', 'k', 'l'} | _KEYWORDS
        names |= {o.v1_symbol for o in self._operators.values()}
        return names

    # ── operator translation ──────────────────────────────────────────
    def _time_names(self) -> list:
        return [n for n, o in self._operators.items() if isinstance(o, TimeDerivative)]

    @staticmethod
    def _has_token(text: str, name: str) -> bool:
        return re.search(rf'(?<!\w){re.escape(name)}(?!\w)', str(text)) is not None

    def _undeclared_canonical(self, text: str, where: str):
        """Refuse ``Dt`` / ``Laplacian`` / ``Lap`` in a text when the model does
        not declare that operator name (the time derivative renamed; a
        Laplacian not declared).  ``Laplacian`` / ``Lap`` declared as something
        else (a parameter) stay that."""
        for c in sorted(_CANONICAL_OPS - set(self._operators)):
            if not self._has_token(text, c) or self._name_owner(c):
                continue
            if c == 'Dt':
                raise SDEError(f'{where}: Dt is not declared in this model; its time '
                               f'derivative is named {" or ".join(self._time_names())}.')
            lap = [n for n, o in self._operators.items() if isinstance(o, Laplacian)]
            raise SDEError(f'{where}: {c} is not a declared operator of this model; '
                           + (f'its Laplacian is named {" or ".join(lap)}.' if lap else
                              f"declare it with .operator({c!r}, Laplacian()) "
                              f"(an SPDE: SDE(name, space=d))."))

    def _op_pattern(self):
        names = sorted(self._operators, key=len, reverse=True)
        return re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, names)) + r')(?!\w)')

    def _translate(self, text, style: str = 'v1', info: Optional[dict] = None,
                   where: str = 'expression') -> str:
        """``text`` with the declared operator names in the engine's spelling.

        ``style='v1'``: the plain multiplicative symbols (``Dt``, ``Laplacian``):
        product style ``D_t*x`` -> ``Dt*x``, call style ``D_t(x)`` -> ``Dt*x``
        (parenthesised where precedence needs it).  ``style='ir'``: the spatial
        operator IR calls (``Dt(u)``, ``Lap(u^2)``); a product-style use is
        left as a placeholder for :meth:`_ir_text`.  ``info`` collects the
        operator kinds used (``'used'``), whether a call acts on a nonlinear
        expression (``'nonlinear'``) and whether product style occurs.  A model
        that uses only ``Dt`` in product style gets ``text`` back unchanged."""
        info = {} if info is None else info
        text = str(text)
        self._undeclared_canonical(text, where)
        pat = self._op_pattern()
        out, pos = '', 0
        for m in pat.finditer(text):
            if m.start() < pos:
                continue
            name = m.group(1)
            op = self._operators[name]
            info.setdefault('used', set()).add(type(op))
            j = m.end()
            while j < len(text) and text[j] == ' ':
                j += 1
            if j < len(text) and text[j] == '(':                     # call style
                depth, k = 0, j
                while k < len(text):
                    depth += {'(': 1, ')': -1, '[': 1, ']': -1}.get(text[k], 0)
                    k += 1
                    if depth == 0:
                        break
                if depth:
                    raise SDEError(f'{where}: unbalanced parentheses after {name}(.')
                arg_raw = text[j + 1:k - 1]
                if not arg_raw.strip() or len(_split_top_level(arg_raw)) != 1:
                    shown, at = _norm_at(text, m.start())
                    raise SDEError(format_caret_error(
                        shown, at, f'{where}: {name}(...) takes one argument; got '
                        f'{text[m.start():k]!r}', f'write {name}(u) for one field or '
                        f'expression u'))
                arg_v1 = self._translate(arg_raw, 'v1', info, where).strip()
                try:
                    linear = self._linear_in_fields(arg_v1, f'{where}: {name}({arg_raw.strip()})')
                except SDEError as exc:
                    shown, at = _norm_at(text, m.start())
                    raise SDEError(format_caret_error(shown, at, str(exc))) from exc
                if not linear:
                    if not op.nonlinear_argument:
                        raise SDEError(
                            f'{where}: {name}({arg_raw.strip()}) acts on a nonlinear '
                            f'expression; the {type(op).__name__} '
                            f'acts linearly on fields here (e.g. {name}(x), '
                            f'{name}(a*x + b*y)).')
                    info['nonlinear'] = True
                if style == 'ir':
                    rep = f'{op.ir_call}({self._translate(arg_raw, "ir", info, where).strip()})'
                else:
                    # an atom or a power of one (phi, x[i], phi^2) needs no parentheses
                    tight = re.fullmatch(r'[\w\[\],.]+(\^(\w+|\([^()]*\)))?', arg_v1)
                    rep = f'{op.v1_symbol}*{arg_v1 if tight else _wrap(arg_v1)}'
                    before = text[:m.start()].rstrip()[-1:]
                    after = text[k:].lstrip()[:1]
                    if before not in ('', '+', '-', '(', '*', ',') or \
                            after not in ('', '+', '-', ')', '*', '/', ','):
                        rep = f'({rep})'
                out += text[pos:m.start()] + rep
                pos = k
            else:                                                     # product style
                self._refuse_operator_argument(text, m, name, where)
                info['product'] = True
                rep = op.v1_symbol if style == 'v1' else f'_opprod_{op.ir_call}'
                out += text[pos:m.start()] + rep
                pos = m.end()
        return out + text[pos:]

    def _refuse_operator_argument(self, text: str, m, name: str, where: str):
        """An operator written bare inside the argument of a function
        (``exp(lap)``, ``sin(dt)``): operators act on fields."""
        depth = 0
        for k in range(m.start() - 1, -1, -1):
            if text[k] in ')]':
                depth += 1
            elif text[k] in '([':
                if depth == 0:
                    fn = re.search(r'([A-Za-z_]\w*)\s*$', text[:k])
                    if (text[k] == '(' and fn and fn.group(1) not in self._operators
                            and fn.group(1) not in ('sum', 'Conv')):
                        shown, at = _norm_at(text, m.start())
                        raise SDEError(format_caret_error(
                            shown, at,
                            f'{where}: the operator {name!r} is the argument of the function '
                            f'{fn.group(1)!r}',
                            f'operators act on fields, not inside a function: write '
                            f'{name}(u) for a field u (or {name}*u), outside {fn.group(1)}(...)'))
                    return
                depth -= 1

    def _field_syms(self) -> list:
        return [SR.var(f + '__i') for f in self._fields]

    def _linear_in_fields(self, text_v1: str, where: str) -> bool:
        """True when ``text_v1`` is homogeneous of degree one in the fields
        (``u``, ``a*u + b*v``, ``Laplacian*u``): an operator argument the plain
        multiplicative form represents exactly."""
        try:
            E = parse_expr(self._index(text_v1), what=where)
        except NoiseError as exc:
            raise SDEError(f'{where}: cannot read the operator argument ({exc}).') from exc
        xs = [x for x in self._field_syms() if E.has(x)]
        if not xs:
            # (Dt(mu), lap(c): zero, or a symbol the engine would misread)
            raise SDEError(f'{where} acts on no field; an operator acts on the '
                           f"model's fields (e.g. lap(u), lap(u^2)).")
        lam = SR.var('_lin_scale')
        return bool((E.subs({x: lam * x for x in xs}) - lam * E).expand()
                    .simplify_full().is_zero())

    def _ir_text(self, raw: str, where: str) -> str:
        """``raw`` in the spatial operator IR form (``Dt(u)``, ``Lap(u)``,
        ``Lap(u^2)``), unindexed; a product-style use (``(dt + mu - D*lap)*u``)
        is distributed onto its field (``Dt(u) + mu*u - D*Lap(u)``)."""
        text = self._translate(raw, 'ir', where=where)
        if '_opprod_' not in text:
            return text
        from sage.symbolic.operators import add_vararg
        P = {SR.var('_opprod_Dt'): 'Dt', SR.var('_opprod_Lap'): 'Lap'}
        E = parse_expr(self._index(text), what=where).expand()
        out = SR(0)
        for t in (E.operands() if E.operator() is add_vararg else [E]):
            pw = {p: t.degree(p) if t.is_polynomial(p) else -1 for p in P}
            if all(v == 0 for v in pw.values()):
                out += t
                continue
            c = t.subs({p: 1 for p in P})
            xs = [x for x in self._field_syms() if c.has(x)]
            if (min(pw.values()) < 0 or len(xs) != 1 or not c.is_polynomial(xs[0])
                    or c.degree(xs[0]) != 1):
                raise SDEError(
                    f'{where}: a product-style operator multiplies {expr_to_text(c)}; it '
                    f'must multiply exactly one field linearly (write the call style, '
                    f'e.g. lap(u^2), for an operator acting on an expression).')
            base = xs[0]
            for p, call in P.items():
                for _ in range(pw[p]):
                    base = function(call)(base)
            out += c.coefficient(xs[0], 1) * base
        return _strip_index(expr_to_text(out))

    def _check_v1_operator_terms(self, text_indexed: str, where: str):
        """Spatial plain form: every term holding ``Dt`` or ``Laplacian`` is one
        field times a field-free coefficient (``D*Laplacian*u``); a product-style
        operator in front of a nonlinear term (``lap*u^2``) is refused."""
        from sage.symbolic.operators import add_vararg
        E = parse_expr(text_indexed, what=where).expand()
        for t in (E.operands() if E.operator() is add_vararg else [E]):
            if not (t.has(_DT) or t.has(_LAP)):
                continue
            c = t.subs({_DT: 1, _LAP: 1})
            xs = [x for x in self._field_syms() if c.has(x)]
            if (len(xs) != 1 or not c.is_polynomial(xs[0]) or c.degree(xs[0]) != 1
                    or not (t.is_polynomial(_DT) and t.is_polynomial(_LAP))):
                shown = expr_to_text(t)
                op = next((o for o in ('Laplacian', 'Dt') if self._has_token(shown, o)), 'Laplacian')
                raise SDEError(format_caret_error(
                    shown, max(self._token_at(shown, op), 0),
                    f'{where}: the operator term {shown} does not act on one field linearly; '
                    f'write an operator acting on an expression in call style (e.g. lap(u^2)).',
                    'an operator acts on one field, outside any function: lap(u) or '
                    'D*lap(u)'))

    def _check_text(self, text: str, what: str, display: Optional[str] = None):
        """Refuse a symbol of ``text`` that is not declared.  ``display``: the
        text as the user wrote it (``text`` has the operators in the engine's
        spelling), where the caret is placed."""
        toks = set(re.findall(r'\b([A-Za-z_]\w*)\b', text))
        allowed = self._allowed_names()
        unknown = sorted(toks - allowed)
        if unknown:
            shown = ' '.join(str(text if display is None else display).split())
            first = min(unknown, key=lambda u: (self._token_at(shown, u), u))
            at = self._token_at(shown, first)
            if at < 0:
                shown, at = ' '.join(str(text).split()), self._token_at(' '.join(str(text).split()), first)
            more = [u for u in unknown if u != first]
            raise SDEError(format_caret_error(
                shown, max(at, 0),
                f'{what}: undeclared symbol {first!r}'
                + (f' (also undeclared: {more})' if more else ''),
                self._name_hint(first, allowed, shown, at)))

    def _source_error(self, name, src, allowed, exc) -> SDEError:
        """The undeclared symbol of a noise source's parameter, with a caret
        under it in that parameter's text."""
        names = set(allowed) | self._allowed_names()
        for k, v in (getattr(src, 'params', None) or {}).items():
            text = expr_to_text(v)
            unknown = [u for u in re.findall(r'\b([A-Za-z_]\w*)\b', text)
                       if u not in names and u not in ('theta',)]
            if unknown:
                u = unknown[0]
                at = self._token_at(text, u)
                return SDEError(format_caret_error(
                    text, max(at, 0),
                    f'noise({name!r}) {type(src).__name__}({k}=...): undeclared symbol {u!r}',
                    self._name_hint(u, names, text, at)))
        return SDEError(f'noise({name!r}): {exc}')

    @staticmethod
    def _token_at(text: str, name: str) -> int:
        m = re.search(rf'(?<![\w.]){re.escape(name)}(?![\w])', text)
        return m.start() if m else -1

    def _name_hint(self, name: str, allowed, text: str = '', at: int = -1) -> str:
        import difflib
        pool = sorted(a for a in allowed if not a.endswith('__i') and a not in _KEYWORDS
                      and a not in ('pop', 'pop_all', 'i', 'j', 'k', 'l'))
        close = difflib.get_close_matches(name, pool, n=1, cutoff=0.6)
        if close:
            return f'did you mean {close[0]!r}? (otherwise declare {name!r})'
        called = at >= 0 and re.match(r'\s*\(', text[at + len(name):]) is not None
        if called:
            return (f"declare the function with .define_function({name!r}, args=[...], "
                    f"expression=...) (a math function is exp, log, sqrt, sin, ...)")
        return (f'declare {name!r} with .parameter({name!r}, default=...) or .field({name!r}); '
                f'every symbol must be a declared field, parameter, function, kernel or index')

    def _index(self, text) -> str:
        """Bare field names -> ``x[i]``; bare function calls -> ``phi[i](``;
        bare vector parameters -> ``p[i]``."""
        text = expr_to_text(text) if not isinstance(text, str) else text
        text = self._rename_pop(text)
        for f in self._fields:
            text = re.sub(rf'\b{re.escape(f)}\b(?!\s*[\[\(])', f'{f}[i]', text)
        for fn in self._function_names:
            text = re.sub(rf'\b{re.escape(fn)}\s*\(', f'{fn}[i](', text)
        for pn in self._vector_params():
            text = re.sub(rf'\b{re.escape(pn)}\b(?!\s*[\[\(])', f'{pn}[i]', text)
        return text

    def _vector_params(self):
        """Parameters indexed by one population (a bare name means ``p[i]``)."""
        out = []
        for (name,), kw in self._params:
            ib = kw.get('indexed_by')
            if (ib is not None and len(ib) == 1) or (ib is None and kw.get('indexed') in (True, 'vector')):
                out.append(name)
        return out

    def _parse(self, text, what):
        try:
            return parse_expr(self._index(text), what=what)
        except NoiseError as exc:
            if not isinstance(exc, SDEError):
                # point at the user's own text, not the indexed one
                raw = ' '.join(str(text).split())
                if raw and expression_syntax_problem(raw):
                    raise SDEError(parse_failure_message(raw, what)) from exc
                raise SDEError(str(exc)) from exc
            raise

    def _sym(self, f: str):
        return SR.var(f'{f}__i')

    def _kernel_dc(self, name: str):
        """The kernel's integral (its frequency image at omega = 0)."""
        k = self._kernels[name]
        if k['freq_image'] is not None:
            return parse_expr(k['freq_image'], what=f'kernel {name}').subs(
                {SR.var('omega'): 0})
        e = parse_expr(k['time_expr'], what=f'kernel {name}')
        t = SR.var('t')
        dd = SR.var('_dirac_')
        e = e.subs({function('dirac_delta')(t): dd})
        from sage.all import dirac_delta
        e = e.subs({dirac_delta(t): dd})
        delta_part = e.coefficient(dd, 1)
        rest = (e - delta_part * dd).simplify_full()
        delta_raw = delta_part
        # ``(c*dirac_delta(t) + h(t))*heaviside(t)``: the delta of a causal kernel
        # sits on the boundary and counts with its full weight c (the kernel's
        # frequency image is c + h~(omega), as for the same kernel written
        # ``c*dirac_delta(t) + h(t)*heaviside(t)``), so the step is not carried
        # into the delta part; a t-dependent factor is evaluated at the delta (t = 0).
        from sage.all import heaviside as _H
        delta_part = delta_part.subs({_H(t): SR(1)}).subs({t: SR(0)})
        # assume only what is not assumed already, and forget only that: a
        # parameter that a model builder declared positive keeps its global
        # assumption (forgetting it leaked into every later computation)
        # (a positivity already implied by a stronger held assumption, e.g.
        # taug > 1, is 'redundant' to Sage: skipped too; every assume sits in
        # the try, so a failure part-way still forgets what was added)
        held = {str(a) for a in assumptions()}
        assumed = []
        # A retarded kernel carries heaviside(t): Sage/Maxima does not always evaluate the integral over (-oo, oo)
        # with it (the result can be an unevaluated ``cases``), so integrate its t > 0 part over (0, oo) instead.
        from sage.all import heaviside
        lower = -oo
        causal = e - delta_raw * dd             # unsimplified: simplify_full mangles sums of exponentials
        if rest.has(heaviside(t)):
            rest = rest.subs({heaviside(t): SR(1)})
            lower = SR(0)
        val = None
        try:
            for v in rest.variables():
                if str(v) == 't' or str(v > 0) in held:
                    continue
                try:
                    assume(v > 0)
                except ValueError:                 # redundant
                    continue
                assumed.append(v > 0)
            try:
                val = integrate(rest, t, lower, oo) if not rest.is_zero() else SR(0)
            except (ValueError, TypeError):
                # Maxima cannot order the rates of e.g. a difference of exponentials
                # ('requested additional constraints'): the structural closed form below
                val = None
        finally:
            for a in assumed:
                forget(a)
        if val is None or 'integrate' in str(val):
            val = self._kernel_dc_causal(name, causal, t)
        return (val + delta_part).simplify_full()

    def _kernel_dc_causal(self, name, causal, t):
        """The integral of a causal sum of ``t^n exp(-a t)`` terms by the closed form
        ``n!/a^(n+1)`` (``model_compiler.causal_exponential_ft`` at omega = 0), the
        decay rates being positive by declaration (``domain='positive'``); no
        assumption is made on any symbol."""
        from api.model_compiler import causal_exponential_ft
        declared = {p[0][0] for p in self._params if p[1].get('domain') == 'positive'}
        positive = {str(v) for v in causal.variables()
                    if str(v).split('__')[0] in declared}
        try:
            return SR(causal_exponential_ft(causal, t, SR(0), positive=positive))
        except ValueError as exc:
            raise SDEError(f'kernel {name}: cannot integrate time_expr symbolically '
                           f'for the mean-field equations ({exc}); declare '
                           f"domain='positive' on its time constants, or give freq_image=.") from exc

    def _collapse_kernels(self, text: str) -> str:
        """Replace kernel symbols by their integral (stationary mean field)."""
        if not self._kernels:
            return text
        # Conv(K, X) -> (K)*(X)
        while True:
            m = re.search(r'\bConv\s*\(', text)
            if not m:
                break
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            args = _split_top_level(text[m.end():k - 1])
            if len(args) != 2:
                raise SDEError(f'cannot read Conv(...) in {text!r}.')
            text = text[:m.start()] + f'({args[0].strip()})*({args[1].strip()})' + text[k:]
        for name in self._kernels:
            dc = self._kernel_dc(name)

            def rep(m, _dc=dc):
                val = expr_to_text(_dc)
                if m.group(1):
                    idx = [s.strip() for s in m.group(1).split(',')]
                    mp = dict(zip(('i', 'j'), idx))
                    # one simultaneous pass inside each [...] (a sequential i->j, j->i
                    # rename would cancel a swap)
                    val = re.sub(r'\[([^\]]*)\]', lambda q: '[' + re.sub(
                        r'\b([ij])\b', lambda r: mp.get(r.group(1), r.group(1)), q.group(1)) + ']', val)
                return f'({val})'
            text = re.sub(rf'\b{re.escape(name)}\b(?:\s*\[([^\]]*)\])?', rep, text)
        return text

    def _check_pp(self, name, pp):
        what = f'point_process({name!r}) rate'
        rate = self._compose_kernels(pp['rate'], f'point_process({name!r})')
        self._check_text(rate, what)
        self._check_text(pp['size'], f'point_process({name!r}) size')
        spikes = {f for f, info in self._fields.items() if info['exposed'] is not None}
        if re.search(r'\b(' + '|'.join(map(re.escape, sorted(spikes))) + r')\b',
                     pp['size']):
            raise SDEError(f'point_process({name!r}): size may not depend on a spike train.')
        bare = set(re.findall(r'\b([A-Za-z_]\w*)\b', self._strip_conv(rate))) & spikes
        if bare:
            raise SDEError(
                f'{what}: the spike train(s) {sorted(bare)} appear outside Conv(...); a '
                f'spike train enters a rate only filtered, Conv(K, {sorted(bare)[0]}).')

    @staticmethod
    def _conv_spans(text):
        """``(start, open_paren_end, close_end)`` of each top-level ``Conv(``."""
        out, pos = [], 0
        while True:
            m = re.compile(r'\bConv\s*\(').search(text, pos)
            if not m:
                return out
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            if depth:
                raise SDEError(f'unbalanced parentheses in {text!r}.')
            out.append((m.start(), m.end(), k))
            pos = k

    # ── Conv of a nonlinear function of the fields: Markov embedding ──
    def _embed_nonlinear_convs(self):
        """``Conv(K, X)`` with ``X`` a NONLINEAR function of the fields
        (``Conv(k, v^2/a)``, ``Conv(k, phi(v))``) is realized as auxiliary
        linear fields (the Markov embedding): a chain of first-order stages
        ``(b*Dt + a)*u = prev`` per linear factor ``a + b z`` of the kernel's
        denominator (``z = i omega``), a companion block per irreducible factor
        of higher degree, fed by ``X``; ``Conv(K, X)`` becomes the chain's
        output.  The engine's kernel route cannot take such a term: it
        convolves ONE copy of a fluctuation leg of ``X`` (``x (g * x)``) where
        every leg of ``X`` sits at the same earlier time.  Idempotent: the
        previous embedding is undone first (fields and texts restored), so
        declarations made after a derivation are seen."""
        prev = getattr(self, '_conv_embedding', None)
        if prev:
            for f in prev['fields']:
                self._fields.pop(f, None)
                self._eqs.pop(f, None)
            for f, txt in prev['eq_rhs'].items():
                if f in self._eqs:
                    self._eqs[f]['rhs_raw'] = txt
            for n, txt in prev['pp_rate'].items():
                if n in self._sources:
                    self._sources[n]['pp']['rate'] = txt
        emb = {'fields': [], 'eq_rhs': {}, 'pp_rate': {}, 'cache': {}, 'records': []}
        self._conv_embedding = emb
        if (self._space is not None or not self._kernels
                or any(op.spatial for op in self._operators.values())):
            return
        for f in [f for f in self._eqs if f not in emb['fields']]:
            eq = self._eqs[f]
            new = self._embed_text(eq['rhs_raw'], self._pop_of_field(f), f'equation({f!r})')
            if new != eq['rhs_raw']:
                emb['eq_rhs'][f] = eq['rhs_raw']
                eq['rhs_raw'] = new
        for name, src in list(self._sources.items()):
            if not src.get('pp'):
                continue
            where = f'point_process({name!r})'
            rate = self._compose_kernels(src['pp']['rate'], where)
            new = self._embed_text(rate, self._pop_of_field(src['expose']), where)
            if new != rate:
                emb['pp_rate'][name] = src['pp']['rate']
                src['pp']['rate'] = new
        for r in emb['records']:
            self.notes.append(
                f"Conv({r['kernel']}, {r['arg']}): a kernel of a nonlinear function of the "
                f"fields, embedded as the auxiliary linear field(s) {', '.join(r['fields'])} "
                f"(Markov embedding of the kernel image {r['image']}).")

    def _embed_text(self, text, pop, where):
        """``text`` with every ``Conv(K, X)`` of a nonlinear ``X`` replaced by
        the output of its embedding (``_embed_nonlinear_convs``)."""
        out, pos = '', 0
        for a, b, c in self._conv_spans(text):
            args = _split_top_level(text[b:c - 1])
            if len(args) != 2:
                continue
            kref, X = args[0].strip(), args[1].strip()
            m = re.fullmatch(r'([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?', kref)
            if not m or m.group(1) not in self._kernels or m.group(1) in self._jitter:
                # (a Gaussian jitter is never embedded: _check_jitter decides it)
                continue
            X = self._embed_text(X, pop, where)
            if self._affine_conv_arg(X, where):
                if X != args[1].strip():
                    out += text[pos:a] + f'Conv({kref}, {X})'
                    pos = c
                continue
            idx = [q.strip() for q in m.group(2).split(',')] if m.group(2) else []
            # the units X reads (a declared function's own index, phi[i](...), is
            # not one: the function is the same for every unit)
            reads = re.sub(r'\b[A-Za-z_]\w*\s*\[[^\]]*\]\s*\(', '(', self._index(X))
            letters = set(idx) | {q.strip() for grp in re.findall(r'\[([^\]]*)\]', reads)
                                  for q in grp.split(',')}
            sub_pop, rename = pop, False
            if letters <= {'i'}:
                pass
            elif letters == {'j'}:
                mj = re.search(r'\bfor\s+j\s+in\s+([A-Za-z_]\w*)', text[c:])
                if not mj:
                    raise SDEError(f'{where}: Conv({kref}, {X}) reads the index j outside a '
                                   f'population sum.')
                sub_pop, rename = mj.group(1), True
            else:
                raise SDEError(
                    f'{where}: Conv({kref}, {X}) convolves a nonlinear function of the fields '
                    f'with a kernel indexed by {sorted(letters)}: the kernel route cannot take a '
                    f'nonlinear argument and the automatic Markov embedding takes kernels '
                    f'indexed by one population only; write it as an auxiliary linear field '
                    f'per pair (declare the filter states with .field / .equation).')
            ij = (lambda t: re.sub(r'\[([^\]]*)\]', lambda q: '[' + re.sub(
                r'\bj\b', 'i', q.group(1)) + ']', t)) if rename else (lambda t: t)
            out_txt = self._embed_one(m.group(1), [('i' if q == 'j' else q) for q in idx]
                                      if rename else idx, ij(self._index(X)), sub_pop,
                                      f'{where}: Conv({kref}, {X})')
            if rename:
                out_txt = re.sub(r'\[([^\]]*)\]', lambda q: '[' + re.sub(
                    r'\bi\b', 'j', q.group(1)) + ']', out_txt)
            out += text[pos:a] + f'({out_txt})'
            pos = c
        return out + text[pos:]

    def _affine_conv_arg(self, X, where) -> bool:
        """True when ``X`` is affine in the fields (an inner ``Conv`` counts as
        a field); True too when ``X`` cannot be read here (a population sum):
        the engine refuses it by name if it is nonlinear."""
        inner, pos, txt = 0, 0, ''
        for a, _b, c in self._conv_spans(X):
            txt += X[pos:a] + f' _convleaf{inner}_ '
            inner += 1
            pos = c
        txt += X[pos:]
        try:
            E = parse_expr(self._index(txt), what=where)
        except NoiseError:
            return True
        fields = set(self._fields)
        xs = [v for v in E.variables()
              if str(v).split('__')[0] in fields or str(v).startswith('_convleaf')]
        if not xs:
            return True
        lam = SR.var('_conv_affine_scale')
        Es = E.subs({x: lam * x for x in xs})
        try:
            return bool(SR(Es.diff(lam, 2)).simplify_full().is_zero())
        except Exception:
            return False

    def _embed_one(self, kname, idx, X, pop, where):
        """The auxiliary fields of ``Conv(kname[idx], X)`` (``X`` indexed,
        letters ``i`` only) in population ``pop``; returns the text of the
        output (``i`` letters).  Cached per (kernel, indices, X, population)."""
        emb = self._conv_embedding
        key = (kname, tuple(idx), X, pop)
        if key in emb['cache']:
            return emb['cache'][key]
        k = self._kernels[kname]
        if k.get('delay'):
            raise SDEError(f'{where}: a delayed kernel of a nonlinear function of the fields '
                           f'is not embedded automatically; declare the filter as auxiliary '
                           f'fields and delay their output (Conv(Kernel.delay(d), u)).')
        if k.get('freq_image') is None:
            raise SDEError(f'{where}: a kernel of a nonlinear function of the fields is '
                           f'embedded as auxiliary linear fields from its rational frequency '
                           f'image; kernel {kname!r} has only a time_expr (give freq_image= or '
                           f'kernel=).')
        own = ['i', 'j'][:len(k.get('indexed_by') or [])]
        mp = dict(zip(own, idx))
        img = re.sub(r'\[([^\]]*)\]', lambda q: '[' + re.sub(
            r'\b([ij])\b', lambda r: '\0' + mp.get(r.group(1), r.group(1)), q.group(1))
            .replace('\0', '') + ']', k['freq_image'])
        omega, z = SR.var('omega'), SR.var('_conv_z')
        try:
            E = parse_expr(img, what=f'kernel {kname}').subs({omega: -SR.I() * z})
            E = SR(E).rational_simplify()
            N, D = SR(E.numerator()).expand(), SR(E.denominator()).expand()
            ok = (N.is_polynomial(z) and D.is_polynomial(z) and not N.has(omega)
                  and not D.has(omega) and D.degree(z) >= 1 and N.degree(z) <= D.degree(z))
        except Exception:
            ok = False
        if not ok:
            raise SDEError(f'{where}: the kernel {kname!r} (image {k["freq_image"]}) is not a '
                           f'proper rational function of omega; a kernel of a nonlinear '
                           f'function of the fields is embedded as auxiliary linear fields only '
                           f'then.')
        base = f'{kname}_emb{len(emb["records"]) + 1}'
        names = []

        def new_field(lhs, rhs):
            nm = f'{base}_{len(names) + 1}'
            if nm in self._allowed_names():
                raise SDEError(f'{where}: the auxiliary field name {nm!r} is taken.')
            self._fields[nm] = {'population': pop if len(self._populations()) > 1 or
                                self._pops else None,
                                'description': f'Markov embedding of Conv({kname}, ...)',
                                'domain': None, 'latex': '', 'exposed': None}
            lhs = lhs.replace('@', nm)
            self._eqs[nm] = {'lhs': self._translate(lhs, 'v1', where=f'{where} (embedding)'),
                             'rhs': self._translate(rhs, 'v1', where=f'{where} (embedding)'),
                             'lhs_raw': lhs, 'rhs_raw': rhs}
            names.append(nm)
            emb['fields'].append(nm)
            return nm

        def tx(e):
            return expr_to_text(SR(e))
        # a chain of stages, one per factor of D (with multiplicity): a linear
        # factor a + b z is (b*Dt + a)*u = prev; an irreducible factor of degree
        # d >= 2 a companion block whose states carry a restoring term on the lhs
        # ((Dt + 1)*p = p' + p) so each mean-field equation defines its field.
        # ``deriv`` holds Dt of every state as a linear form in the states and X,
        # from which the numerator N(Dt) is read off the last output.
        Xs = SR.var('_conv_X')
        deriv, cur, gain = {}, Xs, SR(1)
        for fac, mult in SR(D).factor_list():
            fac = SR(fac)
            deg = int(fac.degree(z)) if fac.has(z) else 0
            if deg == 0:
                gain = gain / fac ** mult
                continue
            cs = [SR(fac.coefficient(z, q)) for q in range(deg + 1)]
            for _ in range(int(mult)):
                prev = tx(cur).replace('_conv_X', f'({X})')
                if deg == 1:
                    nm = new_field(f'(({tx(cs[1])})*Dt + ({tx(cs[0])}))*@', prev)
                    u = SR.var(nm)
                    deriv[u] = (cur - cs[0] * u) / cs[1]
                    cur = u
                    continue
                ps = [SR.var(f'{base}_{len(names) + q + 1}') for q in range(deg)]
                for q in range(deg - 1):
                    new_field('(Dt + 1)*@', f'{ps[q + 1]} + {ps[q]}')
                    deriv[ps[q]] = ps[q + 1]
                last = cur - sum(cs[q] * ps[q] for q in range(deg))
                new_field(f'(({tx(cs[deg])})*Dt + 1)*@',
                          f'{tx(last).replace("_conv_X", f"({X})")} + {ps[deg - 1]}')
                deriv[ps[deg - 1]] = last / cs[deg]
                cur = ps[0]
        nd = int(N.degree(z))
        out_e, term = SR(0), cur
        for q in range(nd + 1):
            out_e += SR(N.coefficient(z, q)) * term
            if q < nd:
                term = SR(term).expand()
                if term.has(Xs):
                    raise SDEError(f'{where}: the kernel {kname!r} is not proper.')
                term = sum(term.coefficient(v_, 1) * d_ for v_, d_ in deriv.items())
        out = tx((gain * out_e).expand()).replace('_conv_X', f'({X})')
        # the output in the indexed spelling of the use site
        for nm in names:
            out = re.sub(rf'\b{re.escape(nm)}\b(?!\s*\[)', f'{nm}[i]', out)
        emb['records'].append({'kernel': kname, 'arg': X, 'fields': names,
                               'image': k['freq_image']})
        emb['cache'][key] = out
        return out

    def _strip_conv(self, text):
        out, pos = '', 0
        for a, _, c in self._conv_spans(text):
            out += text[pos:a] + ' 0 '
            pos = c
        return out + text[pos:]

    def _compose_kernels(self, text: str, where: str) -> str:
        """``Conv(h[i]*g[i,j], x)`` (a product of declared kernels) ->
        ``Conv(h_g[i,j], x)`` with the convolution ``h_g`` (product of the
        frequency images, sum of the delays) declared on the fly."""
        out, pos = '', 0
        for a, b, c in self._conv_spans(text):
            args = _split_top_level(text[b:c - 1])
            if len(args) != 2:
                raise SDEError(f'{where}: cannot read {text[a:c]!r} (Conv(kernel, field)).')
            facs = [f.strip() for f in _split_top_level(args[0], '*')]
            if len(facs) == 1:
                out += text[pos:c]
                pos = c
                continue
            refs = []
            for f in facs:
                m = re.fullmatch(r'([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?', f)
                if not m or m.group(1) not in self._kernels:
                    raise SDEError(f'{where}: {args[0].strip()!r} in Conv(...) must be a '
                                   f'declared kernel or a product of declared kernels '
                                   f'({f!r} is not one).')
                idx = [q.strip() for q in m.group(2).split(',')] if m.group(2) else []
                refs.append((m.group(1), idx))
            sites = []
            for _, idx in refs:
                for q in idx:
                    if q not in sites:
                        sites.append(q)
            # the composite's own index order: the population index i first
            sites.sort(key=lambda q: {'i': 0, 'j': 1}.get(q, 2))
            if len(sites) > 2:
                raise SDEError(f'{where}: a kernel product carries at most two indices '
                               f'(got {sites}).')
            local = dict(zip(sites, ('i', 'j')))
            images, delays, pops = [], [], [None] * len(sites)
            for kname, idx in refs:
                k = self._kernels[kname]
                if k['freq_image'] is None:
                    raise SDEError(f'{where}: kernel {kname!r} has only a time_expr; a '
                                   f'kernel product needs frequency images (give '
                                   f'freq_image= or kernel=).')
                kidx = list(k.get('indexed_by') or [])
                if len(idx) != len(kidx):
                    raise SDEError(f'{where}: kernel {kname!r} takes {len(kidx)} '
                                   f'index(es), got {idx}.')
                mp = {old: local[new] for old, new in zip(('i', 'j'), idx)}
                for new, popn in zip(idx, kidx):
                    pops[sites.index(new)] = popn

                def remap(t, _mp=mp):
                    def sub(mm):
                        inner = re.sub(r'\b([ij])\b',
                                       lambda q: '\0' + _mp.get(q.group(1), q.group(1)),
                                       mm.group(1))
                        return '[' + inner + ']'
                    return re.sub(r'\[([^\]]*)\]', sub, t).replace('\0', '')
                if k['freq_image'] != '1':
                    images.append(f'({remap(k["freq_image"])})')
                if k['delay']:
                    delays.append(f'({remap(k["delay"])})')
            cname = '_'.join(r[0] for r in refs)
            jit = [r[0] for r in refs if r[0] in self._jitter]
            if len(jit) > 1:
                raise SDEError(f'{where}: {args[0].strip()!r} holds two Gaussian jitter '
                               f'kernels {jit}; declare one Kernel.gaussian with sigma = '
                               f'sqrt(sigma1^2 + sigma2^2).')
            spec = dict(time_expr=None, freq_image='*'.join(images) or '1', latex_name='',
                        indexed=False, indexed_by=pops if sites else None,
                        delay=' + '.join(delays) if delays else None)
            have = self._kernels.get(cname)
            if have is None:
                if cname in self._allowed_names():
                    raise SDEError(f'{where}: the composite kernel name {cname!r} is taken.')
                self._kernels[cname] = spec
            elif any(have.get(q) != spec[q] for q in ('freq_image', 'delay', 'indexed_by')):
                raise SDEError(f'{where}: kernel {cname!r} exists and is not the convolution '
                               f'{args[0].strip()!r} (index the factors the same way at '
                               f'every use, or declare the product with define_kernel).')
            if jit:
                self._jitter[cname] = dict(self._jitter[jit[0]], factors=[r[0] for r in refs])
            ref = cname + (f'[{",".join(sites)}]' if sites else '')
            out += text[pos:a] + f'Conv({ref}, {args[1].strip()})'
            pos = c
        return out + text[pos:]

    # ── derivation ────────────────────────────────────────────────────
    def _derive(self) -> dict:
        """Derive the action terms and the mean-field equations."""
        self.notes = [f'equation({f!r}): order-{e["order"]} lhs {e["lhs_raw"]!r} embedded as a '
                      f'first-order system with {", ".join(e["fields"])}; noise into {f!r} forces '
                      f'{e["top"]!r}.' for f, e in self._embedded.items()]
        self._action_log, self._drift_log, self._mf_only_log = [], [], []
        self._embed_nonlinear_convs()
        for f, info in self._fields.items():
            if info['exposed'] is None and f not in self._eqs:
                raise SDEError(f'field {f!r} has no equation; call .equation() or .drift().')
        # declared operators -> the engine's spelling; a Laplacian (or space=d)
        # makes the model an SPDE, a Laplacian on an expression (lap(u^2)) puts
        # its action in the spatial operator IR
        info = {}
        for f, eq in self._eqs.items():
            eq['lhs'] = self._translate(eq['lhs_raw'], 'v1', info, f'equation({f!r}) lhs')
            eq['rhs'] = self._translate(eq['rhs_raw'], 'v1', info, f'equation({f!r}) rhs')
        extras = [self._translate(e, 'v1', info, 'extra_action') for e in self._extra]
        # operators act in the equations (and extra_action) only: one in a
        # function body would reach the engine as an undeclared symbol
        for (name, _a, expr), _kw in self._functions:
            self._check_no_operator(expr, f'define_function({name!r}): its expression')
        for name, s in self._sources.items():
            if s.get('source') is not None:
                for k, v in s['source'].params.items():
                    self._check_no_operator(str(v), f'noise({name!r}) {k}=')
            for f, c in s['couples'].items():
                self._check_no_operator(str(c), f'noise({name!r}) coefficient for {f!r}')
        used_space = any(op.spatial for op in info.get('used', ()))
        spatial = self._space is not None or used_space
        ir = spatial and bool(info.get('nonlinear'))
        if spatial:
            self._check_spatial_scope()
        else:
            self._check_temporal_scope(info)
        if self._marcus_sources():
            self._check_marcus_joint()
        allowed_src = (set(self._fields) | set(self._param_names)
                       | set(self._function_names) | {'i', 'j'})
        for name, s in self._sources.items():
            if s.get('pp'):
                self._check_pp(name, s['pp'])
                continue
            try:
                s['source'].validate(allowed_src)
            except NoiseError as exc:
                if s['expose'] and 'Conv' in str(exc):
                    raise SDEError(f'noise({name!r}): {exc} A spike train whose rate filters '
                                   f'fields (Conv) is declared with point_process({s["expose"]!r}, '
                                   f'rate=...).') from exc
                raise self._source_error(name, s['source'], allowed_src, exc) from exc
            colored = s.get('colored', {}).get('into', {})
            for f, c in list(s['couples'].items()) + list(colored.items()):
                if f not in self._fields:
                    raise SDEError(f'noise({name!r}): couples to undeclared field {f!r}.')
                for k, e in enumerate(c if isinstance(c, (list, tuple)) else [c]):
                    what = (f'noise({name!r}) coefficient {k + 1} for {f!r}'
                            if isinstance(c, (list, tuple)) else f'noise({name!r}) coefficient for {f!r}')
                    self._check_text(str(e), what)
                    if 't' in expr_names(parse_expr(self._index(str(e)))):
                        raise SDEError(f'noise({name!r}): coefficient depends on t.')
            if isinstance(s['source'], GaussianVector) and any(
                    self._field_dependent_text(expr_to_text(c)) for r in s['source'].cov for c in r):
                raise SDEError(f'noise({name!r}): the GaussianVector cov must be free of the fields '
                               f'(put a state-dependent coupling in into=).')
        for f, eq in self._eqs.items():
            self._check_text(eq['lhs'], f'equation({f!r}) lhs', eq['lhs_raw'])
            self._check_text(eq['rhs'], f'equation({f!r}) rhs', eq['rhs_raw'])
        for name, s in self._sources.items():
            if self._src_interp(name) != 'stratonovich':
                continue
            own = 'interp' in s
            where = (f"noise({name!r}, interpretation='stratonovich')" if own
                     else "interpretation('stratonovich')")
            if s.get('pp') or not s['source'].is_gaussian:
                raise SDEError(
                    f"{where}: source {name!r} is a jump/"
                    f"non-Gaussian source; the Stratonovich rule is defined here "
                    f"for Gaussian noise only; jump SDEs take the Marcus rule, "
                    + (f"noise({name!r}, ..., interpretation='marcus')." if own
                       else "interpretation('marcus')."))
            if s['expose']:
                raise SDEError(
                    f"{where}: exposed source {name!r} is "
                    f"not supported; couple it with into= instead.")
        if self._marcus_sources():
            self._check_marcus()
        self._jitter_uses = {} if spatial else self._check_jitter(extras)

        extra_rhs = {f: [] for f in self._eqs}       # deterministic additions (action + MF)
        mf_rhs = {f: [] for f in self._eqs}          # MF-only additions (non-Gaussian means)
        source_terms = {}                            # per population: action text terms
        exposed_eqs = []                             # (field, rhs text)
        gauss_units = []                             # (name, Gaussian source, couples): Stratonovich
        self._marcus_terms = OrderedDict()
        self._ito_drift = {}                         # field -> Stratonovich corrections (simulator)
        self._colored_drift = {}                     # field -> coloured-noise terms (simulator)
        vector_units = []                            # (name, GaussianVector, couplings): Stratonovich

        for name, s in self._sources.items():
            src = s['source']
            pop = self._pop_of_source(name)
            if s.get('pp'):
                n, size = s['expose'], s['pp']['size']
                rate = self._index(self._compose_kernels(s['pp']['rate'], f'point_process({n!r})'))
                jump = f'{n}t[i]' if size == '1' else f'{_wrap(size)}*{n}t[i]'
                self._emit(source_terms, pop, f'{n}t[i]*{n}[i] - (exp({jump}) - 1)*{_wrap(rate)}', name)
                exposed_eqs.append((n, rate if size == '1' else f'{_wrap(size)}*{_wrap(rate)}'))
                continue
            if isinstance(src, GaussianVector):
                coupv = {f: [self._parse(str(e), f'coefficient {k + 1} of {name} in {f}')
                             for k, e in enumerate(c)] for f, c in s['couples'].items()}
                cov = [[self._parse(expr_to_text(e), f'{name}.cov') for e in r] for r in src.cov]
                th = [sum((coupv[f][k] * SR.var(f'{f}t__i') for f in coupv), SR(0))
                      for k in range(src.n)]
                K = sum(cov[k][l] * th[k] * th[l] for k in range(src.n) for l in range(src.n)) / 2
                for f, cs in coupv.items():
                    mean = sum((cs[k] * self._parse(expr_to_text(src.mean_vector[k]), f'{name}.mean')
                                for k in range(src.n)), SR(0))
                    if not mean.is_zero():
                        extra_rhs[f].append(mean)
                self._emit(source_terms, pop, f'-{_wrap(expr_to_text(K.expand()))}', name)
                vector_units.append((name, cov, coupv, self._src_interp(name)))
                continue
            coup = {f: self._parse(str(c), f'coefficient of {name} in {f}')
                    for f, c in s['couples'].items()}
            for f in coup:
                if self._fields[f]['exposed'] is not None:
                    raise SDEError(f'noise({name!r}): cannot couple to the exposed '
                                   f'source field {f!r}.')
            if self._src_interp(name) == 'marcus' and not s['expose']:
                units = self._marcus_units(name, s, coup)
            else:
                units = [(name, src, False)]
            for uname, src, series in units:
                if series:
                    self._marcus_series(uname, src, coup, pop, source_terms, mf_rhs)
                    continue
                K_full = parse_expr(self._index(expr_to_text(
                    src.action_cgf(self._series_order))), what=f'K of {uname}')
                mean = parse_expr(self._index(expr_to_text(src.mean())), what=f'mean of {uname}')
                if s['expose']:
                    n = s['expose']
                    nt = SR.var(f'{n}t__i')
                    term = f'{n}t[i]*{n}[i] - {_wrap(expr_to_text(K_full.subs({THETA: nt})))}'
                    self._emit(source_terms, pop, term, uname)
                    exposed_eqs.append((n, expr_to_text(mean)))
                    for f, c in coup.items():
                        extra_rhs[f].append(c * SR.var(f'{n}__i'))
                    continue
                if src.is_gaussian:
                    gm = src.params['mean']
                    gauss_units.append((uname, src, s['couples'], self._src_interp(name)))
                    if not SR(gm).is_zero():
                        gm_i = parse_expr(self._index(expr_to_text(gm)))
                        for f, c in coup.items():
                            extra_rhs[f].append(c * gm_i)
                        self.notes.append(f'Gaussian source {uname!r}: mean '
                                          f'{expr_to_text(gm)} moved into the drift of '
                                          f'{", ".join(coup)}.')
                        # The action keeps only the variance part of K.
                        K_full = parse_expr(self._index(expr_to_text(src.params['var'])),
                                            what=f'{uname}.var') * THETA ** 2 / 2
                else:
                    for f, c in coup.items():
                        if not SR(mean).is_zero():
                            mf_rhs[f].append(c * mean)
                            self._mf_only_log.append({'field': f, 'source': uname,
                                                      'text': expr_to_text(c * mean)})
                theta_val = sum(c * SR.var(f'{f}t__i') for f, c in coup.items())
                K_term = K_full.subs({THETA: theta_val})
                self._emit(source_terms, pop, f'-{_wrap(expr_to_text(K_term))}', uname)

        # coloured Gaussian noise: c*eta into each field (eta a combination of the
        # filter's states, see _colored_noise)
        for name, s in self._sources.items():
            col = s.get('colored')
            if not col:
                continue
            out = self._parse(col['output'], f'coloured noise of {name}')
            for f, c in col['into'].items():
                term = self._parse(str(c), f'coefficient of {name} in {f}') * out
                extra_rhs[f].append(term)
                self._colored_drift.setdefault(f, []).append(term)

        # Stratonovich -> Ito drift correction (Gaussian sources; under Marcus
        # the Gaussian sources and Gaussian parts, whose Marcus rule it is).
        vector_units = [u for u in vector_units if u[3] in ('stratonovich', 'marcus')]
        if vector_units:
            mode = self._interp_mode()
            dtcoef = self._dt_coefficients()
            for name, cov, coupv, rule in vector_units:
                rule = mode if mode != 'mixed' else rule
                n = len(cov)
                b = {f: [c / dtcoef[f] for c in cs] for f, cs in coupv.items()}
                for fi in b:
                    corr = sum((cov[k][l] * b[fj][l] * b[fi][k].diff(self._sym(fj))
                                for k in range(n) for l in range(n) for fj in b), SR(0)) / 2
                    corr = corr.simplify_full()
                    if not corr.is_zero():
                        extra_rhs[fi].append(dtcoef[fi] * corr)
                        self._ito_drift.setdefault(fi, []).append(dtcoef[fi] * corr)
                        self._drift_log.append({'field': fi, 'source': name, 'rule': rule,
                                                'text': expr_to_text(dtcoef[fi] * corr)})
                        self.notes.append(
                            f'Stratonovich -> Ito: drift of {fi!r} += '
                            f'{expr_to_text(dtcoef[fi] * corr)} (source {name!r}).')
        gauss_units = [u for u in gauss_units if u[3] in ('stratonovich', 'marcus')]
        if gauss_units:
            mode = self._interp_mode()
            dtcoef = self._dt_coefficients()
            for name, src, couples, rule in gauss_units:
                rule = mode if mode != 'mixed' else rule
                V = self._parse(expr_to_text(src.params['var']), f'{name}.var')
                b = {f: self._parse(str(c), 'coefficient') / dtcoef[f]
                     for f, c in couples.items()}
                for fi in b:
                    corr = SR(0)
                    for fj in b:
                        xj = self._sym(fj)
                        corr += (V * b[fj] * b[fi].diff(xj)
                                 + b[fj] * b[fi] * V.diff(xj) / 2) / 2
                    corr = corr.simplify_full()
                    if not corr.is_zero():
                        extra_rhs[fi].append(dtcoef[fi] * corr)
                        self._ito_drift.setdefault(fi, []).append(dtcoef[fi] * corr)
                        self._drift_log.append({'field': fi, 'source': name,
                                                'rule': rule,
                                                'text': expr_to_text(dtcoef[fi] * corr)})
                        self.notes.append(
                            f'Stratonovich -> Ito: drift of {fi!r} += '
                            f'{expr_to_text(dtcoef[fi] * corr)} (source {name!r}).')

        # Per-field action terms and mean-field equations.
        field_terms, mf_eqs = {}, []
        for f, eq in self._eqs.items():
            pop = self._pop_of_field(f)
            lhs_txt, rhs_txt = self._index(eq['lhs']), self._index(eq['rhs'])
            add = sum(extra_rhs[f], SR(0))
            rhs_act = rhs_txt if add.is_zero() else f'{rhs_txt} + {_wrap(expr_to_text(add))}'
            if ir:
                # the action in the operator IR (Dt(u), Lap(u^2)); the mean-field
                # equations keep the plain form (both operators vanish there)
                where = f'equation({f!r})'
                lhs_ir = self._index(self._ir_text(eq['lhs_raw'], f'{where} lhs'))
                rhs_ir = self._index(self._ir_text(eq['rhs_raw'], f'{where} rhs'))
                rhs_ir = rhs_ir if add.is_zero() else f'{rhs_ir} + {_wrap(expr_to_text(add))}'
                field_terms.setdefault(pop, []).append(f'{f}t[i]*({lhs_ir} - {_wrap(rhs_ir)})')
                self._action_log.append({'kind': 'dynamics', 'field': f,
                                         'text': field_terms[pop][-1],
                                         'lhs': lhs_ir, 'rhs': rhs_ir})
            else:
                if spatial:
                    self._check_v1_operator_terms(f'({lhs_txt}) - ({rhs_act})',
                                                  f'equation({f!r})')
                field_terms.setdefault(pop, []).append(
                    f'{f}t[i]*({lhs_txt} - {_wrap(rhs_act)})')
                self._action_log.append({'kind': 'dynamics', 'field': f,
                                         'text': field_terms[pop][-1],
                                         'lhs': lhs_txt, 'rhs': rhs_act})
            add_mf = sum(mf_rhs[f], SR(0))
            rhs_mf = rhs_act if add_mf.is_zero() else f'{rhs_act} + {_wrap(expr_to_text(add_mf))}'
            lhs_mf, rhs_mf = self._normalise_lhs(f, lhs_txt, rhs_mf)
            mf_eqs.append({'field': f, 'lhs': lhs_mf,
                           'rhs': self._collapse_kernels(rhs_mf), 'population': pop})
        for n, rhs in exposed_eqs:
            mf_eqs.append({'field': n, 'lhs': f'{n}[i]',
                           'rhs': self._collapse_kernels(rhs),
                           'population': self._pop_of_field(n)})
        # Order the MF equations by field declaration order.
        order = list(self._fields)
        mf_eqs.sort(key=lambda e: order.index(e['field']))

        if spatial:
            if not ir:
                for e_v1 in extras:
                    self._check_v1_operator_terms(self._index(e_v1), 'extra_action')
            # one scalar action and scalar equations (SpatialModelBuilder form)
            p = self._populations()[0]['name']
            body = ' + '.join(field_terms.get(p, []) + source_terms.get(p, [])).replace('+ -', '- ')
            action = _strip_index(body)
            for e, e_v1 in zip(self._extra, extras):
                e_txt = self._ir_text(e, 'extra_action') if ir else e_v1
                action += f' + ({_strip_index(" ".join(e_txt.split()))})'
                self._action_log.append({'kind': 'extra', 'text': _strip_index(" ".join(e_txt.split()))})
            for e in mf_eqs:
                e.update(lhs=_strip_index(e['lhs']), rhs=_strip_index(e['rhs']), population=None)
            self.notes = [_strip_index(n) for n in self.notes]
            return {'action': action, 'equations': mf_eqs, 'spatial': True, 'operator_ir': ir}

        parts = []
        for p in self._populations():
            terms = field_terms.get(p['name'], []) + source_terms.get(p['name'], [])
            if not terms:
                continue
            body = '\n    + '.join(terms).replace('+ -', '- ')
            parts.append(f"sum(\n    {body}\n    for i in {p['name']})")
        action = '\n+ '.join(parts)
        for e in extras:
            action += f'\n+ ({self._rename_pop(" ".join(e.split()))})'
            self._action_log.append({'kind': 'extra', 'text': self._rename_pop(" ".join(e.split()))})
        return {'action': action, 'equations': mf_eqs}

    def _check_jitter(self, extras) -> dict:
        """The uses of the Gaussian jitter kernels (``Kernel.gaussian``).  The
        Gaussian density is acausal; its fit is the density shifted by ``T``
        with the shift dropped, which changes no stationary statistic only when
        the WHOLE input is shifted: the kernel must read an exogenous source (a
        point process, or an exposed noise source, whose rate / parameters hold
        no field: nothing feeds back into it), and every use of that source
        must carry the same jitter (one common shift).  Returns ``{source
        field: {'sigma', 'tol', 'fit', 'kernels'}}``."""
        if not self._jitter:
            return {}
        texts = []
        for f, eq in self._eqs.items():
            texts += [(f'equation({f!r})', eq['lhs']), (f'equation({f!r})', eq['rhs'])]
        for name, s in self._sources.items():
            if s.get('pp'):
                texts.append((f'point_process({name!r}) rate',
                              self._compose_kernels(s['pp']['rate'], f'point_process({name!r})')))
                texts.append((f'point_process({name!r}) size', s['pp']['size']))
                continue
            texts += [(f'noise({name!r}) coefficient for {f!r}', str(c))
                      for f, c in s['couples'].items()]
            texts += [(f'noise({name!r}) {k}=', str(v)) for k, v in s['source'].params.items()]
        texts += [(f'define_function({fn!r})', expr) for (fn, _a, expr), _kw in self._functions]
        texts += [('extra_action', e) for e in extras]
        fields = set(self._fields)

        def depends(text):
            return sorted(set(re.findall(r'\b([A-Za-z_]\w*)\b', str(text))) & fields)

        def jittered(kern):
            # every jitter kernel named in a Conv's kernel argument, however
            # written: a composite (kX_g), a product not composed (kX*g, in a
            # noise coefficient), parentheses ((g))
            return [n for n in re.findall(r'\b([A-Za-z_]\w*)\b', kern) if n in self._jitter]

        uses = {}
        for where, text in texts:
            for a, b, c in self._conv_spans(text):
                args = _split_top_level(text[b:c - 1])
                kern = args[0].strip() if args else ''
                jk = jittered(kern)
                if len(args) != 2 or not jk:
                    continue
                if len(set(jk)) > 1:
                    raise SDEError(f'{where}: Conv({kern}, ...) holds two Gaussian jitter '
                                   f'kernels {sorted(set(jk))}.')
                kname, arg = jk[0], args[1].strip()
                dep_sigma = depends(self._jitter[kname]['sigma'])
                if dep_sigma:
                    raise SDEError(f'{where}: the Gaussian jitter {kname!r} has sigma = '
                                   f'{self._jitter[kname]["sigma"]}, which holds the field(s) '
                                   f'{dep_sigma}; sigma is a parameter expression.')
                mm = re.fullmatch(r'([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?', arg)
                src = mm.group(1) if mm else None
                exposed = self._fields.get(src, {}).get('exposed') if src else None
                what = (f'{where}: Conv({kern}, {arg}): the Gaussian jitter kernel '
                        f'{kname!r} is acausal; its time shift is free only on an '
                        f'exogenous input (a point_process, or a noise source with as_field=, '
                        f'whose rate holds no field)')
                if exposed is None:
                    raise SDEError(f'{what}, and {arg!r} is not such a source.')
                s = self._sources[exposed]
                if s['couples']:
                    raise SDEError(
                        f'{where}: the jittered input {src!r} also drives '
                        f'{sorted(s["couples"])} directly (into=); the jitter is a common '
                        f'time shift of the input, so every use of {src!r} must be '
                        f'Conv(K*g, {src}) with the same Kernel.gaussian g.')
                if s.get('pp'):
                    dep = depends(self._compose_kernels(s['pp']['rate'], where)) + depends(
                        s['pp']['size'])
                else:
                    dep = [d for v in s['source'].params.values() for d in depends(v)]
                if dep:
                    raise SDEError(f'{what}; the rate of {src!r} depends on the field(s) '
                                   f'{sorted(set(dep))} (feedback into the source), so '
                                   f'shifting it changes the statistics.')
                key = (self._jitter[kname]['sigma'], self._jitter[kname]['tol'])
                u = uses.setdefault(src, {'key': key, 'kernels': set()})
                if u['key'] != key:
                    raise SDEError(f'{where}: the input {src!r} is jittered with sigma, tol = '
                                   f'{u["key"]} and {key}; its uses must share one Gaussian '
                                   f'jitter (one common time shift of the input).')
                u['kernels'].add(kname)
        for src, u in uses.items():
            for where, text in texts:
                out, pos = '', 0
                for a, b, c in self._conv_spans(text):
                    args = _split_top_level(text[b:c - 1])
                    rd = re.match(r'([A-Za-z_]\w*)', args[1].strip()) if len(args) == 2 else None
                    if rd and rd.group(1) == src and jittered(args[0]):
                        out += text[pos:a] + ' 0 '
                        pos = c
                out += text[pos:]
                if re.search(rf'\b{re.escape(src)}\b', out):
                    raise SDEError(
                        f'{where}: the jittered input {src!r} is used without its Gaussian '
                        f'jitter; the jitter is a common time shift of the input, so every '
                        f'use of {src!r} must be Conv(K*g, {src}) with the same '
                        f'Kernel.gaussian g.')
        out = {}
        for src, u in uses.items():
            j = self._jitter[sorted(u['kernels'])[0]]
            fit = j['fit']
            out[src] = {'sigma': j['sigma'], 'tol': j['tol'], 'fit': fit,
                        'kernels': sorted(u['kernels'])}
            self.notes.append(
                f'input {src!r}: Gaussian jitter (sigma = {j["sigma"]}) as a rational fit of '
                f'{fit.order} states of the density shifted by {fit.shift:g}*sigma (shift '
                f'dropped: exogenous stationary input); L1 error {fit.l1:.2g}.')
        return out

    def _check_spatial_scope(self):
        """An SPDE (``space=d`` or a Laplacian) builds the spatial pipeline's
        model: refuse what that pipeline does not take, never drop it."""
        who = f'SDE({self.name!r})'
        if self._space is None:
            raise SDEError(f'{who}: a Laplacian makes the model an SPDE; give its spatial '
                           f'dimension, SDE({self.name!r}, space=d).')
        temporal_only = ('temporal-only feature, not supported by the spatial pipeline '
                         '(an SPDE built from SDE(..., space=d) or a Laplacian)')
        if self._pops or self._n_pop_default > 1:
            raise SDEError(f'{who}: populations are a {temporal_only}; declare one field per '
                           f'species.')
        if self._kernels:
            raise SDEError(f'{who}: convolution kernels and delays (Conv(K, x), x(t - d), '
                           f'define_kernel) are a {temporal_only}: '
                           f'{", ".join(self._kernels)}.')
        for f, info in self._fields.items():
            if info['population'] is not None:
                raise SDEError(f'{who}: field {f!r} has a population; populations are a '
                               f'{temporal_only}.')
        for (name,), kw in self._params:
            if kw.get('indexed') or kw.get('indexed_by'):
                raise SDEError(f'{who}: parameter {name!r} is indexed by a population; '
                               f'populations are a {temporal_only}.')
        for (name, _a, _e), kw in self._functions:
            if kw.get('population'):
                raise SDEError(f'{who}: function {name!r} has a population; populations are '
                               f'a {temporal_only}.')
        for n in ('Laplacian', 'Lap'):
            owner = self._name_owner(n)
            if owner:
                raise SDEError(f'{who}: the {owner} is reserved in a spatial model (the '
                               f'Laplacian); rename it.')
        if self._interp != 'ito':
            raise SDEError(f"{who}: interpretation({self._interp!r}) is a {temporal_only}; "
                           f"spatial noise is Ito.")
        for name in self._sources:
            if self._src_interp(name) != 'ito':
                raise SDEError(f"{who}: noise({name!r}) is {self._src_interp(name)}; the "
                               f"Stratonovich and Marcus rules are a {temporal_only}; spatial "
                               f"noise is Ito.")
        for name, s in self._sources.items():
            if s.get('pp'):
                raise SDEError(f'{who}: point_process({name!r}) is a {temporal_only}.')
            if s['expose']:
                raise SDEError(f'{who}: an exposed noise source ({name!r}, as_field=) is a '
                               f'{temporal_only}.')
            src = s['source']
            if not src.is_gaussian:
                raise SDEError(f'{who}: noise({name!r}) is a {type(src).__name__} source; an '
                               f'SPDE takes space-time white Gaussian noise, '
                               f"Gaussian(var=..., space='white') (non-Gaussian spatial noise "
                               f'is not supported yet).')
            if getattr(src, 'space', None) != 'white':
                raise SDEError(f"{who}: noise({name!r}) needs its spatial structure: an SPDE "
                               f"takes space-time white noise, Gaussian(var=..., "
                               f"space='white').")

    def _check_temporal_scope(self, info):
        """A time-only SDE: refuse the spatial-only declarations."""
        who = f'SDE({self.name!r})'
        for name, s in self._sources.items():
            if getattr(s.get('source'), 'space', None) is not None:
                raise SDEError(f"{who}: noise({name!r}) is spatial (space="
                               f"{s['source'].space!r}); give SDE(..., space=d) and a "
                               f"Laplacian for an SPDE.")
        if self._boundary is not None or self._dyson is not None:
            raise SDEError(f'{who}: boundary() / dyson() describe an SPDE; give '
                           f'SDE(..., space=d).')
        for op in info.get('used', ()):
            if not op().supports('temporal'):
                raise SDEError(f'{who}: the {op.__name__} is not supported by the temporal '
                               f'pipeline.')

    def _normalise_lhs(self, f, lhs, rhs):
        """If ``lhs`` vanishes at Dt = 0 (``Dt*x``), make the mean-field
        equation define ``x*``; the residual ``lhs - rhs`` is unchanged.

        Preferred: move the linear restoring term of ``rhs`` (field-independent
        coefficient) onto the lhs.  Fallback, where ``rhs`` has no such term
        (a rational or exponential drift, or a linear coefficient that contains
        a field): add ``c*x`` to both sides, ``lhs -> lhs + c*x`` and
        ``rhs -> rhs + c*x``, with ``c = 1``.  The mean-field solve is a root
        solve on ``lhs|_{Dt=0} - rhs`` in which ``c*x`` cancels, and linear
        stability reads ``d(lhs - rhs)/dx`` (also ``c``-free), so ``c`` only
        names the saddle text ``x* = (rhs + c*x*)/c``; 1 keeps it unscaled."""
        workaround = (f"write the restoring term on the lhs, e.g. lhs='(Dt + 1)*{f}' "
                      f"with rhs = <drift> + {f}")
        try:
            L = parse_expr(lhs, what=f'lhs of {f}')
        except NoiseError as exc:
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} cannot be read symbolically ({exc}); '
                f'if it vanishes at Dt=0, {workaround}.') from exc
        zero = {_DT: 0}
        if any(isinstance(o, Laplacian) for o in self._operators.values()):
            zero[_LAP] = 0          # a homogeneous saddle: the Laplacian vanishes like Dt
        if not L.subs(zero).simplify_full().is_zero():
            return lhs, rhs
        try:
            R = parse_expr(rhs, what=f'rhs of {f}')
        except NoiseError as exc:
            raise SDEError(
                f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs cannot be '
                f'split symbolically ({exc}); {workaround}.') from exc
        x = self._sym(f)
        Re = R.expand()
        # (a diffusion term D*Laplacian*x is no restoring term: it vanishes at the saddle)
        c1 = Re.subs({_LAP: 0}).coefficient(x, 1) if _LAP in zero else Re.coefficient(x, 1)
        fieldsyms = {g + '__i' for g in self._fields}
        if not (c1.is_zero() or set(map(str, c1.variables())) & fieldsyms):
            new_lhs = expr_to_text((L - c1 * x).expand())
            new_rhs = expr_to_text((Re - c1 * x).expand())
            self.notes.append(f'mean-field equation of {f!r}: linear term '
                              f'{expr_to_text(c1 * x)} moved to the lhs.')
            return new_lhs, (new_rhs if new_rhs.strip() else '0')
        if str(x) not in set(map(str, R.variables())):
            if not self._field_dependent(R):
                raise SDEError(
                    f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs does not '
                    f'depend on {f} or any other field, so the mean-field equation cannot '
                    f'define {f}*; {workaround}.')
            if not any(self._has_token(e['rhs'], f) or self._has_token(e['lhs'], f)
                       for g, e in self._eqs.items() if g != f):
                raise SDEError(
                    f'equation({f!r}): lhs={lhs!r} vanishes at Dt=0 and rhs does not '
                    f'depend on {f}, and no other equation holds {f}, so no mean-field '
                    f'equation defines {f}* (a pure integrator needs another equation that '
                    f'feeds {f} back); {workaround}.')
            # a pure integrator (Dt*x = v): its mean-field equation 0 = rhs constrains the
            # other fields, and the saddle of x comes from the system solved together
            self.notes.append(f'mean-field equation of {f!r}: a pure integrator (its rhs holds no '
                              f'{f}); 0 = rhs constrains the other fields and {f}* is fixed by '
                              f'the system together.')
        # No syntactic field-independent linear term: add and subtract c*x.
        cx = expr_to_text(x)
        new_lhs = expr_to_text((L + x).expand())
        new_rhs = f'{_wrap(rhs)} + {cx}'
        self.notes.append(
            f'mean-field equation of {f!r}: rhs has no field-independent linear '
            f'term in {f}; added {cx} to both sides (lhs {new_lhs!r}, c = 1; the '
            f'residual and the linear stability are unchanged).')
        return new_lhs, new_rhs

    # ── Marcus rule ───────────────────────────────────────────────────
    def _dt_coefficients(self) -> dict:
        """The constant ``Dt`` coefficient ``a`` of each equation's lhs
        (``a*Dt*x + ...``): a coupling ``c`` in lhs units is the vector field
        ``c/a`` of the field."""
        dtcoef = {}
        for f, eq in self._eqs.items():
            L = self._parse(eq['lhs'], f'lhs of {f}').expand()
            dt_terms = L.coefficient(_DT, 1)
            a = dt_terms.coefficient(self._sym(f), 1)
            # the x part of the Dt terms must be exactly a*x with a free of the fields: ``(1 + x^2)*Dt*x`` has
            # the Dt-coefficient x + x^3, whose x-part alone (1) would be taken for a
            if (a.is_zero() or set(map(str, a.variables())) & {g + '__i' for g in self._fields}
                    or (dt_terms - a * self._sym(f)).expand().has(self._sym(f))):
                raise SDEError(f"{self._interp_mode()}: cannot read a constant Dt "
                               f"coefficient off lhs={eq['lhs']!r}.")
            dtcoef[f] = a
        return dtcoef

    def _field_dependent(self, expr) -> bool:
        """True when ``expr`` (indexed form, ``x__i``) holds a field."""
        return any('__' in str(v) and str(v).split('__')[0] in self._fields
                   for v in SR(expr).variables())

    def _check_marcus(self):
        """Refuse what the Marcus rule is not defined for here: exposed
        sources, and point processes whose spikes enter an equation with a
        state-dependent coefficient (a constant one is the same jump in every
        interpretation)."""
        for name in self._marcus_sources():
            s = self._sources[name]
            if s.get('pp'):
                self._check_marcus_pp(name, s['pp'])
            elif s['expose']:
                raise SDEError(
                    f"interpretation('marcus'): exposed source {name!r} is not "
                    f"supported (it enters the equations as a field, so no jump "
                    f"flow is defined); couple it with into= instead.")

    def _check_marcus_joint(self):
        """A coupling that reads other units of the population (``x[j]``, a
        population sum) would move several units with one jump: a joint
        (multivariate) jump, which needs mixed moments.  Refused before the
        coupling is parsed (the parser refuses it under every interpretation,
        with a message about parsing)."""
        names = '|'.join(re.escape(f) for f in sorted(self._fields, key=len, reverse=True))
        for name in self._marcus_sources():
            s = self._sources[name]
            for f, c in s['couples'].items():
                c = str(c)
                if ((names and re.search(rf'\b({names})\s*\[\s*(?!i\s*\])', c))
                        or re.search(r'\bsum\s*\(', c)):
                    raise SDEError(
                        f"interpretation('marcus'): the coupling {c!r} of source "
                        f"{name!r} to {f!r} reads other units of the population, so "
                        f"one jump would move several units jointly; joint "
                        f"(multivariate) jumps need mixed moments and are not "
                        f"supported. One scalar source coupled to several fields of "
                        f"the same unit (into={{'x': ..., 'y': ...}}) is a jump "
                        f"along a vector field and is supported; independent sources "
                        f"per unit are too.")

    def _check_marcus_pp(self, n, pp):
        fix = (f"declare the jump as noise(..., Poisson(...) or CompoundPoisson(...), "
               f"into={{field: coefficient}}), whose state-dependent coupling "
               f"takes the Marcus flow, or use interpretation('ito')")
        bare_in = []
        for f, eq in self._eqs.items():
            rhs = self._strip_conv(eq['rhs'])     # a filtered train Conv(K, n) is no jump
            if not re.search(rf'\b{re.escape(n)}\b', rhs):
                continue
            bare_in.append(f)
            try:
                R = self._parse(rhs, f'rhs of {f}')
            except NoiseError:
                raise SDEError(
                    f"interpretation('marcus'): spike train {n!r} enters "
                    f"equation({f!r}) outside Conv(...) in a form whose coefficient "
                    f"cannot be read (a population sum?); a point process with a "
                    f"state-dependent coupling is not supported under Marcus: "
                    f"{fix}.") from None
            nsym = SR.var(f'{n}__i')
            coef = R.diff(nsym)
            if coef.has(nsym) or self._field_dependent(coef):
                raise SDEError(
                    f"interpretation('marcus'): spike train {n!r} enters "
                    f"equation({f!r}) with the state-dependent coefficient "
                    f"{expr_to_text(coef)!r}; a point process with a state-dependent "
                    f"coupling is not supported under Marcus (a constant coupling "
                    f"is the same jump in every interpretation): {fix}.")
        if bare_in and self._field_dependent(self._parse(pp['size'], f'size of {n}')):
            raise SDEError(
                f"interpretation('marcus'): spike train {n!r} has the state-dependent "
                f"size {pp['size']!r} and enters equation({bare_in[0]!r}) as a jump; a "
                f"point process with a state-dependent coupling is not supported "
                f"under Marcus: {fix}.")

    def _marcus_units(self, name, s, coup):
        """Split a coupled source into the pieces the Marcus rule treats
        apart: ``[(name, source, series)]``.  Gaussian sources and Gaussian
        parts of a sum take the Stratonovich route (``series`` False); a jump
        source (or the jump parts of a sum) with a state-dependent coupling
        takes the Marcus series (``series`` True); with a constant coupling
        and no state-dependent Gaussian part the source is unchanged (Marcus
        = Ito: the flow of a constant field is a translation)."""
        from api.noise import SumSource, Cumulants, CGF
        src = s['source']
        fields = list(self._fields)
        parts = list(src.parts) if isinstance(src, SumSource) else [src]
        for q in parts:
            if isinstance(q, (Cumulants, CGF)) and q.field_dependent(fields):
                raise SDEError(
                    f"interpretation('marcus'): source {name!r} is a state-dependent "
                    f"{type(q).__name__} source, whose Gaussian part cannot be told "
                    f"from its jumps (the two take different Marcus rules); declare "
                    f"it as Gaussian(...) + a jump source, or use interpretation('ito').")
        if len(parts) == 1 and src.is_gaussian:
            return [(name, src, False)]
        sd = any(self._field_dependent(c) for c in coup.values())
        gauss = [q for q in parts if q.is_gaussian]
        jumps = [q for q in parts if not q.is_gaussian]
        if not sd and not any(q.field_dependent(fields) for q in gauss):
            return [(name, src, False)]
        units = [(name if len(parts) == 1 else f'{name} (part {k + 1})', q, False)
                 for k, q in enumerate(parts) if q.is_gaussian]
        if jumps:
            j = jumps[0] if len(jumps) == 1 else SumSource(*jumps)
            units.append((name, j, sd))
        return units

    def _marcus_series(self, name, src, coup, pop, source_terms, mf_rhs):
        """The Marcus noise term of the jump source ``src`` coupled through the
        vector field ``G = sum_f b_f d/dx_f`` (``b_f = c_f/a_f``):
        ``H = sum_{k<=N} kappa_k/k! D_k``, ``D_k = e^{-xt.x} G^k e^{xt.x}``
        (``D_1 = theta = sum_f c_f ft``, ``D_{k+1} = G D_k + theta D_k``),
        ``N = SDE.series_order``.  The action gets ``-H`` (its theta-linear
        part, the mean drift, included); the mean-field equations get that
        linear part (MF only), as for a non-Gaussian source on the Ito route.
        The cumulant rates ``kappa_k`` stay outside the flow (a state-dependent
        rate is the pre-jump rate)."""
        from sage.all import factorial
        N = int(self._series_order)
        src._check_order(N, f"the Marcus series of source {name!r} (series order {N})")
        a = self._dt_coefficients()
        slope = (self._marcus_affine_flow(coup, a)
                 if self._marcus_route == 'auto' and src._max_order() is None else None)
        if slope is not None:
            self._marcus_exact(name, src, coup, slope, pop, source_terms, mf_rhs)
            return
        resp = {f: SR.var(f'{f}t__i') for f in coup}
        theta = sum((coup[f] * resp[f] for f in coup), SR(0))
        b = {f: coup[f] / a[f] for f in coup}
        top = src._max_order()
        n_kap = N + 1 if top is None or top >= N + 1 else N
        kap = {k: parse_expr(self._index(expr_to_text(src.cumulant(k))),
                             what=f'cumulant {k} of {name}')
               for k in range(1, n_kap + 1)}
        D = theta.expand()
        H, nxt = kap[1] * D, None
        for k in range(2, n_kap + 1):
            D = (sum((b[f] * D.diff(self._sym(f)) for f in coup), SR(0)) + theta * D).expand()
            term = kap[k] / factorial(k) * D
            if k <= N:
                H += term
            else:
                nxt = term
        H = H.expand()
        zero = {r: 0 for r in resp.values()}
        for f in coup:
            lin = H.diff(resp[f]).subs(zero)
            if not lin.is_zero():
                mf_rhs[f].append(lin)
                self._mf_only_log.append({'field': f, 'source': name, 'text': expr_to_text(lin)})
        self._emit(source_terms, pop, f'-{_wrap(expr_to_text(H))}', name)
        self.notes.append(
            f"Marcus: source {name!r} acts through the flow of its coupling; the "
            f"action carries sum_k kappa_k/k! D_k to order {N} (SDE.series_order); "
            f"the size of the next term at the saddle is result['marcus_truncation'].")
        self._marcus_terms[name] = {
            'route': 'series', 'order': N, 'population': pop,
            'responses': [f'{f}t' for f in coup], 'fields': list(self._fields),
            'kept': expr_to_text(H),
            'next': None if nxt is None else expr_to_text(nxt.expand())}

    def _marcus_affine_flow(self, coup, a):
        """The slopes ``s_f = d b_f/d x_f`` of the coupling ``G = sum_f b_f
        d/dx_f`` (``b_f = c_f/a_f``) when its flow is closed form: every
        ``b_f`` affine in its own field and free of the other coupled fields
        (a diagonal linear vector field; fields the source does not move may
        enter, being constant along the flow).  Then
        ``phi_z(x)_f - x_f = (b_f/s_f)(e^{s_f z} - 1)``, or ``b_f z`` where
        ``s_f = 0``.  ``None`` otherwise (the series route)."""
        moved = {f: self._sym(f) for f in coup}
        slope = {}
        for f, c in coup.items():
            g = SR(c) / a[f]
            if any(g.has(xh) for h, xh in moved.items() if h != f):
                return None
            s = g.diff(moved[f])
            if any(s.has(xh) for xh in moved.values()):
                return None
            slope[f] = s
        return slope

    def _marcus_exact(self, name, src, coup, slope, pop, source_terms, mf_rhs):
        """The Marcus noise term of a jump source with a closed-form flow
        (:meth:`_marcus_affine_flow`), exact.  With ``p = sum_f a_f ft x_f``,
        ``H = e^{-p} K(G) e^{p} = sum_n T[(sum_f w_f (e^{s_f z} - 1) + v z)^n]/n!``,
        ``w_f = c_f ft/s_f`` (``s_f != 0``), ``v = sum_{s_f = 0} c_f ft``, where
        the functional ``T`` takes ``z^m e^{sz}`` to ``K^{(m)}(s)``.  It is kept
        to ``n <= SDE.series_order`` response legs (as the Ito action keeps an
        irregular CGF): every kept vertex is exact, with no moment series in
        ``z``.  ``T`` acts term by term on ``K``: ``A e^{alpha theta}`` gives
        ``A (sum_f w_f (e^{alpha s_f} - 1) + v alpha)^n`` (a jump of size
        ``alpha``), ``A theta^p`` the ``p``-th ``z``-derivative at 0, and any
        other term the alternating binomial sum of its values at
        ``sum_f j_f s_f``, put over one denominator when rational (which removes
        its cancellation).  ``compute_cumulants`` refuses parameters at which
        the jump law's mgf does not exist where a usable vertex needs it, and
        sizes the rounding of the kept coefficients
        (``result['marcus_truncation']``)."""
        from itertools import product as _iprod
        from sage.all import binomial, factorial, exp as _exp
        N = int(self._series_order)
        resp = {f: SR.var(f'{f}t__i') for f in coup}
        moving = [f for f in coup if not SR(slope[f]).is_zero()]
        still = [f for f in coup if SR(slope[f]).is_zero()]
        K = parse_expr(self._index(expr_to_text(src.cgf())), what=f'K of {name}')
        shapes, bases = _cgf_shapes(K)
        z = SR.var('z__marcus')
        v = sum((coup[f] * resp[f] for f in still), SR(0))

        def T(mu, m):
            n = sum(mu) + m
            tot = SR(0)
            for kind, key, cf in shapes:
                if kind == 'exp':
                    t = key ** m
                    for f, q in zip(moving, mu):
                        t *= (_exp(key * slope[f]) - 1) ** q
                elif kind == 'poly':
                    if key < n:                 # the z-power series starts at z^n
                        continue
                    e = z ** m
                    for f, q in zip(moving, mu):
                        e *= (_exp(slope[f] * z) - 1) ** q
                    t = e.taylor(z, 0, key).coefficient(z, key) * factorial(key)
                else:
                    gm = key.diff(THETA, m) if m else key
                    t = SR(0)
                    for js in _iprod(*[range(q + 1) for q in mu]):
                        w = SR(1)
                        for q, j in zip(mu, js):
                            w *= binomial(q, j) * (-1) ** (q - j)
                        t += w * gm.subs({THETA: sum((j * slope[f] for f, j in zip(moving, js)),
                                                     SR(0))})
                    if _is_rational_in(key, THETA):
                        t = t.factor()
                tot += cf * t
            return tot

        H, coefs = SR(0), {}
        for n in range(1, N + 1):
            for mu in _compositions(n, len(moving) + (1 if still else 0)):
                mu, m = (mu[:-1], mu[-1]) if still else (mu, 0)
                P = T(mu, m)
                for f, q in zip(moving, mu):
                    P = P / slope[f] ** q
                if SR(P).is_zero():
                    continue
                mono = v ** m / factorial(m)
                for f, q in zip(moving, mu):
                    mono *= (coup[f] * resp[f]) ** q / factorial(q)
                coefs[','.join(map(str, tuple(mu) + ((m,) if still else ())))] = expr_to_text(P)
                H += P * mono
        zero = {r: 0 for r in resp.values()}
        for f in coup:
            lin = H.diff(resp[f]).subs(zero)
            if not lin.is_zero():
                mf_rhs[f].append(lin)
                self._mf_only_log.append({'field': f, 'source': name, 'text': expr_to_text(lin)})
        self._emit(source_terms, pop, f'-{_wrap(expr_to_text(H))}', name)
        self.notes.append(
            f"Marcus: source {name!r} acts through the closed-form flow of its affine "
            f"coupling; the action carries its exact noise term to {N} response legs "
            f"(SDE.series_order), with no moment series.")
        self._marcus_terms[name] = {
            'route': 'exact', 'order': N, 'population': pop,
            'responses': [f'{f}t' for f in coup], 'fields': list(self._fields),
            'slopes': {f: expr_to_text(SR(slope[f])) for f in coup},
            'cgf': expr_to_text(K), 'mgf_bases': [expr_to_text(b) for b in bases],
            'coefficients': coefs}

    # ── outputs ───────────────────────────────────────────────────────
    def action_text(self) -> str:
        return self._derive()['action']

    def equations(self) -> list[dict]:
        """The derived ``.equation(lhs=..., rhs=..., population=...)`` calls."""
        return [{k: e[k] for k in ('lhs', 'rhs', 'population')}
                for e in self._derive()['equations']]

    def to_builder(self):
        """A :class:`TemporalModelBuilder` with the derived action and
        mean-field equations set (continue by hand from here); for an SPDE
        (``space=d`` or a Laplacian), a :class:`SpatialModelBuilder`."""
        from api.model import TemporalModelBuilder
        d = self._derive()
        if d.get('spatial'):
            return self._spatial_builder(d)
        b = TemporalModelBuilder(self.name)
        pops = self._populations()
        for p in pops:
            b.population(p['name'], size=p['size'], description=p['description'])
        single = pops[0]['name'] if len(pops) == 1 else None
        for f, info in self._fields.items():
            kw = dict(population=self._pop_of_field(f), description=info['description'])
            if info['domain'] is not None:
                kw['domain'] = info['domain']
            if info['latex']:
                kw['latex'] = info['latex']
            b.physical_field(f, **kw)
        for (name,), kw in self._params:
            kw = dict(kw)
            if kw['indexed_by'] is None and kw['indexed'] and single:
                kw['indexed_by'] = ([single, single] if kw['indexed'] == 'matrix'
                                    else [single])
            b.parameter(name, **{k: v for k, v in kw.items() if k != 'indexed'
                                 or kw['indexed_by'] is None})
        for (name, args, expr), kw in self._functions:
            kw = dict(kw)
            if kw['population'] is None and single:
                kw['population'] = single
            b.define_function(name, args=args, expression=expr,
                              **{k: v for k, v in kw.items() if v is not None})
        for name, k in self._kernels.items():
            kw = {kk: v for kk, v in k.items() if v not in (None, '', False)}
            b.define_kernel(name, **kw)
        b.set_action_text(d['action'])
        star = self._star_equations(d['equations'])
        if star is not None:
            for sname, rhs in star:
                b.set_mf_equation(sname, rhs)
        else:
            for e in d['equations']:
                b.equation(lhs=e['lhs'], rhs=e['rhs'], population=e['population'])
        if self._stability is not None:
            b.stability_analysis(self._stability)
        return b

    def _spatial_builder(self, d):
        """The :class:`SpatialModelBuilder` of an SPDE: the calls a hand-written
        spatial model makes (fields of dimension ``space``, parameters, the
        scalar action -- in the operator IR when an operator acts on an
        expression --, the equations, the boundary and the stationary initial
        condition)."""
        from api.model import SpatialModelBuilder
        b = SpatialModelBuilder(self.name)
        for f, info in self._fields.items():
            kw = dict(spatial_dim=self._space, description=info['description'])
            if info['domain'] is not None:
                kw['domain'] = info['domain']
            if info['latex']:
                kw['latex'] = info['latex']
            b.physical_field(f, **kw)
        for (name,), kw in self._params:
            b.parameter(name, default=kw['default'], domain=kw['domain'],
                        description=kw['description'])
        for (name, args, expr), kw in self._functions:
            b.define_function(name, args=args, expression=expr,
                              **{k: v for k, v in kw.items()
                                 if v is not None and k != 'population'})
        b.set_action_text(d['action'])
        if d['operator_ir']:
            b.operator_ir()
        for e in d['equations']:
            b.equation(lhs=e['lhs'], rhs=e['rhs'])
        if self._stability is not None:
            b.stability_analysis(self._stability)
        bd = dict(self._boundary or {'mode': 'infinite'})
        b.boundary(bd.pop('mode'), **bd)
        b.initial('stationary')
        if self._dyson is not None:
            b.dyson(**self._dyson)
        return b

    def _star_equations(self, eqs):
        """For a model with a point process whose every mean-field equation
        reads ``x = rhs`` at Dt = 0: the saddle equations ``xstar = rhs*``
        (``set_mf_equation``), the form hand-written point-process models use
        (the same saddle, solved by the same route, bit for bit).  ``None``
        otherwise (the residual ``.equation`` form)."""
        if not any(s.get('pp') for s in self._sources.values()):
            return None
        names = sorted(self._fields, key=len, reverse=True)
        pat = re.compile(r'\b(' + '|'.join(map(re.escape, names)) + r')\s*\[')
        out = []
        for e in eqs:
            f = e['field']
            try:
                L = parse_expr(e['lhs'], what=f'lhs of {f}').subs({_DT: 0})
            except NoiseError:
                return None
            if not (L - self._sym(f)).simplify_full().is_zero():
                return None
            if (self._fields[f]['exposed'] is None
                    and re.search(rf'\b{re.escape(f)}\s*\[', e['rhs'])):
                # a continuous field with x = F(x, ...) (e.g. a conductance
                # term (Es - v)*Conv(g, n)): an implicit saddle equation, not
                # an explicit xstar = ... (which the engine substitutes as
                # written); the residual ``.equation`` form solves it.  (A
                # spike train's nstar = F(nstar) is the iteration saddle the
                # hand-written point-process models use: unchanged.)
                return None
            out.append((f'{f}star', pat.sub(lambda m: m.group(1) + 'star[', e['rhs'])))
        return out

    def simulate(self, **kw):
        """Simulate the model's point processes, drifts and Gaussian sources
        (:func:`api.pp_simulate.simulate`; keywords ``T``, ``dt``, ``n_paths``,
        ``burn_in``, ``bin``, ``params``, ``seed``)."""
        if self._derive().get('spatial'):
            raise SDEError(f'SDE({self.name!r}).simulate(): the SDE simulator is temporal; an '
                           f'SPDE is simulated with simulations/spatial_field_1d_sim / '
                           f'spatial_field_2d_sim.')
        from api.pp_simulate import simulate
        return simulate(self, **kw)

    def _emit(self, source_terms, pop, text, owner):
        """Add the noise term ``text`` of source ``owner`` to the action."""
        source_terms.setdefault(pop, []).append(text)
        self._action_log.append({'kind': 'noise', 'source': owner, 'text': text})

    def _check_derived_names(self):
        """Refuse a parameter, function or kernel named like a symbol the
        compiler derives from a field ``f``: its response field ``ft``, its
        fluctuation ``df`` or its saddle ``fstar`` (the clash was otherwise
        found only by the analyzer at run time)."""
        derived = {}
        for f in self._fields:
            for n, what in ((f + 't', 'response field'),
                            ('d' + f, 'fluctuation'),
                            (f + 'star', 'mean-field saddle')):
                derived[n] = (what, f)
        for kind, names in (('parameter', self._param_names),
                            ('function', self._function_names),
                            ('kernel', list(self._kernels))):
            for n in names:
                if n in derived:
                    what, f = derived[n]
                    raise SDEError(
                        f'{kind} name {n!r} is the {what} the compiler derives '
                        f'from field {f!r}; choose another name.')

    def build(self) -> dict:
        self._check_derived_names()
        m = self.to_builder().build()
        trunc = sorted(set(n for n, s in self._sources.items()
                           if not s.get('pp') and not getattr(s['source'], 'regular', True))
                       | set(n for n, t in self._marcus_terms.items() if t['route'] == 'series'))
        exact = sorted(n for n, t in self._marcus_terms.items() if t['route'] == 'exact')
        if exact:
            # the exact Marcus terms are kept to SDE.series_order response
            # legs; compute_cumulants refuses k + max_ell above it
            m['marcus_exact_order'] = {'order': self._series_order, 'sources': exact}
        if self._interp == 'marcus' or self._marcus_sources():
            m['interpretation'] = 'marcus'
        if self._marcus_terms:
            # the Marcus series of each jump source with a state-dependent
            # coupling, kept to SDE.series_order, and its next term:
            # compute_cumulants sizes that term at the saddle
            m['marcus_series'] = {n: dict(t) for n, t in self._marcus_terms.items()}
        if trunc:
            # the action holds these CGFs as Taylor polynomials of this order;
            # compute_cumulants refuses (k, max_ell) that would need more
            m['noise_series_order'] = {'order': self._series_order, 'sources': trunc}
        if self._jitter_uses:
            # the jittered exogenous inputs: compute_cumulants refuses them as
            # external legs and switch-on runs (the dropped shift is free only for
            # stationary statistics of the rest of the network) and records the fit
            m['exogenous_jitter'] = {
                src: {'sigma': u['sigma'], 'tol': u['tol'], 'kernels': list(u['kernels']),
                      'shift': f'{u["fit"].shift!r}*{u["sigma"]}', 'fit': u['fit'].record()}
                for src, u in self._jitter_uses.items()}
        declared = {n: s['source'].declared_order for n, s in self._sources.items()
                    if s.get('source') is not None and hasattr(s['source'], 'declared_order')}
        if declared:
            # Cumulants([...]) sources: cumulants above the declared order are 0;
            # compute_cumulants warns when (k, max_ell) can reach them
            m['noise_declared_orders'] = declared
        return m

    # ── diagnostics ───────────────────────────────────────────────────
    def check(self, k: int = 2, max_ell: int = 1, *, params: Optional[dict] = None,
              verbose: bool = True) -> 'SDEReport':
        """What the front end derives from the declarations, without any
        Phase J work.

        Returns an :class:`SDEReport` (a dict, printed as text when
        ``verbose``; ``str(report)`` is the text) with the keys

        * ``declarations``: fields (with their response fields), parameters,
          functions, kernels, populations, operators;
        * ``action``: the derived MSR action (``text``, identical to
          :meth:`action_text`) and its ``terms``: one ``dynamics`` term per
          field, one ``noise`` term per source (``source``, the source's type
          and parameters), the ``extra_action`` terms;
        * ``mean_field``: the mean-field equations (as :meth:`equations`);
        * ``corrections``: the interpretation-conversion drift terms
          (Stratonovich, Marcus Gaussian parts), each with its ``source``,
          and the Marcus noise terms of the jump sources (``route``,
          ``order``);
        * ``vertices``: the expanded action at the mean-field saddle for the
          cumulant order ``k`` with loops up to ``max_ell`` (Taylor order
          ``max(k + 2*max_ell, 2)``): one row per monomial with the numbers of
          response and field legs, the coefficient, the degree and the loop
          orders (for ``k`` external legs) it can enter; ``None`` when the
          model cannot be expanded (``analysis_error`` says why);
        * ``propagator``: the structure of the linear operator and the noise
          matrix (class, poles, stability, saddle), from the analyzer;
        * ``warnings``: ``{'code', 'message'}`` for a truncated Marcus series
          or noise CGF, terms that enter the mean field only, and the
          analyzer's warnings and errors.

        ``params`` overrides the declared parameter defaults for the
        saddle.  Nothing is integrated: the cost is the build and the
        expansion of the action (about a second for a small model).  The
        action and the equations are those of :meth:`action_text` and
        :meth:`equations`; ``check`` only reads them."""
        for what, v, lo in (('k', k, 1), ('max_ell', max_ell, 0)):
            try:
                from operator import index
                ok = not isinstance(v, bool) and index(v) >= lo      # (int, numpy and Sage integers)
            except TypeError:
                ok = False
            if not ok:
                raise SDEError(f'check({what}={v!r}): {what} must be an integer >= {lo}.')
        k, max_ell = index(k), index(max_ell)
        self._check_derived_names()
        d = self._derive()
        notes = list(self.notes)
        rep = SDEReport(
            name=self.name, interpretation=self._interp_mode(), space=self._space,
            declarations=self._declarations_report(),
            action=self._action_report(d),
            mean_field=[{k_: e[k_] for k_ in ('field', 'lhs', 'rhs', 'population')}
                        for e in d['equations']],
            corrections=self._corrections_report(),
            notes=notes, k=int(k), max_ell=int(max_ell))
        if rep['interpretation'] == 'mixed':
            rep['interpretations'] = {n: self._src_interp(n) for n in self._sources}
        warnings_ = self._truncation_warnings()
        rep['vertices'], rep['propagator'], rep['analysis_error'] = None, None, None
        try:
            model = self.build()
            self._analysis_report(rep, model, int(k), int(max_ell), params, warnings_)
        except Exception as exc:                          # noqa: BLE001
            rep['analysis_error'] = f'{type(exc).__name__}: {exc}'
            warnings_.append({'code': 'ANALYSIS_UNAVAILABLE', 'level': 'warning',
                              'message': f'the vertex table and the propagator could not be '
                                         f'derived ({type(exc).__name__}: {exc})'})
        rep['warnings'] = warnings_
        if verbose:
            print(rep.text())
        return rep

    def latex(self) -> dict:
        """LaTeX strings for the model: ``'sde'`` (the equations as written,
        ``dx = (...) dt + ... dW``, operators typeset as ``\\partial_t`` and
        ``\\nabla^2``), ``'noise'`` (the sources: ``W`` Gaussian, ``N`` Poisson
        or compound Poisson, ``L`` other Levy sources), ``'action'`` (the
        derived MSR action, as :meth:`action_text`) and ``'mean_field'`` (the
        stationary mean-field equations, fields at the saddle).  Each is an
        ``aligned`` environment (``'noise'`` is empty without sources).
        Nothing is compiled; the output depends on the declarations only."""
        from api.sde_latex import render_latex
        self._check_derived_names()
        return render_latex(self)

    def _declarations_report(self) -> dict:
        pops = self._populations()
        return {
            'populations': [{'name': p['name'], 'size': p['size']} for p in pops],
            'fields': [{'name': f, 'response': f + 't', 'population': info['population'],
                        'exposed_source': info['exposed'], 'description': info['description']}
                       for f, info in self._fields.items()],
            'parameters': [{'name': n, 'default': kw['default'], 'domain': kw['domain'],
                            'indexed_by': kw['indexed_by'] if kw['indexed_by'] is not None
                            else (['pop'] if kw['indexed'] else None)}
                           for (n,), kw in self._params],
            'functions': [{'name': n, 'args': list(a), 'expression': e}
                          for (n, a, e), _kw in self._functions],
            'kernels': [{'name': n, **{q: v for q, v in kn.items()
                                       if q in ('time_expr', 'freq_image', 'delay')
                                       and v is not None},
                         **({'gaussian_jitter': {
                             'sigma': self._jitter[n]['sigma'], 'tol': self._jitter[n]['tol'],
                             'states': self._jitter[n]['fit'].order,
                             'shift': f"{self._jitter[n]['fit'].shift!r}*{self._jitter[n]['sigma']}",
                             'factors': self._jitter[n].get('factors')}}
                            if n in self._jitter else {})}
                        for n, kn in self._kernels.items()],
            'operators': {n: type(o).__name__ for n, o in self._operators.items()},
        }

    @staticmethod
    def _describe_source(src) -> dict:
        if src is None:
            return {'type': 'PointProcess'}
        out = {'type': type(src).__name__}
        params = getattr(src, 'params', None)
        if params:
            out['params'] = {q: expr_to_text(v) for q, v in params.items()}
        jump = getattr(src, 'jump', None)
        if jump is not None:
            out['jump'] = {'type': type(jump).__name__,
                           'params': {q: expr_to_text(v) for q, v in jump.params.items()}}
        return out

    def _action_report(self, d) -> dict:
        terms = []
        for t in self._action_log:
            t = dict(t)
            if t['kind'] == 'noise':
                s = self._sources.get(t['source'].split(' (part')[0])
                if s is not None:
                    t['source_type'] = (self._describe_source(s.get('source')) if not s.get('pp')
                                        else {'type': 'PointProcess',
                                              'rate': s['pp']['rate'], 'size': s['pp']['size']})
                    t['couples'] = {f: str(c) for f, c in s['couples'].items()}
                    if s['expose']:
                        t['exposed_as'] = s['expose']
            terms.append(t)
        rank = {'dynamics': 0, 'noise': 1, 'extra': 2}
        terms.sort(key=lambda t: rank[t['kind']])               # (stable: the action's order)
        return {'text': d['action'], 'terms': terms,
                'operator_ir': bool(d.get('operator_ir'))}

    def _corrections_report(self) -> list:
        out = []
        for c in self._drift_log:
            out.append({'kind': 'ito_drift', 'rule': c['rule'], 'source': c['source'],
                        'field': c['field'], 'text': c['text'],
                        'meaning': f"{c['rule']} -> Ito: drift of {c['field']!r} += {c['text']}"})
        for name, t in self._marcus_terms.items():
            out.append({'kind': 'marcus_noise', 'rule': 'marcus', 'source': name,
                        'route': t['route'], 'order': t['order'],
                        'responses': list(t['responses']),
                        'meaning': (f"Marcus: the noise term of {name!r} is the "
                                    + ('exact ' if t['route'] == 'exact' else 'series ')
                                    + f"term kept to {t['order']} response legs")})
        return out

    def _truncation_warnings(self) -> list:
        w = []
        for name, t in self._marcus_terms.items():
            if t['route'] == 'series':
                w.append({'code': 'MARCUS_SERIES_TRUNCATED', 'level': 'warning',
                          'source': name,
                          'message': f"Marcus series of source {name!r} is truncated at order "
                                     f"{t['order']} (SDE.series_order); compute_cumulants "
                                     f"reports the size of the next term at the saddle "
                                     f"(result['marcus_truncation'])."})
            else:
                w.append({'code': 'MARCUS_EXACT_ORDER', 'level': 'info', 'source': name,
                          'message': f"Marcus noise term of source {name!r} is exact, kept to "
                                     f"{t['order']} response legs (SDE.series_order); "
                                     f"compute_cumulants refuses k + max_ell above it."})
        for name, s in self._sources.items():
            src = s.get('source')
            if src is not None and not getattr(src, 'regular', True):
                w.append({'code': 'NOISE_CGF_TAYLOR', 'level': 'warning', 'source': name,
                          'message': f"source {name!r}: its CGF has a removable singularity and "
                                     f"enters the action as a Taylor polynomial of order "
                                     f"{self._series_order} (SDE.series_order)."})
            if src is not None and hasattr(src, 'declared_order'):
                w.append({'code': 'NOISE_DECLARED_ORDER', 'level': 'info', 'source': name,
                          'message': f"source {name!r}: cumulants above order "
                                     f"{src.declared_order} are taken as 0."})
        for e in self._mf_only_log:
            w.append({'code': 'MEAN_FIELD_ONLY', 'level': 'info', 'source': e['source'],
                      'field': e['field'], 'text': e['text'],
                      'message': f"source {e['source']!r}: its mean (the term {e['text']}) enters "
                                 f"the mean-field equation of {e['field']!r} through the linear "
                                 f"response part of its noise term; it is no interaction vertex."})
        for n in self.notes:
            if not n.startswith(('Stratonovich -> Ito', 'Marcus:')):     # (under corrections)
                w.append({'code': 'NOTE', 'level': 'info', 'message': n})
        return w

    @staticmethod
    def _min_loop(degree: int, k: int) -> int:
        """Fewest loops of a connected k-point diagram that holds a vertex
        of the given degree: with V vertices, ``sum(d_v - 2) = k - 2 + 2*ell``,
        so ``ell >= (degree - k)/2`` (rounded up).  The enumerator applies the
        causality rules on top, so this is a necessary condition."""
        return max(0, -((k - degree) // 2))

    def _analysis_report(self, rep, model, k, max_ell, params, warnings_):
        from api.analyze import analyze_with_context, Question
        from sage.all import SR as _SR
        from api.analyze import Policy
        q = Question(k=k, max_ell=max_ell, parameters=params)
        # the SDE has declared its convention and derived a Taylor-truncated
        # action itself: the run-time consents are not what check() reports
        # (the derived action is Ito-ordered under Stratonovich as well)
        pol = Policy(interpretation='marcus' if self._interp == 'marcus' or self._marcus_sources()
                     else 'ito',
                     allow_taylor_truncation=True)
        report, ctx = analyze_with_context(model, q, pol, use_cache=False)
        tr = report.traits
        nt = int(ctx.n_tilde or 0)
        names = list(ctx.ring_names or [])
        sectors = ctx.sectors or {}
        order = max(k + 2 * max_ell, 2)

        def mono(exps):
            parts = [n if e == 1 else f'{n}^{e}' for n, e in zip(names, exps) if e]
            return '*'.join(parts) if parts else '1'

        verts = []
        for (a, b), poly in sorted(sectors.items(), key=lambda kv: (kv[0][0] + kv[0][1], kv[0])):
            if a + b < 3 or a < 1 or not isinstance(poly, dict):
                continue
            for exps, c in sorted(poly.items()):
                verts.append({
                    'response_legs': int(a), 'field_legs': int(b),
                    'degree': int(a + b), 'monomial': mono(exps),
                    'coefficient': expr_to_text(_SR(c)),
                    'kind': 'interaction' if b >= 1 else 'noise cumulant',
                    'loops': list(range(self._min_loop(a + b, k), max_ell + 1)),
                    'min_loop': {kk: self._min_loop(a + b, kk) for kk in range(1, max(k, 4) + 1)}})
        rep['vertices'] = {'taylor_order': order, 'k': k, 'max_ell': max_ell,
                           'rows': verts,
                           'species': [list(sp) for sp in sorted(sectors) if sum(sp) >= 3
                                       and sp[0] >= 1],
                           'note': 'loops: the loop orders (up to max_ell) at which a vertex of '
                                   'this degree can enter a k-point cumulant (degree count; '
                                   'causality prunes further)'}

        def sector_text(key):
            poly = sectors.get(key)
            if not isinstance(poly, dict):
                return None
            return {mono(e): expr_to_text(_SR(c)) for e, c in sorted(poly.items())}

        rep['propagator'] = {
            'class': tr.propagator_class, 'n_poles': tr.n_poles,
            'poles': [f'{complex(p).real:.6g}{complex(p).imag:+.6g}j' for p in (tr.poles or ())],
            'stable': tr.stable, 'kernel_stability': tr.kernel_stability,
            'noise_class': tr.noise_class, 'n_saddles': tr.n_saddles,
            'saddle': {n: [float(x) for x in v] for n, v in (tr.saddle or ())},
            'linear_operator': sector_text((1, 1)),
            'noise_matrix': sector_text((2, 0)),
            'spatial': bool(tr.spatial)}
        for f in report.findings:
            if f.level in ('warning', 'error'):
                warnings_.append({'code': f.code, 'level': f.level, 'message': f.message,
                                  'remedy': f.remedy})

    def show(self) -> str:
        """Print (and return) the derived action and mean-field equations as
        TemporalModelBuilder calls, copyable into a hand-written model."""
        d = self._derive()
        if d.get('spatial'):
            lines = [f"# SPDE {self.name!r} (space={self._space}; SpatialModelBuilder)",
                     f".set_action_text({d['action']!r})"]
            if d['operator_ir']:
                lines.append('.operator_ir()')
            for e in d['equations']:
                lines.append(f".equation(lhs={e['lhs']!r}, rhs={e['rhs']!r})")
        else:
            lines = [f"# SDE {self.name!r} ({self._interp_mode()})",
                     ".set_action_text('''", d['action'], "''')"]
            for e in d['equations']:
                lines.append(f".equation(lhs={e['lhs']!r}, rhs={e['rhs']!r}, "
                             f"population={e['population']!r})")
        for n in self.notes:
            lines.append(f'# note: {n}')
        out = '\n'.join(lines)
        print(out)
        return out

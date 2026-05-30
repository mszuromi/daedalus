"""
pipeline.spatial_operator_ir
============================
A small **operator intermediate representation** (IR) for the spatial
differential operators that appear in a field-theory action — the Laplacian
``∇²``, the time derivative ``∂_t``, and the partial spatial derivatives
``∂_{x_i}`` — together with the algebra and Fourier rules the diagrammatic
pipeline needs.

Why an IR instead of a bare ``SR.var('Laplacian')`` (the v1 approach) or Sage's
own differential objects:

* A bare *multiplicative* symbol (``D*Laplacian*phi``) loses **which field the
  derivative acts on** — ``phi*Laplacian*phi`` is ambiguous — and carries no
  operator algebra, so linearity and the homogeneous-saddle annihilation have to
  be hand-patched downstream.
* Sage's ``laplacian()`` (SageManifolds, Laplace–Beltrami) wants a manifold +
  metric + ``DiffScalarField`` objects and does **not** compose with the MSR
  field-doubling / multivariate-Taylor machinery, which works on commutative
  polynomial-ring generators.

So we host the operators as **inert Sage function applications** —
``Lap(phi)``, ``Dt(phi)``, ``Dx(phi, i)`` — which bind unambiguously to their
argument and give us free tree-walking / substitution / printing, while ALL
semantics live in explicit passes here:

* :func:`apply_linearity` — the operators are linear: they distribute over sums
  and pull out factors that are independent of the fields AND the spatial
  coordinates (constant coefficients).  ``∂(c·f) = c·∂f`` only when ``c`` is a
  genuine constant; a *position-dependent* coefficient ``f(x)·∇²φ`` is left
  ATOMIC (its product/Leibniz rule and the resulting ``f̂(p)`` momentum
  injection are a deliberately separate, later layer).
* :func:`expand_about_saddle` — substitute ``φ → φ̄ + δφ`` and re-linearize, so
  ``Lap(φ̄ + δφ) → Lap(φ̄) + Lap(δφ)``.  Linearity is applied; the mean term is
  KEPT.
* :func:`kill_means` — a SEPARATE, contingent pass: annihilate ``Lap(φ̄)`` (and
  ``Dx(φ̄,i)``, ``Dt(φ̄)``) ONLY when the saddle is homogeneous / stationary.
  An inhomogeneous mean-field solution (a front, a pattern) that solves the MF
  PDE keeps its ``Lap(φ̄)`` — it cancels the rest of the stationarity condition,
  it is not silently dropped.  This is why linearity and annihilation are two
  passes, not one.
* :func:`to_derived_generators` — replace each atomic operator-on-fluctuation
  ``Lap(δφ)``, ``Lap(Lap(δφ))``, ``Dx(δφ,i)`` with a fresh **ring generator**
  ``v``, so the existing ``FieldTheory.expand`` multivariate Taylor treats it
  like any other field (the ``u=δφ, v=∇²δφ`` trick).  Returns the rewritten
  expression plus a map ``generator → (base, operator-chain)``.
* :func:`form_factor` — the Fourier image of an operator chain on a leg of
  momentum ``k`` / frequency ``ω``:  ``Lap → −k²``, ``Dt → −iω``,
  ``Dx_i → i k_i``, composed multiplicatively (``Lap∘Lap → k⁴``).

Deferred (documented, not yet implemented): the Leibniz/product rule for
position-dependent coefficients and the corresponding ``f̂(p)`` injection;
operators nested *inside products* (``Lap(Lap(δφ)·ψ)``) beyond a single base.
"""
from __future__ import annotations

import operator as _pyop

from sage.all import SR, I, function

try:                                # Sage's n-ary +/* heads
    from sage.symbolic.operators import add_vararg as _ADD, mul_vararg as _MUL
except Exception:                   # pragma: no cover
    _ADD = _MUL = None


# ── the operator nodes (inert Sage function applications) ──────────
_LAP = function('Lap')              # ∇²(·)
_DT = function('Dt')                # ∂_t(·)
_DX = function('Dx')                # ∂_{x_i}(·, i)
_OP_NAMES = {'Lap', 'Dt', 'Dx'}


def Lap(expr):
    """``∇²(expr)`` — the (negative-eigenvalue) Laplacian operator node."""
    return _LAP(SR(expr))


def Dt(expr):
    """``∂_t(expr)`` — the time-derivative operator node."""
    return _DT(SR(expr))


def Dx(expr, i):
    """``∂_{x_i}(expr)`` — the i-th partial spatial derivative."""
    return _DX(SR(expr), int(i))


# ── tree helpers ──────────────────────────────────────────────────
def _head(e):
    try:
        return e.operator()
    except Exception:
        return None


def _is_add(e):
    h = _head(e)
    return h is _pyop.add or (_ADD is not None and h is _ADD)


def _is_mul(e):
    h = _head(e)
    return h is _pyop.mul or (_MUL is not None and h is _MUL)


def _op_name(e):
    """Return ``'Lap'`` / ``'Dt'`` / ``'Dx'`` if ``e`` is one of our operator
    nodes, else ``None``."""
    h = _head(e)
    nm = getattr(h, 'name', None)
    if callable(nm):
        try:
            n = nm()
        except Exception:
            return None
        if n in _OP_NAMES:
            return n
    return None


def _prod(factors):
    out = SR(1)
    for f in factors:
        out = out * f
    return out


def _syms(expr, names):
    """``True`` if ``expr`` depends on any symbol whose name is in ``names``."""
    want = set(names)
    return any(str(v) in want for v in SR(expr).variables())


def _as_names(syms):
    return {str(s) for s in syms}


# ── 1. linearity (the operator algebra — always valid) ────────────
def apply_linearity(expr, fields, coords=('x', 'y', 'z')):
    """Push every operator node through sums and constant coefficients.

    ``fields`` : the field symbols/names the operators act on.
    ``coords`` : spatial-coordinate names (default ``x,y,z``); a factor that
    depends on a coordinate is NOT pulled out (that is the position-dependent-
    coefficient case, handled by a later Leibniz/``f̂(p)`` layer).

    ``Lap(a·δφ + b·δψ) → a·Lap(δφ) + b·Lap(δψ)`` for field/coord-independent
    ``a,b``.  ``Lap(δφ·δψ)`` (a genuine derivative-of-a-product vertex) is left
    ATOMIC.
    """
    fnames = _as_names(fields)
    cnames = set(coords)
    blocked = fnames | cnames

    def _split_const(e):
        """(const_factor, rest) — pull out factors independent of fields+coords."""
        if _is_mul(e):
            consts, rest = [], []
            for f in e.operands():
                (rest if _syms(f, blocked) else consts).append(f)
            return _prod(consts), _prod(rest)
        return (SR(1), e) if _syms(e, blocked) else (e, SR(1))

    def _rebuild(name, arg, extra):
        if name == 'Lap':
            return _LAP(arg)
        if name == 'Dt':
            return _DT(arg)
        return _DX(arg, extra[0])               # Dx

    def _lin_node(name, arg, extra):
        # Expand the argument so binomial powers / products become sums BEFORE
        # distributing — e.g. Lap((φ̄+δφ)³) → Lap(φ̄³+3φ̄²δφ+3φ̄δφ²+δφ³) →
        # 3φ̄²·Lap(δφ) + 3φ̄·Lap(δφ²) + Lap(δφ³)  (Cahn–Hilliard ∇²φ³).
        arg = SR(arg).expand()
        if _is_add(arg):
            return sum(_lin_node(name, t, extra) for t in arg.operands())
        c, rest = _split_const(arg)             # c: field- AND coord-free factor
        if not _syms(rest, fnames):
            # No fluctuation field left inside (e.g. Lap(φ̄), Lap(φ̄²), or a pure
            # coordinate coefficient Lap(f(x))).  Keep the operator atomic: the
            # homogeneous-mean annihilation is kill_means' contingent job, and a
            # position-dependent coefficient is the deferred Leibniz layer.
            return _rebuild(name, arg, extra)
        if _is_add(rest):
            return c * sum(_lin_node(name, t, extra) for t in rest.operands())
        return c * _rebuild(name, rest, extra)

    def _walk(e):
        name = _op_name(e)
        if name is not None:
            ops = e.operands()
            arg = _walk(ops[0])
            extra = [ops[1]] if name == 'Dx' else []
            return _lin_node(name, arg, extra)
        if _is_add(e):
            return sum(_walk(t) for t in e.operands())
        if _is_mul(e):
            return _prod(_walk(f) for f in e.operands())
        kids = e.operands()
        if kids:
            try:
                return e.operator()(*[_walk(k) for k in kids])
            except Exception:
                return e
        return e

    return _walk(SR(expr))


# ── 2. saddle substitution (linearity applied, mean KEPT) ─────────
def expand_about_saddle(expr, replacements, fields=None, coords=('x', 'y', 'z')):
    """Substitute ``field → mean + fluct`` for each entry of ``replacements``
    (``{field_sym: (mean_sym, fluct_sym)}``) and re-apply linearity, so
    ``Lap(φ̄ + δφ) → Lap(φ̄) + Lap(δφ)``.  The mean term is RETAINED — use
    :func:`kill_means` (separately) to drop it for a homogeneous/stationary
    saddle.

    ``fields`` is the set of DYNAMICAL field symbols the operators act on
    non-trivially (fluctuations + response fields).  A homogeneous mean ``φ̄`` is
    NOT a field — it is a constant and pulls out of the operator
    (``Lap(φ̄·δφ²)→φ̄·Lap(δφ²)``), while ``Lap(φ̄)`` with no fluctuation left is
    kept atomic (for ``kill_means``).  Defaults to the fluctuation symbols of
    ``replacements``; the full pipeline passes every response/fluctuation field.
    """
    subs = {}
    flucts = []
    for fld, (mean, fluct) in replacements.items():
        subs[SR(fld)] = SR(mean) + SR(fluct)
        flucts.append(SR(fluct))
    out = SR(expr).subs(subs)
    return apply_linearity(out, fields if fields is not None else flucts,
                           coords=coords)


# ── 3. contingent annihilation on a homogeneous / stationary mean ─
def kill_means(expr, mean_syms, ops=('Lap', 'Dt', 'Dx')):
    """Annihilate ``Op(arg) → 0`` for ``Op`` in ``ops`` whenever ``arg`` depends
    ONLY on the given ``mean_syms`` (the saddle).  This encodes a CONTINGENT
    fact — a spatially homogeneous (for ``Lap``/``Dx``) and/or stationary (for
    ``Dt``) mean field — and is deliberately separate from the operator algebra
    in :func:`apply_linearity`.  For an inhomogeneous saddle, omit ``Lap`` from
    ``ops`` (or don't call this) so ``Lap(φ̄)`` survives to cancel the rest of
    the stationarity condition.
    """
    mnames = _as_names(mean_syms)
    ops = set(ops)

    def _walk(e):
        name = _op_name(e)
        if name is not None and name in ops:
            arg = e.operands()[0]
            if arg.variables() and _as_names(arg.variables()) <= mnames:
                return SR(0)
        kids = e.operands()
        if kids:
            try:
                return e.operator()(*[_walk(k) for k in kids])
            except Exception:
                return e
        return e

    return _walk(SR(expr))


# ── 4. lower atomic operator-on-fluctuation to ring generators ────
def to_derived_generators(expr, fluct_syms, prefix='Dg'):
    """Replace each atomic ``Op(δφ)`` / ``Op(Op(δφ))`` / ``Dx(δφ,i)`` (acting on
    a fluctuation) by a fresh **ring generator** symbol so the existing
    multivariate-Taylor ``expand`` treats it like an ordinary field (the
    ``u=δφ, v=∇²δφ`` trick).

    Returns ``(expr2, genmap)`` where ``genmap[gen] = (base_expr, op_chain)`` and
    ``op_chain`` is the bottom-up tuple of applied operators, e.g.
    ``(('Lap',),)`` or ``(('Lap',), ('Lap',))`` (∇⁴) or ``(('Dx', 0),)``.  The
    chain + leg momentum feed :func:`form_factor` at evaluation time.

    Bottom-up: an operator wrapping an already-introduced generator extends that
    generator's chain (so ``Lap(Lap(δφ))`` resolves to a single ∇⁴ generator).
    """
    fnames = _as_names(fluct_syms)
    genmap = {}                         # gen_sym -> (base_expr, chain_tuple)
    by_key = {}                         # (base_str, chain_tuple) -> gen_sym
    by_name = {}                        # gen name str -> gen_sym (membership)
    counter = [0]

    def _innermost_node(e):
        """Find an operator node none of whose operator-arguments contain a
        further operator node (an innermost Op)."""
        name = _op_name(e)
        if name is not None:
            inner = _innermost_node(e.operands()[0])
            return inner if inner is not None else e
        for k in e.operands():
            got = _innermost_node(k)
            if got is not None:
                return got
        return None

    def _gen_for(base, chain):
        key = (str(base), chain)
        if key in by_key:
            return by_key[key]
        counter[0] += 1
        nm = f'{prefix}{counter[0]}'
        g = SR.var(nm)
        by_key[key] = g
        by_name[nm] = g
        genmap[g] = (base, chain)
        return g

    cur = SR(expr)
    while True:
        node = _innermost_node(cur)
        if node is None:
            break
        name = _op_name(node)
        arg = node.operands()[0]
        idx = node.operands()[1] if name == 'Dx' else None
        op_entry = ('Dx', int(idx)) if name == 'Dx' else (name,)
        if str(arg) in by_name:                 # extend an existing chain (∇⁴…)
            base, chain = genmap[by_name[str(arg)]]
            g = _gen_for(base, (op_entry,) + chain)
        else:
            if not _syms(arg, fnames):
                # operator on a pure non-fluctuation arg: leave it (kill_means
                # should have removed homogeneous means; anything else is out of
                # scope for v1 and kept symbolic).
                # Substitute a unique passthrough to avoid an infinite loop.
                g = _gen_for(arg, (op_entry,) + (('__mean__',),))
            else:
                g = _gen_for(arg, (op_entry,))
        cur = cur.subs({node: g})
    return cur, genmap


# ── 4b. compose the passes: action → (fields + derived generators) ─
def prepare_action(S, fields, replacements=None, homogeneous=True,
                   coords=('x', 'y', 'z')):
    """Run the full IR preprocessing of an action ``S`` ahead of the
    multivariate-Taylor expansion.

    Two authoring conventions are supported:

    * **fluctuation fields** (``replacements=None``): the action is already
      written in the fluctuation fields (the framework's default — e.g.
      ``reaction_diffusion`` at ``φ*=0``).  We only ``apply_linearity`` and
      lower operators to derived generators.
    * **full fields + explicit saddle** (``replacements={field:(mean,fluct)}``):
      we ``expand_about_saddle`` (substitute + linearity, mean retained), then
      — for a homogeneous/stationary saddle — ``kill_means``, then lower.

    Returns ``(S_gen, genmap)``: the action rewritten with each atomic
    ``Op(δφ)`` replaced by a fresh ring generator, plus
    ``genmap[gen]=(base, op_chain)`` for the Fourier lowering (:func:`fourier_lower`).
    """
    if replacements:
        S = expand_about_saddle(S, replacements, fields=fields, coords=coords)
        if homogeneous:
            S = kill_means(S, [SR(m) for (m, _) in replacements.values()])
        flucts = [SR(f) for (_, f) in replacements.values()]
    else:
        S = apply_linearity(S, fields, coords=coords)
        flucts = [SR(f) for f in fields]
    return to_derived_generators(S, flucts)


def fourier_lower(expr, genmap, k, omega=None):
    """Substitute every derived generator by its Fourier image: ``g →
    form_factor(chain, k, ω)·base``.  Used to read the bilinear kernel
    ``K(ω,k)`` off the (1,1) sector and to attach per-leg form factors to
    vertices — the bridge from the IR to the k-explicit propagator (Phase 3).
    """
    subs = {g: form_factor(chain, k, omega) * SR(base)
            for g, (base, chain) in genmap.items()}
    return SR(expr).subs(subs)


# ── 5. Fourier form factor of an operator chain ───────────────────
def form_factor(chain, k, omega=None):
    """The Fourier image multiplier of an operator ``chain`` acting on a leg of
    spatial momentum ``k`` (a vector / sequence) and frequency ``omega``:

        ``Lap → −|k|²``,  ``Dt → −i ω``,  ``Dx_i → i k_i``,

    composed multiplicatively over the chain (bottom-up order is irrelevant —
    the factors commute).  ``chain`` is the tuple from
    :func:`to_derived_generators`, e.g. ``(('Lap',),)`` → ``−|k|²``,
    ``(('Lap',),('Lap',))`` → ``|k|⁴``, ``(('Dx',0),)`` → ``i k_0``.
    """
    kv = [SR(c) for c in (k if hasattr(k, '__iter__') else [k])]
    k2 = sum(c * c for c in kv)
    out = SR(1)
    for entry in chain:
        op = entry[0]
        if op == 'Lap':
            out *= -k2
        elif op == 'Dt':
            if omega is None:
                raise ValueError('form_factor: Dt needs omega')
            out *= -I * SR(omega)
        elif op == 'Dx':
            out *= I * kv[int(entry[1])]
        elif op == '__mean__':
            out *= SR(0)                 # an un-annihilated mean derivative
        else:
            raise ValueError(f'form_factor: unknown operator {op!r}')
    return out

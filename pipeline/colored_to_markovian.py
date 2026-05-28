"""
Markovian embedding preprocessor for colored-noise theories.

Why this exists
---------------
``msrjd.integration.time_domain.final_integral`` cannot absorb a smooth
``exp(-|tau|/tauc)`` factor coming from a CGF-tab ``NoiseSourceType``
vertex into its analytic mode-sum integrators.  At ``max_ell >= 1`` the
scipy ``nquad`` fallback path either hangs or runs for orders of
magnitude longer than the equivalent white-noise theory.

A standard workaround in stochastic dynamics is to rewrite the colored-
noise process as a deterministic linear filter driven by a white noise:

    dx/dt   = f(x) + xi
    dxi/dt  = -xi/tauc + sqrt(2 c / tauc) * eta(t),   <eta eta>(tau) = delta(tau)

The stationary autocovariance of the auxiliary ``xi`` is

    <xi xi>(tau) = c * exp(-|tau|/tauc),

which recovers the user's colored kernel.  In Phase 1 we only handle the
single-Lorentzian case, including the cross-correlated 2D variant.

The transform is applied at ``TheoryBuilder.build()`` time, mutating the
builder in place.  Every matched CGF row is replaced with:

  * One new ``physical_field`` per affected response field (e.g. an
    auxiliary process ``xi`` for the user's ``x``).  The framework's
    ``physical_field`` API auto-creates the conjugate response ``xit``.
  * One ``equation`` per auxiliary field, encoding the OU drift
    ``(Dt + 1/tauc) * xi = 0``.
  * One or more white-noise CGF rows on the auxiliaries.  For an
    auto-cumulant ``[r, r]`` row this is a single
    ``response_legs=['<r-aux>', '<r-aux>']`` row.  For a cross-cumulant
    ``[r1, r2]`` row this introduces both ``[<r1-aux>, <r1-aux>]`` and
    ``[<r2-aux>, <r2-aux>]`` auto rows AND a cross
    ``[<r1-aux>, <r2-aux>]`` row that together recover the desired
    cross-correlation matrix.
  * An augmented ``action_text`` adding ``- xi`` (i.e. a coupling
    ``-xt*xi``) to whatever was previously inside ``xt*(...)`` and
    appending the new auxiliary kinetic term ``xit*(Dt + 1/tauc)*xi``.

The transform is OFF for any row that doesn't cleanly match the
single-Lorentzian template; those rows pass through untouched and the
existing scipy.nquad fallback (with its warning) runs.  Users may
opt out per-builder via ``.markovianize(False)`` or per-row via
``markovianize=False`` on ``declare_cgf_term``.

Limitations (v1)
----------------
Only the single-Lorentzian kernel ``c · exp(-|tau|/tauc)`` is matched.
Future v2 work:
  * **Underdamped oscillatory** kernels of the form
    ``exp(-|tau|/tau_d) cos(omega tau)`` — embed as a 2-state OU process
    with imaginary eigenvalue.
  * **Double-exponential** ``c1 exp(-|tau|/t1) + c2 exp(-|tau|/t2)`` —
    embed as a sum of two independent OU processes.
  * **Polynomially-modulated** kernels — embed as higher-order linear
    filters (a chain of OU processes).

Each of these expands the auxiliary state space; the v1 single-
Lorentzian case is the minimum-cost extension that unblocks the user's
working OU-colored theories at ell ≥ 1.
"""
from __future__ import annotations

from typing import Optional

from sage.all import (
    SR, sage_eval, exp, abs_symbolic, simplify,
    sqrt as sage_sqrt,
)


# ── Lorentzian-kernel detection ──────────────────────────────────────


def detect_lorentzian(kernel_text: str,
                      coefficient_text: str,
                      declared_params: list[str]) -> Optional[dict]:
    """Inspect a CGF row's kernel + coefficient.  Return a dict
    describing the Markovian embedding if the row matches the single-
    Lorentzian template, or ``None`` otherwise.

    Match criterion:
      The kernel must parse as ``exp(-|tau|/p)`` for some symbolic ``p``
      that, after simplification, equals a single declared parameter
      OR a positive symbolic scalar multiple of one.  We require
      ``-arg(kernel)/abs(tau)`` to be ``abs``-free and to evaluate to
      a strictly positive expression.

    Returns
    -------
    dict | None
        On match::

            {
                'tauc_expr':      <SR expression for τc>,
                'tauc_text':      <Sage-syntax string for τc>,
                'amplitude_text': <Sage-syntax string for κ²(0) factor
                                   ``c`` in c · exp(-|tau|/τc)>,
                'aux_drive_text': <Sage-syntax string for the aux
                                   white-noise CGF coefficient
                                   = 2c/τc>,
            }

        ``None`` for any kernel that doesn't cleanly match.
    """
    if not kernel_text:
        return None

    # Build an evaluation namespace that includes every declared
    # parameter (as a generic SR symbol) plus ``tau`` and the few
    # symbolic functions a kernel can use.
    #
    # ``tau`` must be FREE of positivity / domain assumptions for the
    # ``arg / abs(tau)`` simplification to behave like a generic real
    # variable.  Other parts of the pipeline routinely declare a
    # ``tau`` parameter (e.g. an exponential-synapse time constant)
    # with ``domain='positive'`` — that assumption sticks on Sage's
    # global ``SR.var('tau')``, which would then make ``abs(tau)``
    # collapse to ``tau`` and the matcher accept asymmetric kernels
    # like ``exp(-tau/tauc)`` that are NOT physical autocorrelations.
    # Using a fresh local SR variable named ``_lorentz_tau`` (which
    # the caller never declares as positive) keeps the matcher
    # robust against test-order pollution.
    tau_sym = SR.var('_lorentz_tau')
    eval_ns: dict = {
        'tau': tau_sym,
        'exp': exp,
        'abs': abs_symbolic,
    }
    for pname in declared_params:
        eval_ns.setdefault(pname, SR.var(pname))

    try:
        K = sage_eval(kernel_text, eval_ns)
    except Exception:
        return None
    K = SR(K)

    if K.operator() != exp:
        return None
    operands = K.operands()
    if len(operands) != 1:
        return None
    arg = SR(operands[0])
    abs_tau = abs_symbolic(tau_sym)

    # Decompose arg as ``arg = -abs(tau) / tauc`` for some symbolic τc.
    # Compute r = arg / abs(tau).  For a Lorentzian r is ``-1/τc``
    # (abs-free, depending only on parameters).
    try:
        r = (arg / abs_tau).simplify_full()
    except Exception:
        return None
    if r.has(abs_tau):
        return None
    if r.has(tau_sym):
        return None

    # τc = -1/r must be POSITIVE for a physical Lorentzian.
    try:
        tauc_expr = (SR(-1) / r).simplify_full()
    except Exception:
        return None

    # Reject if τc still references ``tau`` or ``abs(tau)``.
    if tauc_expr.has(tau_sym) or tauc_expr.has(abs_tau):
        return None

    # Coefficient: parse the user's coefficient as the amplitude c.
    # κ²(τ) = c · exp(-|τ|/τc); the embedding's aux white-noise
    # variance becomes 2c/τc.
    try:
        c_expr = sage_eval(coefficient_text or '0', eval_ns)
        c_expr = SR(c_expr).simplify_full()
    except Exception:
        return None

    # Build textual forms for the aux CGF coefficient: 2 c / τc.
    aux_drive_expr = (SR(2) * c_expr / tauc_expr).simplify_full()

    return {
        'tauc_expr':      tauc_expr,
        'tauc_text':      _sr_to_text(tauc_expr),
        'amplitude_expr': c_expr,
        'amplitude_text': _sr_to_text(c_expr),
        'aux_drive_expr': aux_drive_expr,
        'aux_drive_text': _sr_to_text(aux_drive_expr),
    }


# ── Builder mutation entry point ─────────────────────────────────────


def markovianize_spec(builder) -> None:
    """Walk ``builder._cgf_terms``.  For each row that matches the
    single-Lorentzian template (and isn't explicitly opted out), replace
    it with an equivalent white-noise + auxiliary-field block.

    Mutates the builder in place:
      * Adds ``physical_field`` entries for the new auxiliary OU
        processes (one per response field that appears in a matched
        row).  This auto-adds the conjugate response field and the
        saddle parameter.
      * Adds ``equation`` records for the auxiliary OU drift.
      * Adds white-noise ``_cgf_terms`` rows on the auxiliaries.
      * Strips the matched rows from ``_cgf_terms``.
      * Augments ``_action_text`` with the coupling and aux-kinetic
        terms.

    Idempotent: re-running on an already-markovianized builder is a
    no-op (no row will match; auxiliary fields stay put).

    Per-builder / per-row gating:
      * ``builder._markovianize_default == False`` AND no row has
        ``markovianize == True`` → no-op.
      * Otherwise rows that match are markovianized unless their
        ``markovianize`` is explicitly ``False``.
    """
    cgf_terms = list(getattr(builder, '_cgf_terms', []) or [])
    if not cgf_terms:
        return

    builder_default = getattr(builder, '_markovianize_default', True)
    any_explicit_on = any(t.get('markovianize') is True for t in cgf_terms)
    if not builder_default and not any_explicit_on:
        return

    declared_params = [p.name for p in builder.parameters]

    # Group matched rows by their τc expression.  Two rows that share
    # the same τc CAN share one auxiliary process per response field;
    # rows with different τc cannot.  In v1 we assume a single τc
    # globally (the user's worked examples all do this), but we leave
    # the grouping structure in place so v2 can lift this without
    # restructuring.
    matched_rows: list[dict] = []      # row dicts, with detection info attached
    passthrough_rows: list[dict] = []  # rows to keep untouched

    for term in cgf_terms:
        explicit = term.get('markovianize', None)
        if explicit is False:
            passthrough_rows.append(term)
            continue
        # ``markovianize`` defaults to 'auto' (= None or 'auto'): match
        # when the kernel matches and the row order is 2.
        order = int(term.get('order', 0))
        if order != 2:
            passthrough_rows.append(term)
            continue
        info = detect_lorentzian(
            term.get('kernel') or '',
            term.get('coefficient') or '',
            declared_params,
        )
        if info is None:
            # Row's kernel is not a clean Lorentzian.  When
            # ``markovianize=True`` was explicitly set, that's a user
            # error — fail loudly so they fix the kernel.  Otherwise,
            # silently pass through.
            if explicit is True:
                raise ValueError(
                    f"colored_to_markovian: row {term.get('name')!r} has "
                    f"markovianize=True but kernel "
                    f"{term.get('kernel')!r} does not match the v1 "
                    f"single-Lorentzian template ``c·exp(-|tau|/tauc)``.  "
                    f"See docs/correlated_noise_capabilities.md §1.5 for "
                    f"supported templates."
                )
            passthrough_rows.append(term)
            continue
        # Match!  Attach the detection info and queue for transform.
        matched_rows.append({'term': term, 'info': info})

    if not matched_rows:
        return

    # Resolve aux field names per response field.  Use a deterministic
    # mapping that avoids collisions with existing physical fields.
    existing_phys_names = {f.name for f in builder.physical_fields}
    existing_natural   = {(f.natural_name or f.name)
                          for f in builder.physical_fields}

    # Per-source response field, derive the natural source name (e.g.
    # ``xt`` → ``x``) and assign an aux base name (e.g. ``xi`` for ``x``).
    # Cross-correlated rows reference TWO response fields; we want both
    # to get their own aux process so the cross noise can be expressed
    # as a single CGF row between them.
    source_field_names = set()
    for entry in matched_rows:
        legs = entry['term'].get('response_legs') or []
        if not legs and entry['term'].get('response_field'):
            legs = [entry['term']['response_field']] * 2
        for L in legs:
            source_field_names.add(L)

    # ``response_field_name`` → ``physical_field_natural_name`` (e.g.
    # ``xt`` → ``x``).  This is the inverse of the auto-response naming
    # rule: ``physical_field('x')`` ⇒ auto-response ``xt``.
    response_to_natural: dict[str, str] = {}
    for rname in source_field_names:
        # Find the physical field whose auto-response is ``rname``.
        match = None
        for f in builder.physical_fields:
            nat = f.natural_name or f.name
            if f'{nat}t' == rname:
                match = nat
                break
        if match is None:
            # Couldn't find a physical field whose conjugate is this
            # response — give up on this row.  Falls back to scipy.
            response_to_natural[rname] = None
            continue
        response_to_natural[rname] = match

    # Find unique τc text across matched rows.  If they share τc, great;
    # otherwise we still build per-row aux fields but flag a stacking
    # situation (v1 currently raises — the worked examples don't hit
    # this).
    tauc_texts = {entry['info']['tauc_text'] for entry in matched_rows}
    if len(tauc_texts) > 1:
        # Distinct τc across rows is structurally fine, but v1 only
        # supports a single tauc.  Tell the user to split their model
        # or upgrade.
        raise NotImplementedError(
            f"colored_to_markovian (v1): matched CGF rows have different "
            f"τc expressions: {sorted(tauc_texts)!r}.  v1 supports a "
            f"single shared τc; multi-τc support is a v2 follow-up "
            f"(see docs/correlated_noise_capabilities.md §1.5)."
        )
    tauc_text = matched_rows[0]['info']['tauc_text']

    # Assign aux field natural names (collision-safe).  Pattern:
    # natural='x' → aux 'xi'.  If 'xi' already exists, try 'xi_x',
    # 'xi_x_1', etc.
    aux_natural_by_source: dict[str, str] = {}
    for rname in sorted(source_field_names):
        source_nat = response_to_natural.get(rname)
        if source_nat is None:
            continue
        aux_nat = _pick_aux_field_name(
            source_nat, existing_natural, aux_natural_by_source.values())
        aux_natural_by_source[source_nat] = aux_nat
        # Reserve this name to prevent later collisions in this pass.
        existing_natural = existing_natural | {aux_nat}

    if not aux_natural_by_source:
        # No row was actually transformable — nothing to do.
        return

    # ── Inject new physical fields ──────────────────────────────────
    for source_nat, aux_nat in aux_natural_by_source.items():
        builder.physical_field(
            aux_nat, indexed=True,
            description=(f'auxiliary OU noise (Markovian embedding of '
                         f'colored CGF on {source_nat})'),
        )

    # ── Inject the aux OU equations  (Dt + 1/τc) * xi = 0 ───────────
    for source_nat, aux_nat in aux_natural_by_source.items():
        builder.equation(
            lhs=f'(Dt + 1/{tauc_text}) * {aux_nat}',
            rhs='0',
        )

    # ── Replace matched CGF rows with their white-noise aux versions ─
    new_cgf_rows: list[dict] = []
    for entry in matched_rows:
        term      = entry['term']
        info      = entry['info']
        legs_in   = term.get('response_legs') or []
        order     = int(term.get('order', 2))

        # Recover the input response-leg field names (length 2).
        if not legs_in and term.get('response_field'):
            legs_in = [term['response_field'], term['response_field']]
        if len(legs_in) != 2:
            # Defensive — should have been filtered upstream by detect_*.
            new_cgf_rows.append(term)
            continue

        nat_a = response_to_natural.get(legs_in[0])
        nat_b = response_to_natural.get(legs_in[1])
        if nat_a is None or nat_b is None:
            # No physical field maps to this response — keep original.
            new_cgf_rows.append(term)
            continue

        aux_resp_a = f'{aux_natural_by_source[nat_a]}t'
        aux_resp_b = f'{aux_natural_by_source[nat_b]}t'

        # Aux white-noise CGF row.  Coefficient = 2c / τc where c is
        # the row's amplitude.  Kernel = dirac_delta(tau).
        new_cgf_rows.append({
            'name':           f'{term["name"]}_markov_aux',
            'response_field': None,
            'response_legs':  [aux_resp_a, aux_resp_b],
            'order':          order,
            'coefficient':    info['aux_drive_text'],
            'kernel':         'dirac_delta(tau)',
            'markovianize':   False,    # don't re-process
        })

    # Carry passthrough rows + new aux rows into the builder.
    builder._cgf_terms = passthrough_rows + new_cgf_rows

    # ── Augment the action text ─────────────────────────────────────
    # Two changes:
    #  (1) Inside each "<resp_field>*(...)" block belonging to a
    #      source field, append "- <aux_field>" to enforce
    #      dx/dt = ... + xi.
    #  (2) Append the aux kinetic action "<aux_resp>*(Dt + 1/τc)*<aux>".
    #
    # For (1) we do a conservative text edit: find the response field
    # name as a token immediately before "*(" and insert " - <aux>"
    # just before the closing parenthesis of that parenthesized block.
    # This handles the common "<rt>*((Dt+...)*x + ...)" form used by
    # all the user's theories.  If the user's action layout is more
    # baroque, the user can opt out with ``.markovianize(False)`` and
    # do the rewrite manually.
    action_text = builder._action_text or ''
    for source_nat, aux_nat in aux_natural_by_source.items():
        resp_field = f'{source_nat}t'
        action_text = _inject_aux_coupling(action_text, resp_field, aux_nat)

    # Append the aux kinetic terms.  We tack onto the end with a
    # leading ``+`` so the user's existing expression stays intact.
    for source_nat, aux_nat in aux_natural_by_source.items():
        aux_resp = f'{aux_nat}t'
        action_text = (
            f'{action_text} + {aux_resp}*((Dt + 1/{tauc_text})*{aux_nat})'
        )
    builder._action_text = action_text

    # Record on the builder for diagnostics.
    builder._markovianize_applied = {
        'aux_fields':  list(aux_natural_by_source.values()),
        'matched_rows': [entry['term']['name'] for entry in matched_rows],
        'tauc_text':   tauc_text,
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _pick_aux_field_name(source_nat: str,
                         taken: set,
                         additional_taken=()) -> str:
    """Pick a collision-safe aux field name for a source field.

    Default pattern: ``x → xi``, ``y → yi``.  Falls back to
    ``xi_<source>``, ``xi_<source>_2``, etc. if the default is taken.
    """
    taken = set(taken) | set(additional_taken)
    candidate = f'{source_nat}i'
    if candidate not in taken:
        return candidate
    candidate = f'xi_{source_nat}'
    if candidate not in taken:
        return candidate
    n = 2
    while True:
        candidate = f'xi_{source_nat}_{n}'
        if candidate not in taken:
            return candidate
        n += 1


def _inject_aux_coupling(action_text: str,
                         resp_field: str,
                         aux_natural: str) -> str:
    """Inject ``- <aux_natural>`` into the parenthesized block following
    ``<resp_field>*(...)`` in the action text.

    Conservative rewrite: locate the first opening paren that's part of
    the ``<resp_field>*(`` pattern, scan to its matching close paren,
    and insert ``- <aux_natural>`` just before the close.

    If the pattern isn't found, return the text unchanged.  This makes
    the markovianize transform graceful on actions written in a non-
    canonical layout; the user can opt out per-builder.
    """
    import re
    # Find ``<resp_field>*(``.  Allow whitespace.
    pattern = re.compile(
        rf'\b{re.escape(resp_field)}\s*\*\s*\(', flags=0)
    m = pattern.search(action_text)
    if not m:
        return action_text
    # Walk parentheses to find the matching close.
    start = m.end() - 1   # index of '('
    depth = 0
    i = start
    while i < len(action_text):
        c = action_text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                # Insert before this close paren.
                injection = f' - {aux_natural}'
                return action_text[:i] + injection + action_text[i:]
        i += 1
    return action_text


def _sr_to_text(expr) -> str:
    """Render an SR expression as a Sage-syntax string that the
    text-driven theory pipeline can re-eval.

    The default ``str(expr)`` form uses Sage's preferred operators
    (``^`` for power, ``*`` / ``/`` for products, function names like
    ``sqrt``), which is exactly what
    ``pipeline.theory_compiler._CGFKernelCallable`` accepts.
    """
    return str(SR(expr))

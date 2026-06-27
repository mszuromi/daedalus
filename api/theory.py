"""
api.theory — declarative theory input (the primary authoring path).

The MSR-JD model dict (the example ``HAWKES_MODEL`` structure) carries
several Sage-aware lambdas (action, mf_bg_conditions, kernel_ft_image,
phi_concrete, mf_equations).  Writing one by hand requires familiarity
with Sage's symbolic algebra and the framework's conventions.  This
module is the high-level alternative: a fluent builder that takes
plain Sage-syntax strings, compiles them to those lambdas, and emits
the model dict for you.

``TemporalTheoryBuilder`` describes time-only (SDE) theories;
``SpatialTheoryBuilder`` is the spatial (SPDE) analogue.  Both subclass
the same accumulator, share the string-based authoring methods, and
produce a dict consumable directly by ``api.compute_cumulants``.

Typical use::

    from api.theory import TemporalTheoryBuilder

    model = (TemporalTheoryBuilder("My model")
             .physical_field("x", description="...")
             .parameter("mu",  default=1.0,  domain="positive")
             .parameter("eps", default=0.05, domain="positive")
             .parameter("D",   default=1.0,  domain="positive")
             .set_action_text("xt*((Dt+mu)*x + eps*x^3) - D*xt^2")
             .equation(lhs="(Dt+mu)*x", rhs="-eps*x^3")  # or .set_mf_equation("xstar", ...)
             .build())

Response fields (``xt`` above) are introduced automatically from the
action text; you only declare the physical fields and parameters.

Reusable actions live in ``api.theory_templates`` (e.g. ``HawkesAction``)
and are applied via ``.use_action_template(...)`` instead of
``.set_action_text(...)``.  For unusual theories that no string or
template covers, the lambda hooks (``.set_action(...)`` etc.) remain
as an optional escape hatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sage.all import SR


@dataclass
class FieldSpec:
    name: str
    indexed: bool
    latex: str
    description: str = ''
    natural_name: Optional[str] = None   # e.g. 'dn' has natural_name='n'
    population: Optional[str] = None     # heterogeneous-pop annotation
    spatial_dim: int = 0                 # 0 = time-only (default); d≥1 =
                                         # continuous spatial field φ(x, t)
                                         # in d dimensions.  Per-field by
                                         # design (see docs/spatial_design_
                                         # decisions_v1.md D1); a builder-
                                         # level ``.spatial_dim(d)`` bulk-
                                         # sets all fields.  v1 requires all
                                         # spatial fields to agree on d.


@dataclass
class ParameterSpec:
    name: str
    indexed: bool = False         # True = vector, "matrix" = matrix
    matrix: bool = False
    domain: Optional[str] = None
    default: Any = None
    description: str = ''
    mean_field: bool = False      # True for saddle-point quantities (n*, v*, ...)
    natural_name: Optional[str] = None   # e.g. 'nstar' has natural_name='n'
    indexed_by: Optional[list[str]] = None  # heterogeneous-pop annotation:
                                            # [] = scalar, ['E'] = vector,
                                            # ['E', 'I'] = matrix


@dataclass
class KernelSpec:
    name: str
    sage_name: str = ''           # internal SR var name (default 'z_'+name)
    latex_name: str = ''
    frequency_image: Optional[Callable] = None  # lambda omega, params: SR
    description: str = ''
    indexed: bool = False         # True / 'vector' for g[i]; 'matrix' for g[i, j]
    matrix:  bool = False         # True ⇒ N×N per-pair kernel (g[i, j])
    indexed_by: Optional[list] = None  # heterogeneous-pop annotation:
                                       # [] = scalar, ['E'] = vector,
                                       # ['E', 'I'] = matrix


class _SpatialMethods:
    """Spatial-only authoring methods (mixed into SpatialTheoryBuilder
    and the back-compat TheoryBuilder)."""

    def spatial_dim(self, d: int):
        """Bulk-set the spatial dimension of every physical field.

        Convenience for the common single-dimension case (D1 in
        ``docs/spatial_design_decisions_v1.md``).  Sets ``spatial_dim
        = d`` on every physical field already declared AND establishes
        ``d`` as the default for fields declared afterwards.  Per-field
        ``physical_field(spatial_dim=...)`` still overrides this.

        Underneath, spatial dimension is per-field (on ``FieldSpec``);
        this method is purely an ergonomic surface.  Call it before or
        after ``physical_field`` declarations — fields declared before
        get retro-set, fields declared after inherit the default.

        ``d = 0`` reverts to time-only.  ``d ∈ {1, 2, 3}`` are validated
        end-to-end (``d ≥ 2`` uses the radial inverse-FT and is
        infinite-boundary only; periodic is ``d = 1``).  All spatial
        fields must share ONE dimension (mixed-dimension is future work,
        enforced in :meth:`build`).
        """
        d = int(d)
        self._default_spatial_dim = d
        for f in self.physical_fields:
            f.spatial_dim = d
        return self

    def boundary(self, mode: str, **params):
        """Declare the spatial boundary condition.

        Parameters
        ----------
        mode : {'infinite', 'periodic'}
            ``'infinite'`` — unbounded domain (the default if
            ``.boundary`` is never called on a spatial theory).
            ``'periodic'`` — periodic cell; requires a ``length``.
        length : str or float, optional (periodic only)
            Spatial period ``L``.  A **string** names a declared
            ``.parameter`` (sweepable; the recommended form, e.g.
            ``length='L'`` with ``.parameter('L', default=20.0)``).
            A **number** is an inline shortcut: ``build()`` auto-
            creates a hidden positive parameter to back it.  See D2 in
            ``docs/spatial_design_decisions_v1.md``.

        Stored as ``model['boundary']`` and consumed by the propagator
        builder (Phase 2/3).  Only meaningful once at least one field
        is spatial; ``build()`` validates this.
        """
        mode = str(mode)
        if mode not in ('infinite', 'periodic'):
            raise ValueError(
                f"boundary(mode={mode!r}): v1 supports only 'infinite' "
                f"and 'periodic'.  Dirichlet/Neumann/Robin are v2 "
                f"(see docs/spatial_implementation_plan.md §scope).")
        if mode == 'periodic' and 'length' not in params:
            raise ValueError(
                "boundary('periodic', ...): a 'length' is required "
                "(name a parameter, e.g. length='L', or pass a number).")
        self._boundary = {'mode': mode, **params}
        return self

    def initial(self, mode: str = 'stationary', **params):
        """Declare the initial condition.

        Parameters
        ----------
        mode : {'stationary'}
            ``'stationary'`` — the system sits at its mean-field
            stationary state; no extra action term needed.  The only
            mode supported in v1 (transient ICs are v1.5 — see
            Lefèvre-Biroli §2.5's ``S_I`` term).

        Stored as ``model['initial']``.  ``compute_cumulants`` (Phase 4)
        validates that requested observables are compatible with the
        declared IC.
        """
        mode = str(mode)
        if mode != 'stationary':
            raise ValueError(
                f"initial(mode={mode!r}): v1 supports only 'stationary'.  "
                f"Transient ICs are v1.5 (see "
                f"docs/spatial_implementation_plan.md §scope).")
        self._initial = {'mode': mode, **params}
        return self

    def dyson(self, mode: str = 'fixed', order: int = None, tol=None):
        """Declare the Dyson–Duhamel truncation policy for coupled
        unequal-diffusion theories (``𝒟̂ ≠ 0``).

        Parameters
        ----------
        mode : {'off', 'fixed'}
            ``'off'`` — no Dyson dressing (the default when ``.dyson``
            is never called); only the scalar-``D₀`` reference
            propagator runs (exact iff ``𝒟̂ = 0``).
            ``'fixed'`` — truncate every retarded edge's Dyson series
            at a user-picked order ``N`` (the v1 policy; step D-4 in
            ``docs/dyson_duhamel_integration_plan.md``).  Requires
            ``order``.
        order : int ≥ 0, required for ``mode='fixed'``
            Truncation order ``N`` — each retarded edge sums Dyson
            insertions ``n = 0…N``.
        tol : float, optional
            Reserved for the v2+ tolerance policies; unused in v1.

        The same order ``N`` also bounds the Dyson loop-correction
        insertions on every retarded segment — and there is NO upper
        cap: the former loop-order ≤ 1, then ≤ 2 restrictions were
        removed (insertions are exact at every order via the
        ln-derivative partition expansion + generalized-partial-fraction
        𝓗_n labels; cost grows combinatorially with ``N``).

        The policy is a MODEL property — stored as
        ``model['spatial']['dyson']`` and read by the propagator
        builder / pipeline_bridge, never a ``compute_cumulants``
        kwarg (see the plan doc §"Keeping compute_cumulants simple").
        """
        mode = str(mode)
        if mode in ('auto', 'adaptive', 'resum'):
            raise NotImplementedError(
                f"dyson(mode={mode!r}): {mode!r} is a v2+ policy "
                f"('auto' = auto-tol, 'adaptive' = per-edge adaptive, "
                f"'resum' = resummed/exact — see the policy table in "
                f"docs/dyson_duhamel_integration_plan.md).  v1 supports "
                f"only 'off' and 'fixed'.")
        if mode not in ('off', 'fixed'):
            raise ValueError(
                f"dyson(mode={mode!r}): unrecognized policy; v1 "
                f"supports 'off' and 'fixed' (see "
                f"docs/dyson_duhamel_integration_plan.md).")
        if mode == 'fixed':
            # accept any integral type (e.g. a Sage Integer from the notebook
            # preparser) but reject bools, floats, and negatives
            import operator
            try:
                order_i = (None if isinstance(order, bool)
                           else operator.index(order))
            except TypeError:
                order_i = None
            if order_i is None or order_i < 0:
                raise ValueError(
                    f"dyson(mode='fixed', order={order!r}): a "
                    f"truncation order int >= 0 is required (each "
                    f"retarded edge sums Dyson insertions n = 0…order).")
            self._dyson = {'mode': mode, 'order': order_i}
        else:
            self._dyson = {'mode': mode}
        return self

    def dyson_order(self, order: int):
        """Sugar for ``.dyson(mode='fixed', order=order)`` — the v1
        fixed-truncation Dyson policy (step D-4 in
        ``docs/dyson_duhamel_integration_plan.md``)."""
        return self.dyson(mode='fixed', order=order)

    def reference_diffusion(self, D0):
        """Declare the scalar reference diffusion ``D₀`` for the
        ``𝒟 = D₀·I + 𝒟̂`` split underlying the Dyson–Duhamel series
        (see ``docs/dyson_duhamel_integration_plan.md``).

        Parameters
        ----------
        D0 : float > 0
            Reference diffusion constant.  When never declared, the
            propagator builder picks its own default (mean / min
            eigenvalue of ``𝒟``).  Convergence needs ``‖𝒟̂‖/D₀``
            small — a bad ``D₀`` choice diverges.

        Stored as ``model['spatial']['reference_diffusion']``.
        """
        D0 = float(D0)
        if not D0 > 0:
            raise ValueError(
                f"reference_diffusion(D0={D0!r}): D0 must be > 0 "
                f"(the scalar reference in the 𝒟 = D0·I + 𝒟̂ split).")
        self._reference_diffusion = D0
        return self


class _TemporalMethods:
    """Temporal-only authoring methods (mixed into TemporalTheoryBuilder
    and the back-compat TheoryBuilder)."""

    def population(self, name: str, *, size: int = 1,
                   description: str = ''):
        """Declare a population (a named index set with its own size).

        Heterogeneous-population theories chain one ``.population()``
        per group, then annotate each field / parameter / kernel with
        the population(s) it's indexed by.  Recorded on the builder
        but NOT yet propagated into the symbolic / diagrammatic
        pipeline — the pipeline currently treats all populations as
        a single combined index set of size ``sum(sizes)``.  Full
        per-population machinery is a separate refactor.

        Calling ``.population()`` overrides any previous
        ``n_populations=`` constructor argument.
        """
        size = max(int(size), 1)
        self.populations.append({
            'name':        name,
            'size':        size,
            'description': description,
        })
        # Keep n_populations in sync so legacy lookups still see a
        # sensible value (for now: just the count of declared pops).
        self.n_populations = len(self.populations)
        return self

    def kernel(self, name: str, frequency_image: Callable = None,
               sage_name: str = '', latex_name: str = '',
               description: str = '', indexed=False,
               indexed_by: Optional[list] = None):
        """Declare a convolution kernel symbol.

        Parameters
        ----------
        indexed_by : list of population names, optional
            Heterogeneous-population annotation.  ``[]`` or ``None``
            → scalar (one shared symbol, used as ``g``); ``['E']``
            → per-population kernel ``g[i]``; ``['E', 'I']`` →
            per-pair kernel ``g[i, j]``.  Overrides legacy
            ``indexed=`` when both are given.
        indexed : legacy
            ``False`` / ``True`` / ``'vector'`` / ``'matrix'``.
            Pre-heterogeneous-population flag.

        Indexed kernels let each population (or pair) have its own
        frequency image — declared via ``define_kernel(freq_image=...)``
        with an expression that may reference ``i`` (and ``j`` for
        matrix) and indexed parameters like ``tau_g[i, j]``.
        """
        if indexed_by is not None:
            ib = list(indexed_by)
            is_vec = (len(ib) == 1)
            is_mat = (len(ib) == 2)
        else:
            ib = None
            is_vec = (indexed in (True, 'vector'))
            is_mat = (indexed == 'matrix')
        self.kernels.append(KernelSpec(
            name=name, sage_name=sage_name or f'z_{name}',
            latex_name=latex_name or name,
            frequency_image=frequency_image, description=description,
            indexed=(is_vec or is_mat), matrix=is_mat,
            indexed_by=ib,
        ))
        return self

    def define_kernel(self, name: str, *, time_expr: str = None,
                      freq_image: str = None, latex_name: str = '',
                      sage_name: str = '', description: str = '',
                      indexed=False,
                      indexed_by: Optional[list] = None):
        """Declare a convolution kernel via either its time-domain
        expression OR its frequency image (or both).

        At least one of ``time_expr`` / ``freq_image`` must be given.
        ``freq_image`` is preferred because it's used directly by
        the propagator construction; if only ``time_expr`` is given,
        the build will warn that the FT must be supplied.

        Parameters
        ----------
        name : str
            Kernel symbol name (referenced in the action as ``<name>``
            for scalar kernels, ``<name>[i]`` for vector, or
            ``<name>[i, j]`` for matrix).
        time_expr, freq_image : str
            Sage-syntax expressions.  May reference ``i`` (and ``j``
            for matrix-indexed kernels) and any declared parameter.
            E.g. ``'1/(1 + I*omega*tau_g[i, j])'`` for a per-pair
            exponential synapse with matrix time constants.
        indexed : bool / 'vector' / 'matrix', default False
            * ``False`` — single shared kernel (default).
            * ``True`` / ``'vector'`` — per-population kernel ``g[i]``.
            * ``'matrix'`` — per-pair kernel ``g[i, j]``.

            Indexed kernels produce ``N`` (or ``N*N``) SR symbols
            internally (``g_<i+1>``, ``g_<i+1>_<j+1>``); the
            propagator builder substitutes each with its own
            frequency image evaluated at the corresponding ``i`` / ``j``.

        The kernel is also registered as a regular ``kernel(...)``
        symbol for the MSR-JD framework's name resolution.
        """
        if time_expr is None and freq_image is None:
            raise ValueError(
                f"define_kernel({name!r}): supply at least one of "
                f"time_expr= or freq_image=")
        # Register the kernel symbol with the regular .kernel() call.
        # ``indexed_by`` (when given) takes precedence over ``indexed=``.
        if not any(k.name == name for k in self.kernels):
            self.kernel(name, sage_name=sage_name or f'z_{name}',
                        latex_name=latex_name or name,
                        description=description, indexed=indexed,
                        indexed_by=indexed_by)
        spec = {
            'name':       name,
            'time_expr':  time_expr,
            'freq_image': freq_image,
            'indexed':    indexed,
        }
        if indexed_by is not None:
            spec['indexed_by'] = list(indexed_by)
        self._kernel_specs.append(spec)
        return self

    def markovianize(self, enabled: bool = True):
        """Toggle the colored-noise → Markovian-embedding preprocessor.

        Parameters
        ----------
        enabled : bool, default True
            When ``True`` (default), ``.build()`` walks ``_cgf_terms``
            and rewrites every row whose kernel matches
            ``c·exp(-|tau|/tauc)`` into a white-noise CGF row on an
            auxiliary OU field plus the corresponding linear filter
            in the action.  This unblocks ``max_ell >= 1`` colored-
            noise computations, which would otherwise hang in
            ``scipy.nquad``.

            When ``False``, no rewriting is performed and the legacy
            smooth-residual path runs.  Use this if you've hand-coded
            your colored kernel into the action and don't want the
            preprocessor to touch it, or if your kernel doesn't match
            the v1 single-Lorentzian template.

        Per-row overrides
        -----------------
        ``declare_cgf_term(..., markovianize=True | False)`` on an
        individual row takes precedence over this builder-level flag.

        See ``docs/correlated_noise_capabilities.md`` §1.5 for the
        complete reference (supported kernels, naming convention for
        auxiliary fields, v2 follow-ups).
        """
        self._markovianize_default = bool(enabled)
        return self

    def declare_cgf_term(self, name: str,
                         response_field: str | None = None,
                         order: int = 2,
                         coefficient: str = '',
                         kernel: str | None = None,
                         *,
                         response_legs: list[str] | str | None = None,
                         markovianize: bool | str | None = None):
        """Add one term to a non-closed-form cumulant generating
        functional (e.g. GTaS noise, cross-field colored noise).

        Parameters
        ----------
        name : str
            CGF identifier — multiple rows with the same name + order
            sum into a single cumulant.
        response_field : str, optional
            **Legacy / single-field path.**  The response field this
            cumulant's legs all sit on (e.g. ``'mt'``).  At order ``n``,
            this gets broadcast to all ``n`` legs.  Use ``response_legs``
            instead when different legs need different response fields.
        order : int
            Cumulant order: 2 for κ⁽²⁾, 3 for κ⁽³⁾, 4 for κ⁽⁴⁾, ...
        coefficient : str
            Sage-syntax expression for the cumulant coefficient
            (e.g. ``'lambda_X * p_part'``).
        kernel : str, optional
            Optional time-domain kernel multiplier.  At order 2 the
            kernel is a function of ``tau``; at order 3 of ``t1, t2``;
            etc.  Examples: ``'dirac_delta(tau)'`` (white),
            ``'exp(-abs(tau)/tauc)'`` (OU-colored),
            ``'dirac_delta(t1)*dirac_delta(t2)'`` (Poisson shot noise
            at κ³).  Defaults to ``None`` which the compiler treats as
            ``∏_k δ(τ_k)`` — fully-local cumulant.
        response_legs : list of str OR comma-separated str, keyword-only
            **New / multi-field path.**  Per-leg list of response field
            names, length = ``order``.  Use for cross-field cumulants:
            e.g. ``response_legs=['xt', 'yt']`` at order 2 gives
            ``κ²(xt, yt)``.  A single-element list (or string with no
            commas) is broadcast across all legs — equivalent to
            ``response_field=<name>``.  When both are supplied,
            ``response_legs`` wins.
        markovianize : bool or 'auto' or None, keyword-only
            Per-row override for the colored-noise → Markovian-
            embedding preprocessor.  Defaults to ``None`` (=
            ``'auto'``): the row is markovianized if its kernel
            matches the v1 single-Lorentzian template AND the
            builder-level ``.markovianize(...)`` toggle is on
            (default).  Set ``True`` to FAIL LOUDLY if the kernel
            doesn't match (use this when you've hand-tuned a
            Lorentzian and want to be sure the auto-detect doesn't
            silently reject it).  Set ``False`` to keep this row's
            colored kernel even when the builder-level toggle is on
            (e.g. you've prototyped a hand-rolled embedding in the
            action text).

        See ``docs/correlated_noise_capabilities.md`` for the
        complete reference on supported / unsupported noise models —
        in particular §1.5 (Markovian embedding), the n ≥ 3
        smooth-kernel limit, multiplicative-noise workaround, and
        non-stationary / Lévy gaps.
        """
        if response_legs is None and response_field is None:
            raise ValueError(
                "declare_cgf_term: supply either `response_field` "
                "(legacy single) or `response_legs` (per-leg list)."
            )
        # Normalize response_legs to a list of names; None → derive
        # from response_field at compile time (handled in
        # make_correlated_noises_block).
        if isinstance(response_legs, str):
            response_legs = [s.strip() for s in response_legs.split(',')
                             if s.strip()]
        # Normalize ``markovianize=`` to a bool / None for downstream
        # consumers.  Accept the string 'auto' as a synonym for None.
        if isinstance(markovianize, str):
            if markovianize.lower() == 'auto':
                markovianize_norm = None
            else:
                raise ValueError(
                    f"declare_cgf_term: markovianize={markovianize!r} "
                    f"unrecognised; allowed: True / False / 'auto' / None."
                )
        else:
            markovianize_norm = markovianize
        self._cgf_terms.append({
            'name':           name,
            'response_field': response_field,
            'response_legs':  response_legs,    # None when legacy single
            'order':          int(order),
            'coefficient':    coefficient,
            'kernel':         kernel,
            'markovianize':   markovianize_norm,
        })
        return self

    def correlated_noise(self, name: str, **kwargs):
        self._correlated_noises[name] = kwargs
        return self

    def use_synaptic_kernel(self, template):
        """Apply an ExpSynapticKernel (or DeltaKernel) template.

        For ExpSynapticKernel, sets the model's ``kernel_ft_image`` hook
        so the FT of g symbolically becomes 1 / (1 + iωτ_g).

        For DeltaKernel, no FT image is needed; instead the template
        contributes an EXTRA specialization ``g → delta_D`` that gets
        merged into the action template's specializations dict so the
        kernel acts as δ(t) at cell-8 propagator-construction time.
        """
        ft_hook = template.kernel_ft_image()
        if ft_hook is not None:
            self._kernel_ft_image = ft_hook

        # Merge any kernel-side specializations into the action's
        # specializations dict.  Capture the current specs lambda and
        # extend it.
        extra_specs = (template.extra_specializations()
                       if hasattr(template, 'extra_specializations')
                       else None)
        if extra_specs is not None and self._specializations is not None:
            base_specs = self._specializations
            self._specializations = (
                lambda ns, _base=base_specs, _extra=extra_specs: {
                    **_base(ns), **_extra(ns),
                }
            )
        return self

    def add_gtas_noise(self, template):
        """Apply a GTaSNoise template.  Adds the dm/mt fields, the
        mstar parameter, the GTaS feedforward terms in the action /
        saddle / MF equations (via the HawkesAction template's
        ``_gtas`` hook), and the correlated_noises block."""
        # Register the GTaS metadata so the HawkesAction template
        # picks up the feedforward + saddle terms when its lambdas
        # are emitted.  Do this BEFORE use_action_template() if the
        # user calls them in the wrong order, but supporting the more
        # convenient order (action then noise) too.
        self._pending_gtas = template

        # Add the dm physical field + mt response field automatically.
        # natural_name='m' lets users say external_fields=[('m', 1)]
        # and mf['m', 1] without remembering the 'dm'/'mstar' internals.
        #
        # ``auto_response=False``: physical_field() would otherwise
        # auto-generate a conjugate response field named ``mt`` (from
        # natural_name='m'), and we declare the response field
        # explicitly just below.  Leaving auto-response on declares
        # ``mt`` twice → a duplicate ``mt1`` generator in the bigrade
        # PolynomialRing → ``ValueError: variable name 'mt1' appears
        # more than once``.  The explicit declaration also honors a
        # custom ``template.response_field`` name, which the
        # auto-generated ``<natural>t`` would not.
        self.physical_field(
            template.physical_field, indexed=True,
            latex=r'\delta m',
            description='GTaS external rate fluctuation (zero-mean)',
            natural_name='m',
            auto_response=False,
        )
        self.response_field(
            template.response_field, indexed=True,
            latex=r'\tilde m',
            description='GTaS response field conjugate to dm',
        )

        # Add the GTaS parameters automatically (with sane defaults
        # the user can override via .parameter() before .build()).
        for pname, default, dom in [
            (template.feedforward_weight, 0.25,  None),
            (template.mother_rate,        1.7,   'positive'),
            (template.participation,      0.6,   'positive'),
        ]:
            if not any(p.name == pname for p in self.parameters):
                self.parameter(pname, default=default, domain=dom)
        if template.mu_diff is not None and not any(
                p.name == template.mu_diff for p in self.parameters):
            self.parameter(template.mu_diff, default=0.0)
        if template.sigma_diff_sq is not None and not any(
                p.name == template.sigma_diff_sq for p in self.parameters):
            self.parameter(template.sigma_diff_sq, default=1.0,
                           domain='positive')
        if template.background_param is not None and not any(
                p.name == template.background_param for p in self.parameters):
            self.parameter(template.background_param, indexed=True,
                           domain='positive',
                           mean_field=True, natural_name='m',
                           description=f'background external rate b_X = '
                                       f'{template.mother_rate} · '
                                       f'{template.participation}')

        self._correlated_noises.update(template.correlated_noises_block())

        # Re-emit the action with the GTaS hook now wired in.  The
        # HawkesAction template stores ``_gtas`` so the emitted
        # lambdas know to add feedforward terms.
        if hasattr(self, '_action_template'):
            self._action_template._gtas = template
            # Rebuild the lambdas
            self.use_action_template(self._action_template)
        return self

    def use_action_template(self, template):
        """Override the previous use_action_template to remember the
        template object so add_gtas_noise() can re-emit with the GTaS
        hook installed."""
        self._action_template = template
        # Apply any pending GTaS hook
        if getattr(self, '_pending_gtas', None) is not None:
            template._gtas = self._pending_gtas
        self._action          = template.action()
        self._phi_concrete    = template.phi_concrete()
        self._specializations = template.specializations()
        self._mf_bg           = template.mf_bg_conditions()
        self._mf_equations    = template.mf_equations()
        self._mf_substitutions = template.mf_substitutions()
        self._functions       = list(template.functions_list())
        return self


class _BaseTheoryBuilder:
    """
    Accumulator for a model declaration.  Call ``.build()`` to emit the
    MSR-JD model dict (same structure as the example ``HAWKES_MODEL``)
    suitable for ``api.compute_cumulants``.

    Shared base for ``TemporalTheoryBuilder`` and ``SpatialTheoryBuilder``.
    The usual flow is string-based: declare fields and parameters, then
    supply Sage-syntax text via ``set_action_text(...)``,
    ``set_mf_equation(...)`` (or ``equation(...)``), ``define_kernel(...)``,
    and ``define_function(...)``.  ``build()`` compiles those strings to
    the action / mf_bg_conditions / kernel_ft_image / phi_concrete /
    mf_equations lambdas the model dict expects.  Reusable actions are
    applied with ``use_action_template(...)``.  The raw lambda hooks
    (``set_action(...)`` etc.) remain available as an optional escape
    hatch for theories no string or template covers.
    """

    def __init__(self, name: str, n_populations: int = 1):
        self.name = name
        self.n_populations = n_populations
        # ``populations`` is the new explicit heterogeneous-population
        # declaration: list of {'name': str, 'size': int}.  When
        # ``.population()`` is called, ``n_populations`` is replaced by
        # the count of declared populations.  When this list stays
        # empty, the builder behaves like the legacy single-anonymous-
        # population path (one "pop" index of size n_populations).
        self.populations:     list[dict] = []
        self.response_fields: list[FieldSpec] = []
        self.physical_fields: list[FieldSpec] = []
        self.parameters:      list[ParameterSpec] = []
        self.kernels:         list[KernelSpec] = []
        self._action: Optional[Callable] = None
        self._mf_bg: Optional[Callable] = None
        self._mf_bg_solver: Optional[Callable] = None
        self._mf_equations: Optional[Callable] = None
        self._kernel_ft_image: Optional[Callable] = None
        # Time-domain kernel image — companion of ``_kernel_ft_image``,
        # used by the time-domain Phase J integrator to substitute
        # surviving kernel SR symbols at conductance-style vertices.
        self._kernel_td_image: Optional[Callable] = None
        self._phi_concrete: Optional[Callable] = None
        self._specializations: Optional[Callable] = None
        self._mf_substitutions: list[dict] = []
        self._functions: list[dict] = []
        self._operators: list[dict] = []
        self._correlated_noises: dict = {}

        # Text-driven declarations (used by the UI and by
        # ``define_function`` / ``set_action_text`` / etc.).  When any
        # of these are populated, ``build()`` compiles them to lambdas
        # via ``api.theory_compiler``, overriding the lambda hooks
        # above.  Empty by default → existing template-based flow runs
        # unchanged.
        self._function_specs:   list[dict] = []
        self._kernel_specs:     list[dict] = []   # text time/freq exprs
        self._action_text:      Optional[str] = None
        self._operator_ir:      bool = False     # spatial v2: Lap()/Dt()/Dx() syntax
        self._mf_eqs_text:      dict[str, str] = {}
        self._cgf_terms:        list[dict] = []
        self._phi_function_name: Optional[str] = None
        # Explicit DAE specification (new MF-solver path).  Each entry:
        #   {'lhs_text': str, 'rhs_text': str,
        #    'population': str | None, 'kind': 'differential' | 'algebraic'}
        # ``kind`` is inferred from whether ``Dt`` appears as a word in
        # ``lhs_text``.  Populated via ``.equation(...)``.  Consumed by
        # the DAE-based ``solve_mean_field`` (multi-start Newton + sort
        # + linear-stability).  If empty, the legacy iteration solver
        # runs from ``_mf_eqs_text`` instead.
        self._equations:        list[dict] = []
        # Whether the DAE solver should classify every converged root
        # for linear stability and restrict ``fixed_point_index`` to
        # the stable subset.  Default OFF — theories that integrate
        # out their voltages (all-algebraic equations, no ``Dt``)
        # have no differential structure to score, so eigenvalue
        # analysis is vacuous.  Bistable / differential theories that
        # want stability-based root selection must opt in explicitly
        # via ``.stability_analysis(True)``.
        self._stability_analysis: bool = False
        # Whether ``build()`` invokes the colored-noise → Markovian-
        # embedding preprocessor (``api.colored_to_markovian``)
        # on this builder before compiling text declarations.  Default
        # ON: every CGF row that matches the v1 single-Lorentzian
        # template ``c·exp(-|tau|/tauc)`` is rewritten as a white-
        # noise-driven OU auxiliary field.  Set ``False`` via
        # ``.markovianize(False)`` for theories whose colored kernels
        # don't match the template (the existing scipy.nquad fallback
        # then runs unchanged with its warning).  Per-row override is
        # available via the ``markovianize=`` keyword on
        # ``declare_cgf_term``.
        self._markovianize_default: bool = True

        # ── Spatial-extension state (v1) ──────────────────────────
        # Default spatial dimension applied to physical fields
        # declared AFTER a ``.spatial_dim(d)`` call.  0 keeps the
        # framework in its time-only behaviour.  Per-field explicit
        # ``physical_field(spatial_dim=...)`` overrides this default.
        self._default_spatial_dim: int = 0
        # Boundary-condition declaration (``.boundary(mode, **params)``).
        # None until declared; emitted as ``model['boundary']``.  v1
        # supports {'infinite', 'periodic'}.
        self._boundary: Optional[dict] = None
        # Initial-condition declaration (``.initial(mode, **params)``).
        # None until declared; emitted as ``model['initial']``.  v1
        # supports {'stationary'}.
        self._initial: Optional[dict] = None
        # Dyson–Duhamel truncation policy (``.dyson(...)`` /
        # ``.dyson_order(N)``).  None until declared; emitted as
        # ``model['spatial']['dyson']`` (default ``{'mode': 'off'}``).
        # Lives on the base — like ``_boundary`` — so ``build()`` can
        # read it on ANY builder and reject it on non-spatial
        # theories.  See docs/dyson_duhamel_integration_plan.md (D-4).
        self._dyson: Optional[dict] = None
        # Scalar reference diffusion D₀ (``.reference_diffusion(D0)``).
        # None until declared; emitted as
        # ``model['spatial']['reference_diffusion']`` when set.
        self._reference_diffusion: Optional[float] = None

    # ── Population declarations ───────────────────────────────────

    # ── Field declarations ─────────────────────────────────────────
    def response_field(self, name: str, indexed: bool = True,
                       latex: str = '', description: str = '',
                       population: str = None):
        self.response_fields.append(FieldSpec(
            name=name, indexed=indexed,
            latex=latex or name, description=description,
            population=population,
        ))
        return self

    def physical_field(self, name: str, indexed: bool = True,
                       latex: str = '', description: str = '',
                       natural_name: str = None,
                       auto_response: bool = True,
                       auto_saddle: bool = True,
                       population: str = None,
                       spatial_dim: Optional[int] = None):
        """Declare a physical field.

        Spatial fields
        --------------
        ``spatial_dim`` (int) declares a continuous spatial field
        ``φ(x, t)`` in ``spatial_dim`` dimensions.  ``0`` (or the
        builder default, set via ``.spatial_dim(d)``) keeps the field
        time-only.  ``spatial_dim ∈ {1, 2, 3}`` are validated end-to-end
        (``d ≥ 2`` uses the radial inverse-FT, infinite-boundary only).
        Every spatial field in a theory must share the same non-zero
        dimension (mixed-dim is a v2 feature).  When any field is
        spatial, ``build()`` registers a ``Laplacian`` operator symbol
        (usable multiplicatively in the action text, exactly like
        ``Dt``) and emits a ``model['spatial']`` block.  For derivative
        vertices (KPZ/Burgers/Model B) use ``Dt()/Lap()/Dx()`` call
        syntax in the action and turn on :meth:`operator_ir`.  See
        ``docs/spatial_pipeline.md``.

        If ``spatial_dim`` is left as ``None``, the field inherits the
        builder-level default (``self._default_spatial_dim``, normally
        0; set non-zero by a prior ``.spatial_dim(d)`` call).

        Two calling styles, distinguished by whether ``natural_name``
        is supplied:

        **New style** (recommended; what the UI emits)::

            .physical_field('n')                # name IS the natural letter
            .physical_field('v', latex='v')

        Framework derives the **internal** fluctuation name as ``d<name>``
        (so user types ``n`` here, action uses ``n[i]`` or ``dn[i]``,
        both refer to the fluctuation field).  Auto-generates the
        conjugate response field ``<name>t`` and the saddle parameter
        ``<name>star`` (with ``mean_field=True``).

        **Legacy style** (existing hand-written theory files)::

            .physical_field('dn', natural_name='n', latex=r'\\delta\\dot n')

        ``name`` is the literal internal name; ``natural_name`` is the
        separate user-facing letter.  Auto-response and auto-saddle
        also fire (with ``natural_name`` as the base) unless
        ``auto_response=False`` / ``auto_saddle=False`` is passed.
        """
        if natural_name is None:
            # New style — name is the natural letter.
            natural_name  = name
            internal_name = f'd{name}'
        else:
            # Legacy — name is the internal fluctuation name.
            internal_name = name

        # Per-field spatial dimension: explicit kwarg wins; otherwise
        # inherit the builder-level default set by ``.spatial_dim(d)``.
        sdim = (self._default_spatial_dim if spatial_dim is None
                else int(spatial_dim))

        self.physical_fields.append(FieldSpec(
            name=internal_name, indexed=indexed,
            latex=latex or rf'\delta {natural_name}',
            description=description,
            natural_name=natural_name,
            population=population,
            spatial_dim=sdim,
        ))

        # Auto-generate the conjugate response field as ``<natural>t``
        # (matching the existing nt/vt/mt convention).  Skipped if the
        # user already declared a response field with that name.
        # Heterogeneous-pop: the response field inherits the physical
        # field's population so its SR-var array has the right size.
        if auto_response:
            response_name = f'{natural_name}t'
            if not any(f.name == response_name for f in self.response_fields):
                self.response_field(
                    response_name, indexed=indexed,
                    latex=rf'\tilde {natural_name}',
                    description=f'response field conjugate to {natural_name}',
                    population=population,
                )

        # Auto-generate the saddle parameter ``<natural>star``.  Default
        # domain is ``positive`` for ``n`` (rates) and free for others.
        # Heterogeneous-pop: the saddle is indexed by the field's
        # population so the SR-var array has the right size.
        if auto_saddle:
            saddle_name = f'{natural_name}star'
            if not any(p.name == saddle_name for p in self.parameters):
                domain = 'positive' if natural_name == 'n' else None
                kwargs = dict(
                    indexed=True, domain=domain,
                    mean_field=True, natural_name=natural_name,
                    description=f'mean-field saddle value of {natural_name}',
                )
                if population:
                    kwargs['indexed_by'] = [population]
                self.parameter(saddle_name, **kwargs)
        return self

    # ── Parameter declarations ─────────────────────────────────────
    def parameter(self, name: str, default: Any = None,
                  indexed=False, domain: str = None, description: str = '',
                  mean_field: bool = False, natural_name: str = None,
                  indexed_by: Optional[list] = None):
        """Declare a model parameter.

        Parameters
        ----------
        name : str
            Internal parameter name.
        default : Any, optional
            Default numerical value (scalar / list / matrix).
        indexed_by : list of population names, optional
            Heterogeneous-population annotation.  ``[]`` or ``None``
            → scalar; ``['E']`` → vector of size ``size(E)``;
            ``['E', 'I']`` → matrix of shape
            ``(size(E), size(I))`` (row-first).  Overrides the
            legacy ``indexed=`` keyword when both are present.
        indexed : bool / 'vector' / 'matrix' (legacy)
            Pre-heterogeneous-population indexing flag.  ``True`` /
            ``'vector'`` → vector of length ``n_populations``;
            ``'matrix'`` → N×N matrix.  Ignored when ``indexed_by``
            is given.
        domain : str, optional
            ``'positive'`` etc.  Used by the FieldTheory builder.
        mean_field : bool, default False
            Flags the parameter as a saddle-point quantity.
        natural_name : str, optional
            User-facing letter for MF accessor lookup.
        """
        # Heterogeneous-population path wins when indexed_by is given.
        if indexed_by is not None:
            ib = list(indexed_by)
            is_vec = (len(ib) == 1)
            is_mat = (len(ib) == 2)
        else:
            ib = None
            is_vec = (indexed in (True, 'vector'))
            is_mat = (indexed == 'matrix')
        self.parameters.append(ParameterSpec(
            name=name, indexed=(is_vec or is_mat), matrix=is_mat,
            domain=domain, default=default, description=description,
            mean_field=mean_field, natural_name=natural_name,
            indexed_by=ib,
        ))
        return self

    # ── Kernel declaration ─────────────────────────────────────────

    # ── Text-driven theory declaration (UI-friendly) ──────────────
    # Each method below stores a Sage-syntax text string.  At
    # ``.build()`` time the strings are compiled to Python lambdas via
    # ``api.theory_compiler`` and dropped into the model dict.
    # Calling any of these overrides the corresponding lambda hook.

    def define_function(self, name: str, args: list[str],
                        expression: str, latex: str = None,
                        description: str = '',
                        population: Optional[str] = None):
        """Declare a function of field variables (and parameters).

        Parameters
        ----------
        name : str
            Function name as referenced inside the action / MF
            equations (e.g. ``'phi'``).
        args : list of str
            Field-variable names that are formal arguments
            (e.g. ``['v']``).  The function body may also reference
            any declared parameter by name.
        expression : str
            Sage-syntax body, e.g. ``'a*v^2'`` or
            ``'1 / (1 + exp(-v/v_thresh))'``.
        population : str, optional
            Heterogeneous-population annotation: the population whose
            index range this function iterates over when called as
            ``f[i](v[i])`` in the action.  Recorded on the function
            spec but not yet propagated to the symbolic pipeline.
        latex : str, optional
            Display form (defaults to ``name``).
        """
        entry = {
            'name':        name,
            'args':        list(args),
            'expression':  expression,
            'latex':       latex or name,
            'description': description,
        }
        if population:
            entry['population'] = population
        self._function_specs.append(entry)
        return self


    def set_action_text(self, text: str):
        """Set the per-population action integrand ``S_i`` as a
        Sage-syntax string.

        ``i`` is the implicit free index — at build time the integrand
        is summed over ``i in range(n_populations)``.  Inner sums use
        Python comprehension syntax::

            nt[i] * (Dt + 1/tau) * dn[i]
            - nt[i] * phi(vstar[i] + dv[i])
            + vt[i] * (Dt + 1/tau) * dv[i]
            - vt[i] * sum(w[i, j] * g * dn[j] for j in pop)

        ``pop = range(n_populations)`` is pre-bound for inner sums.

        Spatial actions
        ---------------
        For a spatial field ``φ(x, t)`` the action is a SCALAR (no
        per-population index).  Two authoring styles:

        - **Plain (v1)** — the ``Laplacian`` operator symbol is used
          multiplicatively, exactly like ``Dt``::

              pt*(Dt(p) + mu*p - DD*Lap(p) + g*p^2) - T*pt^2

          (Reaction–diffusion with a ``g·p²`` interaction vertex.)
        - **Derivative vertices (KPZ/Burgers/Model B)** — author the
          differential operators as calls ``Dt(φ)``, ``Lap(φ)``,
          ``Dx(φ, i)`` and turn on :meth:`operator_ir`; e.g. KPZ::

              ht*(Dt(h) + mu*h - D*Lap(h) - (c/2)*Dx(h,0)^2) - T*ht^2

        **Non-Gaussian white noise** is a response-field monomial of
        degree ≥ 3 — it becomes a higher noise-cumulant source vertex.
        For example a third noise cumulant ``κ⁽³⁾ = 3!·S3``::

            pt*(Dt(p) + mu*p - DD*Lap(p)) - T*pt^2 - S3*pt^3

        **Coupled fields** carry one term per field; cross-coupling
        appears as off-diagonal terms (e.g. ``+ 0.4*b`` inside the
        ``a`` block).
        """
        self._action_text = text
        return self

    def operator_ir(self, on: bool = True):
        """Opt in to the spatial-v2 **operator IR**: author spatial/temporal
        differential operators as argument-binding calls — ``Lap(phi)``,
        ``Dt(phi)``, ``Dx(phi, i)`` — instead of the v1 bare multiplicative
        symbols (``Dt*phi``, ``D*Laplacian*phi``).

        When on, the action is parsed into the operator IR
        (``api.spatial_operator_ir``) and lowered to derived ring
        generators (the ``u=δφ, v=∇²δφ`` representation) before expansion, so
        derivative-inside-nonlinearity vertices and per-leg momentum form
        factors are represented exactly.  Default OFF — every existing theory is
        unaffected.  See ``docs/spatial_v2_architecture.md``.
        """
        self._operator_ir = bool(on)
        return self

    def set_mf_equation(self, saddle_name: str, rhs_text: str):
        """Declare a per-population mean-field equation.

        ``saddle_name`` is the parameter name (e.g. ``'vstar'``,
        ``'nstar'``, ``'mstar'``) that the equation defines.  The RHS
        is a Sage-syntax string in terms of ``i``, parameters, and
        other saddles::

            set_mf_equation('vstar',
                'E[i] + sum(w[i, j] * g * nstar[j] for j in pop)')
            set_mf_equation('nstar', 'phi(vstar[i])')

        The framework also auto-emits ``phi0_<i> = nstar[i]`` for the
        EOM closure, so users don't need to declare that.
        """
        self._mf_eqs_text[saddle_name] = rhs_text
        return self

    def equation(self, *, lhs: str, rhs: str,
                 population: Optional[str] = None):
        """Declare one residual of the DAE system used by the multi-root
        MF solver and linear-stability check.

        Each call adds ONE equation of the form ``LHS = RHS``, i.e.
        residual ``LHS - RHS = 0``.  At MF evaluation the framework
        substitutes ``Dt → 0`` on the LHS and solves the resulting
        algebraic system via multi-start Newton (``solve_mean_field``).
        At linearization (for stability) the framework keeps ``Dt`` as
        ``-iω`` and assembles the generalized-eigenvalue problem
        ``(B - iω A) δx = 0`` from per-equation partials.

        Parameters
        ----------
        lhs : str
            Sage-syntax expression.  Allowed to contain ``Dt`` (the
            time-derivative operator).  If ``Dt`` appears as a word
            here, the equation is classified ``'differential'``;
            otherwise ``'algebraic'``.  Example:
            ``'(tau[i]*Dt + 1) * v[i]'``.
        rhs : str
            Sage-syntax expression in terms of ``i`` (and ``j`` for
            inner sums), declared parameters, declared functions
            (``phi[i](...)``), and state-variable fields.  Must NOT
            contain ``Dt`` (time derivatives belong on the LHS) or
            ``Conv(...)`` operators (we assume stationary MF, so
            kernel convolutions of constants have been pre-collapsed
            by the user; for a normalized kernel that's just the
            constant itself).  Example:
            ``'Em[i] + sum(w[i,j]*n[j] for j in E)'``.
        population : str, optional
            Population name (matching a ``.population(name=...)``
            declaration).  The equation is expanded over the
            population's index set at solve time — ``[i]`` ranges
            over ``range(pop_size)``.  Pass ``None`` for a scalar
            equation (no ``[i]`` indexing on either side).

        Returns
        -------
        self
            For chaining.

        Notes
        -----
        Equations don't need names — they're stored in declaration
        order and the solver treats them as an unordered system.
        For diagnostics the framework auto-labels each equation by
        its (truncated) LHS text.

        Sort order for multi-root solutions: the framework sorts
        fixed points by the FIRST declared physical field's first
        population index, ascending.  Use ``fixed_point_index=N`` in
        ``compute_cumulants`` to pick the N-th root (default 0 =
        lowest).
        """
        import re

        if not isinstance(lhs, str) or not lhs.strip():
            raise ValueError(
                'TheoryBuilder.equation(): lhs must be a non-empty string.'
            )
        if not isinstance(rhs, str) or not rhs.strip():
            raise ValueError(
                'TheoryBuilder.equation(): rhs must be a non-empty string.'
            )

        # Reject Conv(...) — stationary assumption requires the user
        # to have pre-collapsed kernel convolutions.  Catches both
        # ``Conv(g, n)`` and case variants like ``conv(...)``.
        for side_name, side_text in (('lhs', lhs), ('rhs', rhs)):
            if re.search(r'\bConv\s*\(', side_text, re.IGNORECASE):
                raise ValueError(
                    f"TheoryBuilder.equation(): Conv(...) found in "
                    f"{side_name}; the DAE assumes stationary MF, so "
                    f"all kernel convolutions of constants must be "
                    f"pre-collapsed (for normalized kernels: replace "
                    f"``Conv(g, x)`` with ``x``)."
                )

        # Reject Dt in rhs — derivatives belong on the LHS.
        if re.search(r'\bDt\b', rhs):
            raise ValueError(
                "TheoryBuilder.equation(): Dt found in rhs; time "
                "derivatives must appear on the LHS only."
            )

        # Classify by Dt presence in lhs.
        kind = ('differential' if re.search(r'\bDt\b', lhs)
                else 'algebraic')

        self._equations.append({
            'lhs_text':   lhs,
            'rhs_text':   rhs,
            'population': population,
            'kind':       kind,
        })
        return self


    def stability_analysis(self, enabled: bool):
        """Toggle linear-stability classification for the DAE solver.

        Parameters
        ----------
        enabled : bool
            ``True``: ``solve_mean_field_dae`` runs ``linear_stability``
            on every converged root, filters to the stable subset, and
            ``fixed_point_index`` ranges over those.  Required for
            bistable / multi-saddle theories where the diagrammatic
            expansion is only well-defined at the linearly-stable
            roots.

            ``False`` (default): no stability analysis runs.
            ``fixed_point_index`` ranges over every converged root,
            sorted ascending by the first declared physical field's
            first index.  Use this for theories whose equations are
            ALL algebraic — e.g. voltages have been integrated out —
            where the generalized-eigenvalue ``(σA + B)`` has
            ``A ≡ 0`` and "linear stability" has no meaning.

        The setting is stored on the built model as
        ``model['stability_analysis']`` and consumed by
        ``api._mean_field_dae.solve_mean_field_dae``.
        """
        self._stability_analysis = bool(enabled)
        return self

    # ── Spatial-extension declarations (v1) ────────────────────────




    def set_transfer_function(self, name: str = 'phi'):
        """Mark which declared function plays the role of the MSR-JD
        ``phi_concrete`` (the saddle-point Taylor-expansion target).

        Defaults to ``'phi'``; only needed if you've named the transfer
        function differently.
        """
        self._phi_function_name = name
        return self

    # ── Lambda hooks (optional escape hatch for the harder stuff) ───
    def set_action(self, fn: Callable):
        """``fn(ns) -> SR``  — the model action.  See
        simulations/hawkes_quad_expg_gtas.py for examples."""
        self._action = fn
        return self

    def set_mf_bg_conditions(self, fn: Callable):
        self._mf_bg = fn
        return self

    def set_mf_equations(self, fn: Callable):
        self._mf_equations = fn
        return self

    def set_kernel_ft_image(self, fn: Callable):
        self._kernel_ft_image = fn
        return self

    def set_phi_concrete(self, fn: Callable):
        self._phi_concrete = fn
        return self

    def set_specializations(self, fn: Callable):
        self._specializations = fn
        return self

    def set_mf_substitutions(self, subs: list[dict]):
        self._mf_substitutions = subs
        return self

    def add_function(self, spec: dict):
        self._functions.append(spec)
        return self

    def add_operator(self, spec: dict):
        self._operators.append(spec)
        return self


    # ── High-level template wiring ─────────────────────────────────




    # ── Text → lambda compilation ─────────────────────────────────
    def _auto_populate_mf_eqs_from_equations(self) -> None:
        """When ``.equation(lhs=..., rhs=...)`` calls have been made but
        no legacy ``set_mf_equation(...)`` has been registered, synthesize
        the latter from the former so the FieldTheory sanity check and
        action-substitution chain keep working unchanged.

        Assumption: each equation's LHS at ``Dt = 0`` reduces to a single
        state-variable reference like ``v[i]`` (possibly with a unit
        coefficient).  The saddle name is inferred as
        ``<state_var>star``; state-variable references in the RHS are
        textually rewritten to saddle names (``n[j] → nstar[j]``, etc.).

        Theories with more elaborate LHSs (e.g. ``v[i] - n[i] = …``)
        should either simplify the LHS or fall back to direct
        ``set_mf_equation(...)`` declarations.
        """
        import re

        if not self._equations or self._mf_eqs_text:
            return

        # Field names the user might have written in equations.  Use
        # ``natural_name`` (the user-facing letter) when set, otherwise
        # fall back to the internal name.
        state_var_names = [
            (f.natural_name or f.name) for f in self.physical_fields
        ]
        if not state_var_names:
            return

        for eq in self._equations:
            lhs_text   = eq['lhs_text']
            rhs_text   = eq['rhs_text']

            # Substitute ``Dt → 0`` in the LHS text.  The result should
            # textually contain exactly one state-variable reference.
            lhs_at_dt0 = re.sub(r'\bDt\b', '0', lhs_text)

            # Try indexed form first (``x[i]``), then fall back to
            # scalar form (bare ``x``) for theories without a
            # declared population.  Both forms are accepted.
            primary = None
            scalar_form = False
            for v in state_var_names:
                if re.search(rf'\b{re.escape(v)}\[\s*i\s*\]', lhs_at_dt0):
                    primary = v
                    scalar_form = False
                    break
            if primary is None:
                # No ``[i]`` in LHS — look for bare field name.
                for v in state_var_names:
                    if re.search(rf'\b{re.escape(v)}\b', lhs_at_dt0):
                        primary = v
                        scalar_form = True
                        break
            if primary is None:
                raise ValueError(
                    f'TheoryBuilder.build(): cannot auto-derive an MF '
                    f'saddle name for equation with lhs={lhs_text!r}. '
                    f'At Dt=0 the LHS must reduce to either a bare '
                    f'state-variable reference (``x`` for scalar '
                    f'theories) or an indexed one (``v[i]``).  Either '
                    f'simplify the LHS, or skip ``.equation(...)`` and '
                    f'use ``set_mf_equation(...)`` directly.'
                )

            # Rewrite the RHS, replacing each state-variable name with
            # its saddle counterpart.  Word-boundary regex avoids hits
            # on parameter/kernel names that contain the field name as
            # a substring (e.g. ``Em`` vs ``E``).
            rewritten_rhs = rhs_text
            for v in state_var_names:
                rewritten_rhs = re.sub(
                    rf'\b{re.escape(v)}\b',
                    f'{v}star',
                    rewritten_rhs,
                )
            # For scalar (no-``[i]``) LHS, the saddle RHS should also
            # be in scalar form — rewrite ``xstar`` → ``xstar[0]``
            # because the legacy MF compiler expects per-population
            # indexed access.  (The DAE solver auto-handles both.)
            if scalar_form:
                rewritten_rhs = re.sub(
                    r'\b(\w+)star\b(?!\s*\[)',
                    r'\1star[i]',
                    rewritten_rhs,
                )

            saddle_name = f'{primary}star'
            self._mf_eqs_text[saddle_name] = rewritten_rhs

    def _compile_text_declarations(self) -> None:
        """Walk the text-based declarations and compile each into the
        corresponding lambda hook.  No-op if no text declarations
        were made (template-only builders go through unchanged)."""
        # Step 0: if the user used the new ``.equation(...)`` API but
        # didn't separately call ``set_mf_equation(...)``, derive the
        # legacy mf-eq text dict from the equations so the rest of the
        # compilation chain (sanity check, action substitution) works
        # without change.
        self._auto_populate_mf_eqs_from_equations()

        if not (self._action_text or self._function_specs
                or self._kernel_specs or self._mf_eqs_text
                or self._cgf_terms):
            return

        from api.theory_compiler import (
            make_action_lambda,
            make_phi_concrete_lambda,
            make_specializations_lambda,
            make_mf_bg_conditions_lambda,
            make_mf_bg_solver_lambda,
            make_mf_equations_lambda,
            make_kernel_ft_image_lambda,
            make_kernel_td_image_lambda,
            make_correlated_noises_block,
        )

        n_pop = self.n_populations
        field_names  = ([f.name for f in self.response_fields]
                        + [f.name for f in self.physical_fields])
        param_names  = [p.name for p in self.parameters]
        kernel_names = [k.name for k in self.kernels]

        # Pre-compute the naming convention so the compiler can expose
        # natural-name aliases (n, v, m) in the action's eval namespace
        # AND find each saddle quantity for full-field expansion
        # (n[i] → nstar[i] + dn[i]).
        _fluct_map: dict[str, str] = {}
        for f in self.physical_fields:
            if f.natural_name:
                _fluct_map[f.natural_name] = f.name
        _saddle_map: dict[str, str] = {}
        _mf_param_names: list[str] = []
        for p in self.parameters:
            if p.mean_field:
                _mf_param_names.append(p.name)
                if p.natural_name:
                    _saddle_map[p.natural_name] = p.name
        _action_naming_convention = {
            'fluctuation_fields': _fluct_map,
            'mean_field_saddles': _saddle_map,
            'mf_parameters':      _mf_param_names,
        }

        # Pick a "primary" function for legacy phi_concrete plumbing
        # (one of the user's functions becomes phi_concrete for the
        # saddle solver).  Default: the first declared function, or
        # one explicitly named via ``set_transfer_function``.  This
        # is the ONLY place where one function is selected — the
        # action-side Taylor expansion treats all functions
        # uniformly.
        #
        # ``phi_concrete`` requires a SINGLE-argument function (the
        # saddle solver inverts ``nstar = phi(vstar)`` along one
        # variable), so the auto-pick prefers the first single-arg
        # function rather than blindly grabbing the first declared.
        # Multi-arg functions still participate in the action-side
        # Taylor expansion via FieldTheory's rename machinery.
        if self._phi_function_name:
            primary_fn_name = self._phi_function_name
        else:
            primary_fn_name = next(
                (f['name'] for f in self._function_specs
                 if len(f.get('args') or []) == 1),
                (self._function_specs[0]['name']
                 if self._function_specs else None))
        primary_fn_spec = next(
            (f for f in self._function_specs
             if f['name'] == primary_fn_name), None)
        # If the chosen primary is multi-arg (because no single-arg
        # function exists), skip phi_concrete plumbing — saddle solving
        # then falls back to using the user's mf_eq RHS directly.
        if primary_fn_spec is not None and \
                len(primary_fn_spec.get('args') or []) != 1:
            primary_fn_spec = None

        # Action — every declared function becomes an indexed formal
        # Sage symbol that FieldTheory's auto-Taylor pass expands.
        if self._action_text is not None:
            self._action = make_action_lambda(
                self._action_text,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
                transfer_function = primary_fn_spec,
                naming_convention = _action_naming_convention,
            )

        # Register EVERY declared function as a formal indexed entry in
        # model['functions'].  FieldTheory's auto-Taylor pass walks this
        # list and, for every multi-index (k_1, ..., k_n) of partial
        # derivative orders with sum ≤ taylor_order, produces a clean
        # rename target ``<name><k_1><k_2>...<k_n>_<i+1>``.  Single-arg
        # functions (n=1) collapse to the legacy ``<name><k>_<i+1>``
        # naming, so existing models are unaffected.  Multi-arg
        # functions (n≥2) are supported natively via Sage's true
        # multivariate ``taylor()`` — no chained single-variable Taylor.
        from sage.all import function as sr_function
        self._functions = []
        for fn_spec in self._function_specs:
            fname  = fn_spec['name']
            n_args = len(fn_spec.get('args') or [])
            entry = {
                'name':         fname,
                'indexed':      True,
                'deriv_prefix': fname,
                'n_args':       n_args,
                'latex':        fn_spec.get('latex', fname),
                'description':  fn_spec.get('description',
                                            f'declared function {fname}'),
                'expression':   (lambda i, *xs, _t=fname:
                                 sr_function(f'{_t}_{i+1}')(*xs)),
                # Preserve the user's original text spec so downstream
                # consumers (e.g. the DAE-MF solver) can evaluate the
                # function numerically without going through Sage SR.
                'expression_text': fn_spec.get('expression'),
                'args_text':       list(fn_spec.get('args') or []),
            }
            # Heterogeneous-pop annotation passes through unchanged
            # so downstream pipeline can route ``phi[i](...)`` to the
            # right population's index range.
            if fn_spec.get('population'):
                entry['population'] = fn_spec['population']
            self._functions.append(entry)

        # phi_concrete + specializations: needed for the saddle solver
        # and for the Taylor-rename derivative substitutions
        # (specializations registers <fn><k>_<i+1> → kth derivative at
        # saddle for every function).
        if primary_fn_spec is not None:
            self._phi_concrete = make_phi_concrete_lambda(
                primary_fn_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
            )
            self._specializations = make_specializations_lambda(
                primary_fn_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
                naming_convention = _action_naming_convention,
                mf_eqs       = self._mf_eqs_text,
            )

        # Mean-field equations.  We produce TWO substitution lambdas:
        #
        # - ``self._mf_bg`` (action-side): closures baked in.  The
        #   iteration saddle's mf_eq RHS is evaluated in formal-rename
        #   mode (so e.g. ``nstar = phi(v) + b`` becomes
        #   ``nstar → phi0_<i+1> + b``).  Compound saddles like vstar
        #   are evaluated concretely with the closure substitution
        #   applied to the result, eliminating raw nstar from vstar's
        #   substitution.  This makes the (1, 0) tadpole vanish
        #   symbolically for any saddle EOM form.
        #
        # - ``self._mf_bg_solver`` (numerical solver): vstar in raw
        #   concrete form (with raw nstar so ``solve_mean_field`` can
        #   iterate on it).  The iteration saddle (nstar) is NOT
        #   substituted in this dict.
        if self._mf_eqs_text:
            # Iteration-saddle selection.  For legacy single-pop
            # theories there is one ``nstar`` mf_eq with a single
            # function call (``phi(vstar)``); the compiler closes the
            # saddle EOM by substituting ``nstar → phi0_i`` formally.
            # Heterogeneous-pop theories have ONE ``<n>star`` per
            # population (e.g. ``nEstar = phiE(vEstar)``,
            # ``nIstar = phiI(vIstar)``); each is its own iteration
            # saddle.  The ``'AUTO'`` sentinel tells _classify_mf_eqs
            # to treat every mf_eq whose RHS is a single function
            # call as a closure saddle.
            iter_sentinel = ('AUTO' if self.populations else 'nstar')
            self._mf_bg = make_mf_bg_conditions_lambda(
                self._mf_eqs_text, primary_fn_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
                iteration_saddle = iter_sentinel,
            )
            self._mf_bg_solver = make_mf_bg_solver_lambda(
                self._mf_eqs_text, primary_fn_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
                iteration_saddle = iter_sentinel,
            )
            self._mf_equations = make_mf_equations_lambda(
                self._mf_eqs_text, primary_fn_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
            )

        # Kernel frequency images
        if self._kernel_specs:
            self._kernel_ft_image = make_kernel_ft_image_lambda(
                self._kernel_specs,
                param_names = param_names,
            )
            # Time-domain images — same kernel specs, but evaluated
            # in ``t`` rather than ``omega``.  Used by the Phase J
            # integrator to substitute conductance-vertex kernels at
            # the per-leg τ symbol.
            self._kernel_td_image = make_kernel_td_image_lambda(
                self._kernel_specs,
                param_names = param_names,
            )

        # Correlated noises (CGF cumulant terms)
        if self._cgf_terms:
            self._correlated_noises.update(
                make_correlated_noises_block(
                    self._cgf_terms, param_names=param_names))

        # ``mf_substitutions`` for matrix-shaped parameters.  The
        # MSR-JD framework expects each matrix-indexed parameter
        # to have a substitution that produces ``[[w_{i+1}{j+1}]
        # for j] for i]`` — the elementwise SR-var grid.
        #
        # Sizing rules (in priority order):
        #   * ``indexed_by=[A, B]``  → axes use pop sizes of A, B
        #     (heterogeneous-population path).  Reads
        #     ``ns._pop_size`` if available, else falls back to
        #     ``ns.pop``.
        #   * legacy ``matrix=True`` only  → both axes use ``ns.pop``.
        #
        # The ``domain`` (e.g. 'positive') is propagated to every
        # element SR var so symbolic FT / Maxima can dispatch the
        # right assumption.
        for p in self.parameters:
            if not p.matrix:
                continue
            wname = p.name
            wdom  = p.domain
            wib   = p.indexed_by    # ['A', 'B'] or None
            def _matrix_subst(ns, _w=wname, _d=wdom, _ib=wib):
                # Resolve the row / column index sets.
                if _ib:
                    pop_size = getattr(ns, '_pop_size', {}) or {}
                    n_rows = pop_size.get(_ib[0], len(ns.pop))
                    n_cols = pop_size.get(_ib[1], len(ns.pop))
                else:
                    n_rows = n_cols = len(ns.pop)
                if _d:
                    return [[SR.var(f'{_w}{i+1}{j+1}', domain=_d)
                             for j in range(n_cols)]
                            for i in range(n_rows)]
                return [[SR.var(f'{_w}{i+1}{j+1}')
                         for j in range(n_cols)]
                        for i in range(n_rows)]
            self._mf_substitutions.append({
                'name':  wname,
                'value': _matrix_subst,
            })

    # ── Build the MSR-JD model dict (same structure as the example
    #    HAWKES_MODEL) ───────────────────────────────────────────────
    def _inject_autopop(self) -> None:
        """Scalar-mode autopop: inject a single-position population ``pop``
        (size 1) and bind every unbound physical/response field to it.  Used
        by ``build()`` so it never depends on the public ``population()``
        method (which lives in the temporal mixin after the builder split).

        Only fires for scalar theories (``n_populations <= 1``); the
        ``build()`` caller gates on that.  Legacy multi-population theories
        (``n_populations >= 2`` with no explicit ``.population(...)``) skip
        autopop and keep an empty ``populations`` list — see the call
        site for why."""
        self.populations.append({
            'name':        'pop',
            'size':        1,
            'description': 'auto-injected scalar population',
        })
        self.n_populations = len(self.populations)
        for f in self.physical_fields:
            if f.population is None:
                f.population = 'pop'
        for f in self.response_fields:
            if getattr(f, 'population', None) is None:
                f.population = 'pop'

    def _resolve_spatial_dim(self) -> int:
        """Resolve the theory's single spatial dimension from the physical
        fields (``0`` = time-only); validate the v1 single-dimension
        constraint.  Overridden by the forward builders to assert their
        domain (Temporal: must be 0; Spatial: must be > 0)."""
        spatial_dims = {int(getattr(f, 'spatial_dim', 0) or 0)
                        for f in self.physical_fields}
        nonzero_dims = {d for d in spatial_dims if d > 0}
        if len(nonzero_dims) > 1:
            raise ValueError(
                f'TheoryBuilder("{self.name}").build(): v1 does not '
                f'support mixed spatial dimensions; physical fields '
                f'declare dims {sorted(nonzero_dims)}.  All spatial '
                f'fields must share one dimension (mixed-dim is a v2 '
                f'feature — see docs/spatial_implementation_plan.md '
                f'§"Out of v1 scope").')
        return next(iter(nonzero_dims), 0)

    def build(self) -> dict:
        """Emit the MSR-JD model dict (same structure as the example
        HAWKES_MODEL).

        Compiles text-based declarations (set_action_text,
        define_function, set_mf_equation, declare_cgf_term, …) to
        lambdas and uses them in place of the corresponding lambda
        hooks.  Validates that all required hooks ended up populated.
        Raises ValueError with a clear message if anything is missing.
        """
        # ── Scalar-mode autopop ────────────────────────────────────
        # When no ``.population(...)`` has been declared but physical
        # fields exist, auto-inject a single-position population
        # ``pop`` of size 1 so the rest of the build machinery (which
        # was originally designed for heterogeneous populations) runs
        # uniformly.  Each physical field without a ``population``
        # attribute gets bound to this auto-pop.  Combined with the
        # ``_FieldScalar`` / ``_FullPhysicalField`` scalar-arithmetic
        # wrappers in ``theory_compiler``, this lets users write
        # truly scalar theories — bare ``xt * x`` instead of
        # ``sum(xt[i]*x[i] for i in pop)``.
        #
        # Gate on ``n_populations <= 1``: the autopop always injects a
        # size-1 population, so firing it for a legacy
        # ``n_populations >= 2`` theory (e.g. the 2-pop Hawkes models,
        # which declare indexed fields but never call
        # ``.population(...)``) would clobber the requested population
        # count down to 1 — dropping ``nt2``/``vt2``/… and zeroing
        # every cross-population correlator.  Those theories instead
        # keep an empty ``populations`` list and fall through to the
        # flat ``index_sets = {'pop': range(n_populations)}`` legacy
        # path (and the matching legacy ``solve_mean_field`` branch),
        # exactly as the hand-written ``simulations/hawkes_*`` dicts do.
        if (not self.populations and self.physical_fields
                and self.n_populations <= 1):
            self._inject_autopop()

        # Apply the colored-noise → Markovian-embedding preprocessor
        # BEFORE compiling text declarations: this rewrite mutates
        # ``_cgf_terms`` (white-noise auxiliaries replace colored
        # rows) and augments ``_action_text`` (couples each source
        # field to its auxiliary, adds the OU kinetic term).  Done
        # here, before ``_compile_text_declarations``, so the
        # downstream compiler sees the augmented spec.
        #
        # No-op when the builder-level toggle is OFF and no row
        # opts in explicitly.  See ``api/colored_to_markovian.py``
        # and ``docs/correlated_noise_capabilities.md`` §1.5.
        if self._cgf_terms:
            from api.colored_to_markovian import markovianize_spec
            markovianize_spec(self)

        # Compile any text declarations into lambdas.  Done here, not
        # in the setters, so that all decls are available when each
        # lambda's namespace is assembled.
        self._compile_text_declarations()

        missing = []
        if self._action is None:           missing.append('action')
        # ``phi_concrete`` is required only when there is at least one
        # single-arg function in the model — the saddle solver uses it
        # to evaluate ``phi(v*)`` numerically.  If every declared
        # function is multi-arg, ``phi_concrete`` is not applicable
        # (saddle solving falls back to the user's mf_eq RHS via the
        # ``nstar_target`` path in ``solve_mean_field``).
        any_single_arg = any(len(f.get('args') or []) == 1
                             for f in self._function_specs)
        if self._phi_concrete is None and any_single_arg:
            missing.append('phi_concrete')
        # kernel_ft_image is OPTIONAL: the DeltaKernel template
        # intentionally returns no image (the symbol is treated as
        # δ(t) directly by the cell-8 propagator construction).
        if missing:
            raise ValueError(
                f'TheoryBuilder("{self.name}").build(): missing required '
                f'hooks: {missing}.  Use set_<hook>(...) before .build().'
            )

        # Naming convention — lets the pipeline accessors translate
        # between user-facing physical letters ('n', 'v', 'r', ...) and
        # the internal MSR-JD names ('dn', 'vstar', etc.).  Built only
        # from explicit user declarations; the pipeline applies a
        # sensible n/v/m fallback when this dict is absent.
        fluct_map: dict[str, str] = {}
        for f in self.physical_fields:
            if f.natural_name:
                fluct_map[f.natural_name] = f.name

        saddle_map: dict[str, str] = {}
        mf_param_names: list[str] = []
        for p in self.parameters:
            if p.mean_field:
                mf_param_names.append(p.name)
                if p.natural_name:
                    saddle_map[p.natural_name] = p.name

        naming_convention = {
            'fluctuation_fields': fluct_map,    # natural → internal
            'mean_field_saddles': saddle_map,   # natural → internal
            'mf_parameters':      mf_param_names,   # internal names
        }

        # Heterogeneous-population index sets.  When .population() has
        # been called, ``populations`` lists ``{'name', 'size'}`` per
        # group.  Index conventions:
        #
        #   * ``pop_<name>`` — LOCAL indices into population <name>,
        #     i.e. ``[0, 1, ..., size-1]``.  Fields, parameters, and
        #     kernels indexed by population <name> use these — e.g.
        #     ``nE[i]`` for ``i in pop_E`` picks the i-th E element.
        #   * ``pop`` — flat catch-all, sized ``sum(sizes)``.  Kept for
        #     legacy callers that don't yet know about per-population
        #     structure (the diagram engine, etc.).
        if self.populations:
            total_size = sum(int(p.get('size', 1)) for p in self.populations)
            flat_pop = list(range(total_size))
            extra_index_sets: dict = {
                f'pop_{p["name"]}': list(range(int(p.get('size', 1))))
                for p in self.populations
            }
        else:
            flat_pop = list(range(self.n_populations))
            extra_index_sets = {}

        # Identify iteration saddles by classifying mf_eqs the same
        # way the compiler does: saddles whose RHS is a single function
        # call are closure / iteration saddles; the rest are compound.
        # Solve_mean_field reads this to know which saddles to iterate
        # on (per-pop sized vector) and which to evaluate compoundly.
        iter_saddle_names: list[str] = []
        if self._mf_eqs_text:
            from api.theory_compiler import (
                _classify_mf_eqs as _classify,
            )
            iter_sentinel = ('AUTO' if self.populations else 'nstar')
            classification = _classify(self._mf_eqs_text, iter_sentinel)
            iter_saddle_names = list(classification['closure'].keys())

        # ── Spatial-extension resolution (v1) ──────────────────────
        # Compute the theory's spatial dimension, validate the v1
        # single-dim constraint, resolve the inline-number boundary
        # ``length`` shortcut, and build the operators list (adding a
        # ``Laplacian`` symbol when spatial).  See
        # ``docs/spatial_design_decisions_v1.md``.
        theory_spatial_dim = self._resolve_spatial_dim()
        is_spatial = theory_spatial_dim > 0

        # ── Reserved-name validation (scoped hard error) ─────────────
        # Names owned by the framework's symbol machinery: reusing one
        # as a field / parameter / function / kernel name collides —
        # SILENTLY — with the Fourier-transform, operator, or spatial-
        # coordinate symbols.  ``t`` / ``omega`` are the FT variables
        # (``SR.var('t')`` / ``SR.var('omega')`` in _propagator.py) and
        # ``Dt`` / ``delta_D`` / ``delta_Dp`` are operator symbols, so
        # they're reserved in EVERY theory.  ``k`` (wavevector,
        # ``SR.var('k')`` in heat_kernel.py — a real silent-wrong-mass
        # bug if a parameter shares it), ``Laplacian`` (diffusion
        # operator) and the spatial coordinates ``x`` / ``y`` / ``z``
        # (also the C(x,τ) output axis) are reserved only when the
        # theory is spatial.  Time-only theories keep ``x`` free — it's
        # the conventional default field name.  See
        # ``docs/spatial_design_decisions_v1.md`` (Decision: reserved
        # names).
        _reserved_always = {'t', 'omega', 'Dt', 'delta_D', 'delta_Dp'}
        _reserved_spatial = {'k', 'Laplacian', 'x', 'y', 'z'}
        reserved = set(_reserved_always)
        if is_spatial:
            reserved |= _reserved_spatial
        _named = []
        for f in self.physical_fields:
            _named.append(('physical field',
                           getattr(f, 'natural_name', None) or f.name))
        for p in self.parameters:
            _named.append(('parameter', p.name))
        for fn in getattr(self, '_functions', []) or []:
            _named.append(('function', getattr(fn, 'name', None)))
        for kr in self.kernels:
            _named.append(('kernel', getattr(kr, 'name', None)))
        _rename_hint = {'x': 'phi', 'y': 'psi', 'z': 'chi', 'k': 'kappa',
                        't': 'tau_var', 'omega': 'w0',
                        'Dt': 'Dt_', 'Laplacian': 'lap_'}
        for kind, nm in _named:
            if nm and nm in reserved:
                spatial_word = nm in _reserved_spatial
                scope = ('spatial theories'
                         if spatial_word else 'all theories')
                role = ('spatial coordinate / wavevector / diffusion '
                        'operator' if spatial_word
                        else 'time / frequency / time-derivative '
                        'operator')
                raise ValueError(
                    f'TheoryBuilder("{self.name}").build(): {kind} name '
                    f'{nm!r} is reserved ({scope}); it collides with the '
                    f'framework\'s {role} symbol of the same name.  '
                    f'Rename it (e.g. {_rename_hint.get(nm, nm + "_")!r}).  '
                    f'In a spatial theory, x/y/z are the spatial '
                    f'coordinates, k is the wavevector, and Laplacian is '
                    f'the diffusion operator — see '
                    f'docs/spatial_design_decisions_v1.md.')

        # Operators list: default [Dt]; append a Laplacian symbol when
        # the theory is spatial so ``field_theory._build_namespace``
        # registers ``ns.Laplacian`` (used multiplicatively in the
        # action exactly like ``Dt``).
        operators_list = list(self._operators) or [
            {'name': 'Dt', 'sage_name': 'Dt',
             'latex_name': r'\partial_t', 'description': 'd/dt'},
        ]
        if is_spatial and not any(o.get('name') == 'Laplacian'
                                  for o in operators_list):
            operators_list = operators_list + [
                {'name': 'Laplacian', 'sage_name': 'Laplacian',
                 'latex_name': r'\nabla^2',
                 'description': 'spatial Laplacian ∇²'},
            ]

        # Boundary / initial validation + inline-length resolution.
        boundary_block = None
        if self._boundary is not None:
            if not is_spatial:
                raise ValueError(
                    f'TheoryBuilder("{self.name}").build(): .boundary() '
                    f'was declared but no physical field is spatial '
                    f'(set spatial_dim≥1 on a field or call '
                    f'.spatial_dim(d)).')
            boundary_block = dict(self._boundary)
            length = boundary_block.get('length')
            if length is not None and not isinstance(length, str):
                # Inline-number shortcut (D2): back it with a hidden
                # positive parameter so downstream code always sees a
                # named, sweepable parameter.
                hidden = '_pbc_length_L0'
                if not any(p.name == hidden for p in self.parameters):
                    self.parameter(hidden, default=float(length),
                                   domain='positive',
                                   description='auto-created PBC length '
                                               '(inline-number shortcut)')
                boundary_block['length'] = hidden

        initial_block = dict(self._initial) if self._initial is not None else None

        # Dyson policy / reference diffusion on a non-spatial theory is
        # a declaration error (mirrors the .boundary() check above).
        if self._dyson is not None and not is_spatial:
            raise ValueError(
                f'TheoryBuilder("{self.name}").build(): .dyson() '
                f'was declared but no physical field is spatial '
                f'(set spatial_dim≥1 on a field or call '
                f'.spatial_dim(d)).')
        if self._reference_diffusion is not None and not is_spatial:
            raise ValueError(
                f'TheoryBuilder("{self.name}").build(): '
                f'.reference_diffusion() was declared but no physical '
                f'field is spatial (set spatial_dim≥1 on a field or '
                f'call .spatial_dim(d)).')

        model = {
            'name':            self.name,
            'populations':     list(self.populations),    # heterogeneous metadata
            'iteration_saddles': iter_saddle_names,
            'index_sets':      {'pop': flat_pop, **extra_index_sets},
            'response_fields': [self._field_dict(f) for f in self.response_fields],
            'physical_fields': [self._field_dict(f) for f in self.physical_fields],
            'parameters':      [self._param_dict(p) for p in self.parameters],
            'kernels':         [self._kernel_dict(k) for k in self.kernels],
            # Explicit DAE residuals (new MF-solver path).  When the
            # list is non-empty, compute_cumulants routes through
            # ``solve_mean_field`` (multi-start Newton + sort +
            # ``fixed_point_index`` selection + linear stability)
            # instead of the legacy iteration solver in
            # ``api._solve_mf``.
            'equations':       [dict(eq) for eq in self._equations],
            'operators':       operators_list,
            'functions':       list(self._functions),
            'mf_substitutions': self._mf_substitutions,
            'phi_concrete':    self._phi_concrete,
            'action':          self._action,
            # Human-readable action text (what the user typed via
            # set_action_text, with any Noise-tab sources appended).
            # None for template-built theories; surfaced by
            # daedalus.describe_model.
            'action_text':     self._action_text,
            'naming_convention': naming_convention,
            # Whether the DAE solver classifies linear stability and
            # restricts ``fixed_point_index`` to the stable subset.
            # Defaults to False — set via ``.stability_analysis(True)``
            # for bistable / multi-saddle differential theories.
            'stability_analysis': bool(self._stability_analysis),
            # Spatial v2: parse the action with the operator IR
            # (Lap()/Dt()/Dx() binding syntax).  Default False.
            'operator_ir':     bool(self._operator_ir),
        }
        if self._mf_bg is not None:
            # Action-side dict (closure-baked): used by
            # FieldTheory.expand for the symbolic action substitution.
            model['mf_bg_conditions_action'] = self._mf_bg
        if self._mf_bg_solver is not None:
            # Solver-friendly dict (raw concrete vstar, no nstar):
            # used by solve_mean_field.  Stored under the legacy key
            # name ``mf_bg_conditions`` so existing loaders keep
            # working.
            model['mf_bg_conditions'] = self._mf_bg_solver
        elif self._mf_bg is not None:
            # Fallback: only one mf_bg exists (legacy template path).
            model['mf_bg_conditions'] = self._mf_bg
        if self._mf_equations is not None:
            model['mf_equations'] = self._mf_equations
        if self._kernel_ft_image is not None:
            model['kernel_ft_image'] = self._kernel_ft_image
        if self._kernel_td_image is not None:
            model['kernel_td_image'] = self._kernel_td_image
        if self._specializations is not None:
            model['specializations'] = self._specializations
        if self._correlated_noises:
            model['correlated_noises'] = self._correlated_noises

        # ── Spatial-extension blocks (v1) ──────────────────────────
        # Emitted only when the theory is spatial, so time-only models
        # see no change to their model dict.  Consumed by the
        # propagator builder (Phase 2/3) and compute_cumulants (Phase 4).
        if is_spatial:
            model['spatial'] = {
                'dim': theory_spatial_dim,
                'fields_with_spatial': [
                    f.name for f in self.physical_fields
                    if int(getattr(f, 'spatial_dim', 0) or 0) > 0
                ],
            }
            # Dyson policy defaults to {'mode': 'off'} when .dyson()
            # was never declared (scalar-D₀ reference propagator only;
            # exact iff 𝒟̂ = 0).  See
            # docs/dyson_duhamel_integration_plan.md (D-4).
            model['spatial']['dyson'] = (dict(self._dyson) if self._dyson
                                         else {'mode': 'off'})
            # Reference diffusion D₀ is emitted only when declared —
            # absent means "let the propagator builder pick".
            if self._reference_diffusion is not None:
                model['spatial']['reference_diffusion'] = \
                    self._reference_diffusion
            # Boundary defaults to 'infinite' when the theory is spatial
            # but no .boundary() was declared.
            model['boundary'] = boundary_block or {'mode': 'infinite'}
            # Initial defaults to 'stationary'.
            model['initial'] = initial_block or {'mode': 'stationary'}
        else:
            # Defensive: a .boundary()/.dyson()/.reference_diffusion()
            # on a non-spatial theory already raised in the resolution
            # block above, so there is nothing to emit here.
            pass
        return model

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _field_dict(f: FieldSpec) -> dict:
        d = {'name': f.name, 'indexed': f.indexed, 'latex': f.latex}
        if f.description:
            d['description'] = f.description
        if f.natural_name:
            d['natural_name'] = f.natural_name
        if f.population:
            d['population'] = f.population
        if getattr(f, 'spatial_dim', 0):
            d['spatial_dim'] = int(f.spatial_dim)
        return d

    @staticmethod
    def _param_dict(p: ParameterSpec) -> dict:
        d = {'name': p.name, 'indexed': p.indexed}
        if p.domain:
            d['domain'] = p.domain
        if p.description:
            d['description'] = p.description
        if p.mean_field:
            d['mean_field'] = True
        if p.natural_name:
            d['natural_name'] = p.natural_name
        if p.indexed_by is not None:
            d['indexed_by'] = list(p.indexed_by)
        if p.default is not None:
            d['default'] = p.default
        return d

    @staticmethod
    def _kernel_dict(k: KernelSpec) -> dict:
        d = {
            'name':       k.name,
            'sage_name':  k.sage_name,
            'latex_name': k.latex_name,
        }
        if k.description:
            d['description'] = k.description
        # Encode indexing so FieldTheory._build_namespace and
        # make_kernel_ft_image_lambda can create the right symbol
        # shape.  ``matrix`` takes precedence over ``vector`` because
        # KernelSpec stores them as two independent bools (``matrix``
        # implies ``indexed``).  Scalar (default) is omitted.
        if k.matrix:
            d['indexed'] = 'matrix'
        elif k.indexed:
            d['indexed'] = 'vector'
        if k.indexed_by is not None:
            d['indexed_by'] = list(k.indexed_by)
        return d


class TemporalTheoryBuilder(_TemporalMethods, _BaseTheoryBuilder):
    """Authoring API for time-domain (non-spatial) theories: carries the
    temporal-only methods (populations, kernels, colored-noise/markovianize,
    action templates); has no spatial methods."""

    def _resolve_spatial_dim(self) -> int:
        d = super()._resolve_spatial_dim()
        if d > 0:
            raise ValueError(
                f'TemporalTheoryBuilder("{self.name}"): a physical field '
                f'declared spatial_dim={d} > 0.  Use SpatialTheoryBuilder for '
                f'spatial (PDE) theories.')
        return 0


class SpatialTheoryBuilder(_SpatialMethods, _BaseTheoryBuilder):
    """Authoring API for spatial (PDE / field-theory) theories: carries the
    spatial-only methods (spatial_dim, boundary, initial); has no temporal
    kernel/colored-noise methods.  Multi-field spatial theories use repeated
    physical_field(...) calls (not populations)."""

    def __init__(self, name: str, n_populations: int = 0):
        super().__init__(name, n_populations=n_populations)

    def _resolve_spatial_dim(self) -> int:
        d = super()._resolve_spatial_dim()
        if d <= 0:
            raise ValueError(
                f'SpatialTheoryBuilder("{self.name}"): no physical field is '
                f'spatial.  Set spatial_dim>=1 on a field '
                f'(physical_field(..., spatial_dim=d)) or call .spatial_dim(d).')
        return d


class TheoryBuilder(_SpatialMethods, _TemporalMethods, _BaseTheoryBuilder):
    """Back-compat unified builder: exposes BOTH method sets and auto-detects
    spatial-vs-temporal at build() (via the base _resolve_spatial_dim).
    Behaviorally identical to the pre-split TheoryBuilder.  New code should
    prefer TemporalTheoryBuilder / SpatialTheoryBuilder."""
    pass

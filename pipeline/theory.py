"""
pipeline.theory — declarative theory input (PROTOTYPE / STUB).

The current MSR-JD pipeline takes a Python dict (``HAWKES_MODEL``) with
several Sage-aware lambdas (action, mf_bg_conditions, kernel_ft_image,
phi_concrete, mf_equations).  Writing one of these by hand requires
familiarity with Sage's symbolic algebra and the framework's
conventions.

The goal of this module is a higher-level, more-user-friendly way to
describe a theory and have the framework generate the model dict
automatically.  Two designs under consideration:

  A) Python builder API (this file's direction)::

        from pipeline.theory import TheoryBuilder, HawkesAction

        t = TheoryBuilder(name="My Hawkes 2-pop", n_populations=2)
        t.response_field('nt', latex=r'\\tilde{n}')
        t.response_field('vt', latex=r'\\tilde{v}')
        t.physical_field('dn', latex=r'\\delta\\dot{n}')
        t.physical_field('dv', latex=r'\\delta v')
        t.parameter('a',     default=1.0)
        t.parameter('tau',   default=10.0,  domain='positive')
        t.parameter('tau_g', default=2.5,   domain='positive')
        t.parameter('E',     default=[1.1, 1.05], indexed='vector')
        t.parameter('w',     default=[[0.35, 0.4], [0.3, 0.5]],
                             indexed='matrix')
        t.kernel('g', frequency_image=lambda omega, p:
                       1 / (1 + 1j * omega * p['tau_g']))
        t.transfer_function('phi', form='linear')   # or 'quadratic'

        # Standard Hawkes action template:
        t.use_action_template(HawkesAction(
            phi_name='phi', kernel_name='g',
            external_drive='E', recurrent_weight='w',
        ))
        # Alternatively, drop down to a Sage lambda for unusual actions:
        # t.set_action(lambda ns: ...)

        model = t.build()    # produces HAWKES_MODEL dict

  B) YAML / JSON schema with a small expression DSL.  Cleaner for
     non-programmers and shareable by config file, but loses some
     expressiveness without a parser.

This file currently provides only the SCAFFOLDING for option A — a
``TheoryBuilder`` class that accumulates declarations and emits a
model dict.  The action / saddle-equation / kernel-image lambdas
still need to be supplied by the user (or by a template) until the
template library is fleshed out.

Roadmap:
  1. Flesh out ``TheoryBuilder.build()`` to produce a working dict
     from the field/parameter/kernel declarations alone (no template).
  2. Add common templates: ``HawkesAction(phi='linear')``,
     ``HawkesAction(phi='quadratic')``, ``OUNoise(...)``, etc.
  3. Add a YAML loader that produces a ``TheoryBuilder`` from a
     declarative file.
  4. Wire ``TheoryBuilder`` into ``pipeline.compute_cumulants`` so
     the user can pass either a model dict OR a Theory.
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


@dataclass
class KernelSpec:
    name: str
    sage_name: str = ''           # internal SR var name (default 'z_'+name)
    latex_name: str = ''
    frequency_image: Optional[Callable] = None  # lambda omega, params: SR
    description: str = ''


class TheoryBuilder:
    """
    Accumulator for a model declaration.  Call ``.build()`` to emit the
    HAWKES_MODEL dict suitable for ``pipeline.compute_cumulants``.

    NOTE: prototype.  The action / mf_bg_conditions / kernel_ft_image
    lambdas still need to be set explicitly via ``set_action(...)`` etc.
    Templates ("HawkesAction") are roadmap items.
    """

    def __init__(self, name: str, n_populations: int = 1):
        self.name = name
        self.n_populations = n_populations
        self.response_fields: list[FieldSpec] = []
        self.physical_fields: list[FieldSpec] = []
        self.parameters:      list[ParameterSpec] = []
        self.kernels:         list[KernelSpec] = []
        self._action: Optional[Callable] = None
        self._mf_bg: Optional[Callable] = None
        self._mf_equations: Optional[Callable] = None
        self._kernel_ft_image: Optional[Callable] = None
        self._phi_concrete: Optional[Callable] = None
        self._specializations: Optional[Callable] = None
        self._mf_substitutions: list[dict] = []
        self._functions: list[dict] = []
        self._operators: list[dict] = []
        self._correlated_noises: dict = {}

        # Text-driven declarations (used by the UI and by
        # ``define_function`` / ``set_action_text`` / etc.).  When any
        # of these are populated, ``build()`` compiles them to lambdas
        # via ``pipeline.theory_compiler``, overriding the lambda hooks
        # above.  Empty by default → existing template-based flow runs
        # unchanged.
        self._function_specs:   list[dict] = []
        self._kernel_specs:     list[dict] = []   # text time/freq exprs
        self._action_text:      Optional[str] = None
        self._mf_eqs_text:      dict[str, str] = {}
        self._cgf_terms:        list[dict] = []
        self._phi_function_name: Optional[str] = None

    # ── Field declarations ─────────────────────────────────────────
    def response_field(self, name: str, indexed: bool = True,
                       latex: str = '', description: str = ''):
        self.response_fields.append(FieldSpec(
            name=name, indexed=indexed,
            latex=latex or name, description=description,
        ))
        return self

    def physical_field(self, name: str, indexed: bool = True,
                       latex: str = '', description: str = '',
                       natural_name: str = None,
                       auto_response: bool = True,
                       auto_saddle: bool = True):
        """Declare a physical field.

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

        self.physical_fields.append(FieldSpec(
            name=internal_name, indexed=indexed,
            latex=latex or rf'\delta {natural_name}',
            description=description,
            natural_name=natural_name,
        ))

        # Auto-generate the conjugate response field as ``<natural>t``
        # (matching the existing nt/vt/mt convention).  Skipped if the
        # user already declared a response field with that name.
        if auto_response:
            response_name = f'{natural_name}t'
            if not any(f.name == response_name for f in self.response_fields):
                self.response_field(
                    response_name, indexed=indexed,
                    latex=rf'\tilde {natural_name}',
                    description=f'response field conjugate to {natural_name}',
                )

        # Auto-generate the saddle parameter ``<natural>star``.  Default
        # domain is ``positive`` for ``n`` (rates) and free for others;
        # the user can override by declaring the parameter explicitly
        # before / after this call (the duplicate check below skips
        # re-adding).
        if auto_saddle:
            saddle_name = f'{natural_name}star'
            if not any(p.name == saddle_name for p in self.parameters):
                # Heuristic: rates are positive; voltages and other
                # fields are free.  Users can edit the parameter row
                # in the UI to refine.
                domain = 'positive' if natural_name == 'n' else None
                self.parameter(
                    saddle_name, indexed=True, domain=domain,
                    mean_field=True, natural_name=natural_name,
                    description=f'mean-field saddle value of {natural_name}',
                )
        return self

    # ── Parameter declarations ─────────────────────────────────────
    def parameter(self, name: str, default: Any = None,
                  indexed=False, domain: str = None, description: str = '',
                  mean_field: bool = False, natural_name: str = None):
        """Declare a model parameter.

        Parameters
        ----------
        name : str
            Internal parameter name (e.g. ``'nstar'`` for the saddle
            firing rate, ``'tau'`` for a time constant).
        default : Any, optional
            Default numerical value (scalar / list / matrix).
        indexed : bool or 'vector' or 'matrix', default False
            Whether the parameter is per-population (vector) or
            per-pair-of-populations (matrix).
        domain : str, optional
            ``'positive'`` etc.  Used by the FieldTheory builder.
        mean_field : bool, default False
            ``True`` flags this parameter as a saddle-point quantity
            (``n*``, ``v*``, ``m*``, ...) so the pipeline's MF
            accessor and saver can discover it without hardcoded
            name lookups.
        natural_name : str, optional
            User-facing letter for MF accessor lookup.  If
            ``parameter('nstar', mean_field=True, natural_name='n')``
            is declared, ``mf['n', 1]`` returns ``n*_1``.
        """
        is_vec = (indexed in (True, 'vector'))
        is_mat = (indexed == 'matrix')
        self.parameters.append(ParameterSpec(
            name=name, indexed=(is_vec or is_mat), matrix=is_mat,
            domain=domain, default=default, description=description,
            mean_field=mean_field, natural_name=natural_name,
        ))
        return self

    # ── Kernel declaration ─────────────────────────────────────────
    def kernel(self, name: str, frequency_image: Callable = None,
               sage_name: str = '', latex_name: str = '',
               description: str = ''):
        self.kernels.append(KernelSpec(
            name=name, sage_name=sage_name or f'z_{name}',
            latex_name=latex_name or name,
            frequency_image=frequency_image, description=description,
        ))
        return self

    # ── Text-driven theory declaration (UI-friendly) ──────────────
    # Each method below stores a Sage-syntax text string.  At
    # ``.build()`` time the strings are compiled to Python lambdas via
    # ``pipeline.theory_compiler`` and dropped into the model dict.
    # Calling any of these overrides the corresponding lambda hook.

    def define_function(self, name: str, args: list[str],
                        expression: str, latex: str = None,
                        description: str = ''):
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
        latex : str, optional
            Display form (defaults to ``name``).
        """
        self._function_specs.append({
            'name':        name,
            'args':        list(args),
            'expression':  expression,
            'latex':       latex or name,
            'description': description,
        })
        return self

    def define_kernel(self, name: str, *, time_expr: str = None,
                      freq_image: str = None, latex_name: str = '',
                      sage_name: str = '', description: str = ''):
        """Declare a convolution kernel via either its time-domain
        expression OR its frequency image (or both).

        At least one of ``time_expr`` / ``freq_image`` must be given.
        ``freq_image`` is preferred because it's used directly by
        the propagator construction; if only ``time_expr`` is given,
        the build will warn that the FT must be supplied.

        The kernel is also registered as a regular ``kernel(...)``
        symbol for the MSR-JD framework's name resolution.
        """
        if time_expr is None and freq_image is None:
            raise ValueError(
                f"define_kernel({name!r}): supply at least one of "
                f"time_expr= or freq_image=")
        # Register the kernel symbol with the regular .kernel() call
        if not any(k.name == name for k in self.kernels):
            self.kernel(name, sage_name=sage_name or f'z_{name}',
                        latex_name=latex_name or name,
                        description=description)
        self._kernel_specs.append({
            'name':       name,
            'time_expr':  time_expr,
            'freq_image': freq_image,
        })
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
        """
        self._action_text = text
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

    def declare_cgf_term(self, name: str, response_field: str, order: int,
                         coefficient: str, kernel: str = None):
        """Add one term to a non-closed-form cumulant generating
        functional (e.g. GTaS noise).

        Parameters
        ----------
        name : str
            CGF identifier — multiple rows with the same name + order
            sum into a single cumulant.
        response_field : str
            The response-field this CGF couples to (e.g. ``'mt'``).
        order : int
            Cumulant order: 2 for κ⁽²⁾, 3 for κ⁽³⁾, 4 for κ⁽⁴⁾, ...
        coefficient : str
            Sage-syntax expression for the cumulant coefficient
            (e.g. ``'lambda_X * p_part'``).
        kernel : str, optional
            Optional time-domain kernel multiplier
            (e.g. ``'_gauss(tau, mu_shift, sigma_sq)'`` for a
            cross-cumulant Gaussian factor).
        """
        self._cgf_terms.append({
            'name':           name,
            'response_field': response_field,
            'order':          int(order),
            'coefficient':    coefficient,
            'kernel':         kernel,
        })
        return self

    def set_transfer_function(self, name: str = 'phi'):
        """Mark which declared function plays the role of the MSR-JD
        ``phi_concrete`` (the saddle-point Taylor-expansion target).

        Defaults to ``'phi'``; only needed if you've named the transfer
        function differently.
        """
        self._phi_function_name = name
        return self

    # ── Lambda hooks for the harder stuff (until templates land) ───
    def set_action(self, fn: Callable):
        """``fn(ns) -> SR``  — the model action.  See
        models/hawkes_quad_expg_gtas.py for examples."""
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

    def correlated_noise(self, name: str, **kwargs):
        self._correlated_noises[name] = kwargs
        return self

    # ── High-level template wiring ─────────────────────────────────
    def use_action_template(self, template):
        """Apply a HawkesAction (or compatible) template.  The template
        emits the action / mf_bg_conditions / mf_equations /
        specializations / phi_concrete / mf_substitutions / functions
        block; everything is registered on this builder.
        """
        self._action          = template.action()
        self._phi_concrete    = template.phi_concrete()
        self._specializations = template.specializations()
        self._mf_bg           = template.mf_bg_conditions()
        self._mf_equations    = template.mf_equations()
        self._mf_substitutions = template.mf_substitutions()
        self._functions       = list(template.functions_list())
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
        self.physical_field(
            template.physical_field, indexed=True,
            latex=r'\delta m',
            description='GTaS external rate fluctuation (zero-mean)',
            natural_name='m',
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

    # ── Text → lambda compilation ─────────────────────────────────
    def _compile_text_declarations(self) -> None:
        """Walk the text-based declarations and compile each into the
        corresponding lambda hook.  No-op if no text declarations
        were made (template-only builders go through unchanged)."""
        if not (self._action_text or self._function_specs
                or self._kernel_specs or self._mf_eqs_text
                or self._cgf_terms):
            return

        from pipeline.theory_compiler import (
            make_action_lambda,
            make_phi_concrete_lambda,
            make_specializations_lambda,
            make_mf_bg_conditions_lambda,
            make_mf_equations_lambda,
            make_kernel_ft_image_lambda,
            make_correlated_noises_block,
        )

        n_pop = self.n_populations
        field_names  = ([f.name for f in self.response_fields]
                        + [f.name for f in self.physical_fields])
        param_names  = [p.name for p in self.parameters]
        kernel_names = [k.name for k in self.kernels]

        # Identify the transfer function (defaults to one named 'phi')
        phi_name = self._phi_function_name or 'phi'
        phi_spec = next((f for f in self._function_specs
                         if f['name'] == phi_name), None)

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

        # Action — write the FULL action; framework no longer wraps
        # with sum-over-i.  ``phi[i](dv[i])`` is the canonical syntax
        # for the transfer function (indexed formal call).
        if self._action_text is not None:
            self._action = make_action_lambda(
                self._action_text,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
                transfer_function = phi_spec,
                naming_convention = _action_naming_convention,
            )

        # phi_concrete (concrete expression for Taylor derivatives at
        # the saddle) + specializations (auto-derived derivative subs)
        if phi_spec is not None:
            self._phi_concrete = make_phi_concrete_lambda(
                phi_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
            )
            self._specializations = make_specializations_lambda(
                phi_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
            )

            # Register the transfer function as a FORMAL indexed entry
            # in model['functions'] so FieldTheory's auto-Taylor pass
            # produces phi0_<i>, phi1_<i>, … derivative symbols (which
            # mf_bg_conditions / specializations then substitute).
            from sage.all import function as sr_function
            self._functions = [{
                'name':         phi_spec['name'],
                'indexed':      True,
                'deriv_prefix': phi_spec['name'],
                'latex':        phi_spec.get('latex', phi_spec['name']),
                'description':  phi_spec.get('description',
                                             'transfer function'),
                'expression':   (lambda i, v, _t=phi_spec['name']:
                                 sr_function(f'{_t}_{i+1}')(v)),
            }]

        # Mean-field equations — phi bound in CONCRETE mode here.
        if self._mf_eqs_text:
            self._mf_bg = make_mf_bg_conditions_lambda(
                self._mf_eqs_text, phi_spec,
                field_names  = field_names,
                param_names  = param_names,
                kernel_names = kernel_names,
                functions    = self._function_specs,
                n_pop        = n_pop,
            )
            self._mf_equations = make_mf_equations_lambda(
                self._mf_eqs_text, phi_spec,
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

        # Correlated noises (CGF cumulant terms)
        if self._cgf_terms:
            self._correlated_noises.update(
                make_correlated_noises_block(
                    self._cgf_terms, param_names=param_names))

        # ``mf_substitutions`` for the recurrent weight matrix expansion.
        # The MSR-JD framework expects each indexed='matrix' parameter
        # to have a substitution that produces ``[[w_{i+1}{j+1}] for j]
        # for i]`` (the elementwise SR var grid).  Auto-generate these
        # for every matrix parameter declared in the builder.
        for p in self.parameters:
            if p.matrix:
                wname = p.name
                self._mf_substitutions.append({
                    'name':  wname,
                    'value': (lambda ns, _w=wname: [
                        [SR.var(f'{_w}{i+1}{j+1}') for j in ns.pop]
                        for i in ns.pop
                    ]),
                })

    # ── Build the HAWKES_MODEL dict ───────────────────────────────
    def build(self) -> dict:
        """Emit a HAWKES_MODEL dict.

        Compiles text-based declarations (set_action_text,
        define_function, set_mf_equation, declare_cgf_term, …) to
        lambdas and uses them in place of the corresponding lambda
        hooks.  Validates that all required hooks ended up populated.
        Raises ValueError with a clear message if anything is missing.
        """
        # Compile any text declarations into lambdas.  Done here, not
        # in the setters, so that all decls are available when each
        # lambda's namespace is assembled.
        self._compile_text_declarations()

        missing = []
        if self._action is None:           missing.append('action')
        if self._phi_concrete is None:     missing.append('phi_concrete')
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

        model = {
            'name':            self.name,
            'index_sets':      {'pop': list(range(self.n_populations))},
            'response_fields': [self._field_dict(f) for f in self.response_fields],
            'physical_fields': [self._field_dict(f) for f in self.physical_fields],
            'parameters':      [self._param_dict(p) for p in self.parameters],
            'kernels':         [self._kernel_dict(k) for k in self.kernels],
            'operators':       list(self._operators) or [
                {'name': 'Dt', 'sage_name': 'Dt',
                 'latex_name': r'\partial_t', 'description': 'd/dt'},
            ],
            'functions':       list(self._functions),
            'mf_substitutions': self._mf_substitutions,
            'phi_concrete':    self._phi_concrete,
            'action':          self._action,
            'naming_convention': naming_convention,
        }
        if self._mf_bg is not None:
            model['mf_bg_conditions'] = self._mf_bg
        if self._mf_equations is not None:
            model['mf_equations'] = self._mf_equations
        if self._kernel_ft_image is not None:
            model['kernel_ft_image'] = self._kernel_ft_image
        if self._specializations is not None:
            model['specializations'] = self._specializations
        if self._correlated_noises:
            model['correlated_noises'] = self._correlated_noises
        return model

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _field_dict(f: FieldSpec) -> dict:
        d = {'name': f.name, 'indexed': f.indexed, 'latex': f.latex}
        if f.description:
            d['description'] = f.description
        if f.natural_name:
            d['natural_name'] = f.natural_name
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
        return d

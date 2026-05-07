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


@dataclass
class FieldSpec:
    name: str
    indexed: bool
    latex: str
    description: str = ''


@dataclass
class ParameterSpec:
    name: str
    indexed: bool = False         # True = vector, "matrix" = matrix
    matrix: bool = False
    domain: Optional[str] = None
    default: Any = None
    description: str = ''


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

    # ── Field declarations ─────────────────────────────────────────
    def response_field(self, name: str, indexed: bool = True,
                       latex: str = '', description: str = ''):
        self.response_fields.append(FieldSpec(
            name=name, indexed=indexed,
            latex=latex or name, description=description,
        ))
        return self

    def physical_field(self, name: str, indexed: bool = True,
                       latex: str = '', description: str = ''):
        self.physical_fields.append(FieldSpec(
            name=name, indexed=indexed,
            latex=latex or name, description=description,
        ))
        return self

    # ── Parameter declarations ─────────────────────────────────────
    def parameter(self, name: str, default: Any = None,
                  indexed=False, domain: str = None, description: str = ''):
        is_vec = (indexed in (True, 'vector'))
        is_mat = (indexed == 'matrix')
        self.parameters.append(ParameterSpec(
            name=name, indexed=(is_vec or is_mat), matrix=is_mat,
            domain=domain, default=default, description=description,
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
        self.physical_field(
            template.physical_field, indexed=True,
            latex=r'\delta m',
            description='GTaS external rate fluctuation (zero-mean)',
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

    # ── Build the HAWKES_MODEL dict ───────────────────────────────
    def build(self) -> dict:
        """Emit a HAWKES_MODEL dict.

        Validates that all required hooks have been provided.  Raises
        ValueError with a clear message if anything is missing.
        """
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
        return d

    @staticmethod
    def _param_dict(p: ParameterSpec) -> dict:
        d = {'name': p.name, 'indexed': p.indexed}
        if p.domain:
            d['domain'] = p.domain
        if p.description:
            d['description'] = p.description
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

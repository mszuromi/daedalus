"""
Action / kernel / noise TEMPLATES for declarative theory building.

These objects know how to emit the Sage lambdas (action, mf_bg_conditions,
mf_equations, kernel_ft_image, phi_concrete, specializations,
mf_substitutions) that the FieldTheory expander expects.  The user
declares fields + parameters via ``TheoryBuilder``, picks a template,
and gets a fully-formed model dict via ``.build()``.

Templates currently provided:

  * ``HawkesAction(phi='linear' | 'linear_unit_gain' | 'quadratic',
                   gain_param='a', synaptic_kernel='g',
                   recurrent_weight='w', external_drive='E',
                   membrane_timescale='tau',
                   response_spike='nt', response_voltage='vt',
                   physical_spike='dn', physical_voltage='dv')``

      The canonical 2-pop Hawkes-style action used by every model in
      ``simulations/hawkes_*.py``.  Generates action, mf_bg_conditions,
      mf_equations, specializations, mf_substitutions, and the
      per-population ``phi`` function.

  * ``ExpSynapticKernel(kernel_name='g', timescale='tau_g')``

      Generates ``kernel_ft_image``: g → 1 / (1 + i ω τ_g).

  * ``DeltaKernel(kernel_name='g')``

      No frequency image needed; the kernel is treated as δ(t) at
      cell-8 propagator-construction time via ns.delta_D wrapping.

  * ``GTaSNoise(noise_name='X', physical_field='dm', response_field='mt',
                feedforward_weight='w_X', mother_rate='lambda_X',
                participation='p_part', mu_diff='mu_shift_diff',
                sigma_diff_sq='sigma_shift_diff_sq',
                background_param='mstar')``

      Adds the dm/mt fields, the GTaS feedforward action term, the
      mt·dm MSR pairing, the saddle shift in mf_bg_conditions, the
      ``mstar = b_X`` MF equation, and the ``correlated_noises['X']``
      block with κ⁽²⁾ (cross + auto) and the κ⁽³⁾ / κ⁽⁴⁾ Poisson auto.

To add a new model variant: subclass the templates and override only
the methods that change.  For genuinely new structure (different
action shape, alternative noise model), drop down to
``TheoryBuilder.set_action(...)`` and supply the lambda directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sage.all import (
    SR, exp, function, dirac_delta, sqrt, pi, I,
)


def _gauss(tau, mu, sigma_sq):
    """Unit-area Gaussian density.  Used by GTaSNoise's cross-cumulant
    kernel."""
    return exp(-(tau - mu)**2 / (2 * sigma_sq)) / sqrt(2 * pi * sigma_sq)


# ───────────────────────────────────────────────────────────────────────
# HawkesAction
# ───────────────────────────────────────────────────────────────────────

@dataclass
class HawkesAction:
    """Generate the canonical Hawkes 2-pop action and its companion
    lambdas (mf_bg_conditions, mf_equations, specializations,
    phi_concrete, mf_substitutions).

    Parameters
    ----------
    phi : str
        ``'linear'``         — phi(v) = a · v   (gain `a` declared as parameter)
        ``'linear_unit_gain'`` — phi(v) = v       (no gain)
        ``'quadratic'``      — phi(v) = a · v²
    gain_param : str
        Name of the gain parameter (``'a'`` by default).  Ignored when
        ``phi == 'linear_unit_gain'``.
    synaptic_kernel : str
        Name of the synaptic kernel symbol (``'g'`` by default).
    recurrent_weight : str
        Name of the synaptic-weight matrix parameter (``'w'``).
    external_drive : str
        Name of the constant external drive parameter (``'E'``).
    membrane_timescale : str
        Name of the membrane time-constant parameter (``'tau'``).
    response_spike, response_voltage : str
        Names of the cortical response fields.  Defaults
        ``'nt'``, ``'vt'``.
    physical_spike, physical_voltage : str
        Names of the cortical physical-fluctuation fields.  Defaults
        ``'dn'``, ``'dv'``.
    """
    phi: str = 'linear'
    gain_param: str = 'a'
    synaptic_kernel: str = 'g'
    recurrent_weight: str = 'w'
    external_drive: str = 'E'
    membrane_timescale: str = 'tau'
    response_spike: str = 'nt'
    response_voltage: str = 'vt'
    physical_spike: str = 'dn'
    physical_voltage: str = 'dv'

    # The optional GTaS noise add-on (set by TheoryBuilder when the user
    # chains .add_correlated_noise(GTaSNoise(...)) ).  None = no GTaS.
    _gtas: Optional['GTaSNoise'] = None

    # ---- Validation ---------------------------------------------------
    def __post_init__(self):
        if self.phi not in ('linear', 'linear_unit_gain', 'quadratic'):
            raise ValueError(
                f"HawkesAction.phi must be 'linear', "
                f"'linear_unit_gain', or 'quadratic'; got {self.phi!r}"
            )

    # ---- phi_concrete -------------------------------------------------
    def phi_concrete(self):
        """Returns the lambda for ``model['phi_concrete']``."""
        if self.phi == 'linear_unit_gain':
            return lambda ns, i, v: v
        elif self.phi == 'linear':
            return lambda ns, i, v, _g=self.gain_param: SR.var(_g) * v
        else:  # quadratic
            return lambda ns, i, v, _g=self.gain_param: SR.var(_g) * v**2

    # ---- specializations dict for FieldTheory -------------------------
    def specializations(self):
        """Returns the lambda for ``model['specializations']``.

        Sets phi_k_i symbols (the formal-derivative names introduced
        by FieldTheory's auto-Taylor pass) to their concrete values
        at the saddle.
        """
        phi = self.phi
        gain = self.gain_param

        if phi == 'linear_unit_gain':
            return lambda ns: {
                # phi(v) = v  ⇒  phi'(v) = 1
                **{SR.var(f'phi1_{i+1}'): SR(1) for i in ns.pop},
                **{SR.var(f'phi{k}_{i+1}'): SR(0)
                   for k in range(2, ns._taylor_order + 1)
                   for i in ns.pop},
            }
        elif phi == 'linear':
            return lambda ns, _g=gain: {
                **{SR.var(f'phi1_{i+1}'): SR.var(_g) for i in ns.pop},
                **{SR.var(f'phi{k}_{i+1}'): SR(0)
                   for k in range(2, ns._taylor_order + 1)
                   for i in ns.pop},
            }
        else:  # quadratic
            return lambda ns, _g=gain: {
                **{SR.var(f'phi1_{i+1}'): 2 * SR.var(_g) * ns.vstar[i]
                   for i in ns.pop},
                **{SR.var(f'phi2_{i+1}'): 2 * SR.var(_g) for i in ns.pop},
                **{SR.var(f'phi{k}_{i+1}'): SR(0)
                   for k in range(3, ns._taylor_order + 1)
                   for i in ns.pop},
            }

    # ---- mf_substitutions ---------------------------------------------
    def mf_substitutions(self):
        """The standard w-matrix expansion + ndot_bg substitution."""
        wname = self.recurrent_weight
        return [
            {
                'name':  wname,
                'value': lambda ns, _w=wname: [
                    [SR.var(f'{_w}{i+1}{j+1}') for j in ns.pop]
                    for i in ns.pop
                ],
            },
            {
                'name':  'ndot_bg',
                'value': lambda ns: [+ns.nstar[i] for i in ns.pop],
            },
        ]

    # ---- mf_bg_conditions ---------------------------------------------
    def mf_bg_conditions(self):
        """Returns the lambda for ``model['mf_bg_conditions']``.

        v*_i  = E_i + Σ_j w_{ij}·g·n*_j  +  (GTaS feedforward if any)
        phi0_i = n*_i  (constant phi value at saddle)
        """
        gtas = self._gtas

        def _bg(ns):
            base = {SR.var(f'phi0_{i+1}'): ns.nstar[i] for i in ns.pop}
            for i in ns.pop:
                vstar_expr = (
                    ns.E[i]
                    + sum(ns.w[i][j] * ns.g * ns.nstar[j] for j in ns.pop)
                )
                if gtas is not None:
                    # GTaS feedforward saddle shift:
                    #   v* += w_X · g · b_X = w_X · g · λ_X · p_part
                    vstar_expr = vstar_expr + (
                        getattr(ns, gtas.feedforward_weight)
                        * ns.g
                        * getattr(ns, gtas.mother_rate)
                        * getattr(ns, gtas.participation)
                    )
                base[ns.vstar[i]] = vstar_expr
            return base

        return _bg

    # ---- mf_equations -------------------------------------------------
    def mf_equations(self):
        """Returns the lambda for ``model['mf_equations']``.

        Linear:        n*_i  =  a · (E_i + Σ w_{ij}·n*_j + w_X·b_X)
        Linear unit:   n*_i  =  E_i + Σ w_{ij}·n*_j + w_X·b_X
        Quadratic:     n*_i  =  a · (E_i + Σ w_{ij}·n*_j + w_X·b_X)²
        """
        phi = self.phi
        gain = self.gain_param
        gtas = self._gtas

        def _eq(ns):
            cortical_equations = []
            for i in ns.pop:
                base = (
                    ns.E[i]
                    + sum(ns.w[i][j] * ns.nstar[j] for j in ns.pop)
                )
                if gtas is not None:
                    base = base + (
                        getattr(ns, gtas.feedforward_weight)
                        * getattr(ns, gtas.mother_rate)
                        * getattr(ns, gtas.participation)
                    )
                if phi == 'linear_unit_gain':
                    rhs = base
                elif phi == 'linear':
                    rhs = SR.var(gain) * base
                else:  # quadratic
                    rhs = SR.var(gain) * base**2
                cortical_equations.append(ns.nstar[i] == rhs)

            extra = []
            if gtas is not None and gtas.background_param is not None:
                # m*_i = b_X = lambda_X · p_part (per-cell marginal rate)
                mstar = getattr(ns, gtas.background_param)
                for i in ns.pop:
                    extra.append(mstar[i] == (
                        getattr(ns, gtas.mother_rate)
                        * getattr(ns, gtas.participation)
                    ))
            return cortical_equations + extra

        return _eq

    # ---- action -------------------------------------------------------
    def action(self):
        """Returns the lambda for ``model['action']``.

        Standard MSR-JD Hawkes action; the GTaS feedforward and MSR
        pairing terms are appended when ``_gtas`` is set.
        """
        gtas = self._gtas
        nt_name = self.response_spike
        vt_name = self.response_voltage
        dn_name = self.physical_spike
        dv_name = self.physical_voltage

        def _action(ns, _gtas=gtas, _nt=nt_name, _vt=vt_name,
                    _dn=dn_name, _dv=dv_name):
            nt = getattr(ns, _nt)
            vt = getattr(ns, _vt)
            dn = getattr(ns, _dn)
            dv = getattr(ns, _dv)

            terms = SR(0)
            for i in ns.pop:
                # Cortical Poisson part
                t = (
                    nt[i] * (ns.ndot_bg[i] + dn[i])
                    - (exp(nt[i]) - 1) * ns.phi(i, dv[i])
                )
                # Cortical voltage equation
                voltage = (
                    (ns.tau * ns.Dt + 1) * dv[i]
                    + ns.vstar[i] - ns.E[i]
                    - sum(ns.w[i][j] * ns.g * (ns.nstar[j] + dn[j])
                          for j in ns.pop)
                )
                if _gtas is not None:
                    # − w_X · g · (λ p + dm[i])
                    dm = getattr(ns, _gtas.physical_field)
                    voltage = voltage - (
                        getattr(ns, _gtas.feedforward_weight) * ns.g
                        * (getattr(ns, _gtas.mother_rate)
                           * getattr(ns, _gtas.participation)
                           + dm[i])
                    )
                t = t + vt[i] * voltage

                # GTaS MSR pairing for the external fluctuation field
                if _gtas is not None:
                    mt = getattr(ns, _gtas.response_field)
                    dm = getattr(ns, _gtas.physical_field)
                    t = t + mt[i] * dm[i]

                terms = terms + t
            return terms

        return _action

    # ---- functions list -----------------------------------------------
    def functions_list(self):
        """phi_i(v) declared as a FORMAL indexed function symbol.

        FieldTheory's auto-expander Taylor-expands the action in dv,
        producing expressions in the FORMAL derivative symbols
        ``phi_{k}_{i+1}`` for k = 0, 1, ..., taylor_order.  These get
        substituted later by:

          * ``mf_bg_conditions``  → phi0_{i+1} = nstar_{i+1}
                                    (the concrete numerical value of
                                    phi at the saddle)
          * ``specializations``   → phi1, phi2 = derivatives of the
                                    concrete phi at the saddle

        Using the concrete expression here would short-circuit this
        machinery (the auto-expander sees phi(0)=0 etc. as numeric
        and never registers symbols), leaving an uncancelled (1,0)
        tadpole in the bigrade analysis.
        """
        return [{
            'name':         'phi',
            'indexed':      True,
            'deriv_prefix': 'phi',
            'latex':        r'\varphi',
            'description':  f'transfer function ({self.phi})',
            'expression':   lambda i, v: function(f'phi_{i+1}')(v),
        }]


# ───────────────────────────────────────────────────────────────────────
# Synaptic kernel templates
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ExpSynapticKernel:
    """Exponential synaptic filter g(t) = (1/τ_g)·e^(-t/τ_g)·Θ(t).

    Frequency image: ĝ(ω) = 1 / (1 + i ω τ_g).
    """
    kernel_name: str = 'g'
    timescale_param: str = 'tau_g'

    def kernel_ft_image(self):
        ts = self.timescale_param
        return lambda ns, omega, _ts=ts: {
            ns.g: SR(1) / (SR(1) + I * omega * SR.var(_ts)),
        }

    def extra_specializations(self):
        """No extra specializations — the kernel symbol is replaced by
        its frequency image post-FT (in cell 8 / build_propagator)."""
        return None


@dataclass
class DeltaKernel:
    """Instantaneous synaptic kernel: g(t) = δ(t).

    No frequency image needed; instead, the kernel symbol is mapped
    to ``ns.delta_D`` (the FieldTheory's δ-distribution placeholder)
    in ``specializations``, so cell 8's `_to_kernel` rewriter sees
    it as a constant in time domain and Fourier-transforms it
    correctly to 1.
    """
    kernel_name: str = 'g'

    def kernel_ft_image(self):
        return None

    def extra_specializations(self):
        """Substitute ``ns.g`` → ``ns.delta_D`` so the kernel acts as
        δ(t) in the time-domain propagator construction.  The
        TheoryBuilder merges this into the HawkesAction template's
        specializations dict at build time."""
        return lambda ns: {ns.g: ns.delta_D}


# ───────────────────────────────────────────────────────────────────────
# GTaS noise template
# ───────────────────────────────────────────────────────────────────────

@dataclass
class GTaSNoise:
    """Bernoulli + Gaussian GTaS external rate process.

    Parameters
    ----------
    noise_name : str
        Internal label for the noise process.  Default ``'X'``.
    physical_field, response_field : str
        Field names for the external fluctuation (``'dm'``) and its
        MSR conjugate (``'mt'``).  TheoryBuilder declares these.
    feedforward_weight : str
        Parameter name for the one-to-one feedforward weight w_X.
    mother_rate, participation : str
        Parameter names for the GTaS mother Poisson rate λ_X and
        per-cell Bernoulli participation p_part.
    mu_diff, sigma_diff_sq : str
        Parameter names for the Gaussian shift-difference mean and
        variance — used only by the cross-cumulant kernel.  Set to
        ``None`` to omit cross-correlations entirely (independent
        Bernoulli per cell).
    background_param : str or None
        Parameter name for the per-cell marginal mean rate ``b_X``.
        Default ``'mstar'``.  Set to ``None`` to skip the m*=b_X
        MF equation.
    """
    noise_name: str = 'X'
    physical_field: str = 'dm'
    response_field: str = 'mt'
    feedforward_weight: str = 'w_X'
    mother_rate: str = 'lambda_X'
    participation: str = 'p_part'
    mu_diff: Optional[str] = 'mu_shift_diff'
    sigma_diff_sq: Optional[str] = 'sigma_shift_diff_sq'
    background_param: Optional[str] = 'mstar'

    def correlated_noises_block(self):
        """Returns the dict to slot into ``model['correlated_noises']``.

        Includes:
          κ⁽²⁾ — Poisson auto + Bernoulli+Gaussian cross
          κ⁽³⁾ — Poisson auto only (mixed indices vanish for N=2)
          κ⁽⁴⁾ — Poisson auto only
        """
        nname = self.noise_name
        pname = self.physical_field
        rname = self.response_field
        mu_d  = self.mu_diff
        si_d  = self.sigma_diff_sq

        def _kappa2(ns, i, j, tau, _mu=mu_d, _si=si_d):
            if i == j:
                return ns.lambda_X * ns.p_part * dirac_delta(tau)
            if _mu is None or _si is None:
                # Independent across cells if user omitted shift params
                return SR(0)
            return (
                ns.lambda_X * ns.p_part**2
                * _gauss(tau, getattr(ns, _mu), getattr(ns, _si))
            )

        def _kappa3(ns, i, j, k, t1, t2):
            if i == j == k:
                return (
                    ns.lambda_X * ns.p_part
                    * dirac_delta(t1) * dirac_delta(t2)
                )
            return SR(0)

        def _kappa4(ns, i, j, k, l, t1, t2, t3):
            if i == j == k == l:
                return (
                    ns.lambda_X * ns.p_part
                    * dirac_delta(t1) * dirac_delta(t2) * dirac_delta(t3)
                )
            return SR(0)

        return {
            nname: {
                'physical_field': pname,
                'response_field': rname,
                'mean': lambda ns: ns.lambda_X * ns.p_part,
                'cumulants': {
                    2: _kappa2,
                    3: _kappa3,
                    4: _kappa4,
                },
            },
        }

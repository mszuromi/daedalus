"""
models/hawkes_sage.py
=====================
Model specification for the nonlinear Hawkes process (SageMath version).

Action (before MF expansion):
  S = sum_i { ñ_i ṅ_i  +  (e^{ñ_i} - 1) φ_i(v_i)
            + ṽ_i [(τ ∂_t + 1) v_i  -  E_i  -  Σ_j w_{ij} (g ∗ ṅ_j)] }

MF expansion:
  ṅ_i = ṅ_i* + δṅ_i   (physical: expanded around background)
  v_i = v_i* + δv_i
  ñ_i, ṽ_i             (response: full integration variables, NOT expanded)

The action lambda uses:
  - exp(ns.nt[i]) - 1   — SageMath's exp(), Taylor-expanded automatically
  - ns.phi(i, ns.dv[i]) — formal SageMath function phi_{i+1}(dv_i),
                           Taylor-expanded automatically with derivative
                           symbols phi0_{i+1}, phi1_{i+1}, phi2_{i+1}, ...

MF saddle conditions applied via 'mf_bg_conditions':
  phi0_i  →  nstar_i          (Poisson saddle: φ_i(v_i*) = nstar_i)
  vstar_i →  E_i + Σ_j w_ij g nstar_j   (voltage background EOM)
"""

from sage.all import SR, exp, function

# ---------------------------------------------------------------------------
# Population count
# ---------------------------------------------------------------------------
N_POP = 2
_pop  = list(range(N_POP))

# ---------------------------------------------------------------------------
# SR parameter symbols
# ---------------------------------------------------------------------------
_nstar = [SR.var(f'nstar{i+1}', domain='positive') for i in _pop]
_w     = [[SR.var(f'w{i+1}{j+1}') for j in _pop] for i in _pop]
_g     = SR.var('g')

# ---------------------------------------------------------------------------
# Model dict
# ---------------------------------------------------------------------------
HAWKES_MODEL: dict = {

    'name': 'Nonlinear Hawkes 2-population',

    # -----------------------------------------------------------------------
    # Index sets
    # -----------------------------------------------------------------------
    'index_sets': {'pop': _pop},

    # -----------------------------------------------------------------------
    # Response fields  (NOT expanded — full integration variables)
    # -----------------------------------------------------------------------
    'response_fields': [
        {'name': 'nt', 'indexed': True, 'latex': r'\tilde{n}',
         'description': 'response field conjugate to spike train'},
        {'name': 'vt', 'indexed': True, 'latex': r'\tilde{v}',
         'description': 'response field conjugate to voltage'},
    ],

    # -----------------------------------------------------------------------
    # Physical fluctuation fields  (expanded around MF background)
    # -----------------------------------------------------------------------
    'physical_fields': [
        {'name': 'dn', 'indexed': True, 'latex': r'\delta\dot{n}',
         'description': 'spike-train fluctuation around MF background'},
        {'name': 'dv', 'indexed': True, 'latex': r'\delta v',
         'description': 'voltage fluctuation around MF background'},
    ],

    # -----------------------------------------------------------------------
    # Scalar parameters  (SR symbols — coefficients of the polynomial ring)
    # -----------------------------------------------------------------------
    'parameters': [
        {'name': 'nstar', 'indexed': True,  'domain': 'positive',
         'description': 'background firing rate  nstar_i = phi_i(vstar_i)'},
        {'name': 'vstar', 'indexed': True,
         'description': 'background voltage'},
        {'name': 'E',     'indexed': True,
         'description': 'external drive'},
        {'name': 'tau',   'indexed': False, 'domain': 'positive',
         'description': 'membrane time constant'},
        {'name': 'a',     'indexed': True,  'domain': 'positive',
         'description': 'quadratic gain coefficient: phi_i(v) = (a_i/2) v^2'},
    ],

    # -----------------------------------------------------------------------
    # Kernels
    # sage_name 'z_g' sorts after 'w' so products render as w_{ij} g, not g w_{ij}
    # -----------------------------------------------------------------------
    'kernels': [
        {'name': 'g', 'sage_name': 'z_g', 'latex_name': 'g',
         'description': 'synaptic filter kernel g(t)'},
    ],

    # -----------------------------------------------------------------------
    # Operators
    # -----------------------------------------------------------------------
    'operators': [
        {'name': 'Dt', 'sage_name': 'Dt', 'latex_name': r'\partial_t',
         'description': r'∂_t  (algebraic placeholder for the time-derivative operator)'},
    ],

    # -----------------------------------------------------------------------
    # Nonlinear functions
    #
    # 'expression': lambda i, v -> SR expression in v
    #   Here phi_i is specified as a formal SageMath function symbol.
    #   The framework Taylor-expands phi_{i+1}(dv_i) around dv_i = 0 and
    #   renames derivatives using 'latex' as the base symbol:
    #     phi_{i+1}(0)             →  phi0_{i+1}   displayed as  φ_{i}
    #     D[0](phi_{i+1})(0)       →  phi1_{i+1}   displayed as  φ'_{i}
    #     D[0,0](phi_{i+1})(0)     →  phi2_{i+1}   displayed as  φ''_{i}  ...
    # -----------------------------------------------------------------------
    'functions': [
        {
            'name':         'phi',
            'indexed':      True,
            'deriv_prefix': 'phi',
            'latex':        r'\varphi',
            'description':  'nonlinear gain function φ_i(δv_i)',
            'expression':   lambda i, v: function(f'phi_{i+1}')(v),
        },
    ],

    # -----------------------------------------------------------------------
    # Namespace-level substitutions  (run once at namespace build)
    # -----------------------------------------------------------------------
    'mf_substitutions': [
        {
            'name':  'w',
            'value': lambda ns: [[SR.var(f'w{i+1}{j+1}') for j in ns.pop]
                                 for i in ns.pop],
        },
        {
            'name':  'ndot_bg',
            # MF Poisson saddle:  ṅ_i* = -φ_i(v_i*) = -nstar_i
            'value': lambda ns: [-ns.nstar[i] for i in ns.pop],
        },
    ],

    # -----------------------------------------------------------------------
    # MF background conditions  (SR substitutions applied after expansion)
    #
    # 1. Poisson saddle:  phi0_i ≡ phi_i(v_i*) = nstar_i
    # 2. Voltage EOM:     vstar_i = E_i + Σ_j w_{ij} g nstar_j
    # -----------------------------------------------------------------------
    'mf_bg_conditions': lambda ns: {
        **{SR.var(f'phi0_{i+1}'): ns.nstar[i]        for i in ns.pop},
        **{ns.vstar[i]: ns.E[i] + sum(ns.w[i][j] * ns.g * ns.nstar[j]
                                      for j in ns.pop)
           for i in ns.pop},
    },

    # -----------------------------------------------------------------------
    # Specializations  (SR substitutions applied after MF background conditions)
    #
    # phi quadratic:  φ_i(v) = nstar_i + φ1_i v + φ2_i/2 v²
    #   → cubic and higher Taylor coefficients are zero
    # g = δ(t):  instantaneous synaptic coupling
    # -----------------------------------------------------------------------
    'specializations': lambda ns: {
        **{SR.var(f'phi{k}_{i+1}'): SR(0)
           for k in range(3, ns._taylor_order + 1)
           for i in ns.pop},
        ns.g: ns.delta_D,
    },

    # -----------------------------------------------------------------------
    # Concrete transfer function  (for numerical evaluation)
    #
    # phi_i(v) = (a_i / 2) v^2      (pure quadratic, no linear/constant)
    # The notebook differentiates this symbolically to the required Taylor
    # order — derivatives are NEVER hard-coded here.
    # -----------------------------------------------------------------------
    'phi_concrete': lambda ns, i, v: (ns.a[i] / SR(2)) * v**2,

    # -----------------------------------------------------------------------
    # Mean-field self-consistency equations  (may require numerical solution)
    #
    # v*_i  = E_i + Σ_j w_{ij} n*_j       (voltage background EOM, g = δ)
    # n*_i  = φ_i(v*_i)                    (Poisson saddle condition)
    #
    # Combined: n*_i = (a_i / 2) (E_i + Σ_j w_{ij} n*_j)^2
    # -----------------------------------------------------------------------
    'mf_equations': lambda ns: [
        ns.nstar[i] == (ns.a[i] / SR(2)) * (
            ns.E[i] + sum(ns.w[i][j] * ns.nstar[j] for j in ns.pop)
        )**2
        for i in ns.pop
    ],

    # -----------------------------------------------------------------------
    # Background rate convention
    # -----------------------------------------------------------------------
    'background_rate_convention': (
        'ṅ_i* = -φ_i(v_i*)  '
        '[positive Poisson term:  +ñ_i ṅ_i + (e^{ñ_i}-1) φ_i]'
    ),

    # -----------------------------------------------------------------------
    # Action  (callable; receives namespace ns; returns SR expression)
    #
    # exp(ns.nt[i]) and ns.phi(i, ns.dv[i]) are nonlinear in the field
    # variables — the framework Taylor-expands them automatically.
    # -----------------------------------------------------------------------
    'action': lambda ns: sum(
        # S1: ñ_i (ṅ_i* + δṅ_i)
        ns.nt[i] * (ns.ndot_bg[i] + ns.dn[i])
        # S2: (e^{ñ_i} - 1) φ_i(δv_i)      ← exp and phi auto-expanded
        + (exp(ns.nt[i]) - 1) * ns.phi(i, ns.dv[i])
        # S3: ṽ_i [(τ ∂_t + 1) δv_i  +  v_i* - E_i  -  Σ_j w_{ij} g (n*_j + δṅ_j)]
        + ns.vt[i] * (
            (ns.tau * ns.Dt + 1) * ns.dv[i]
            + ns.vstar[i] - ns.E[i]
            - sum(ns.w[i][j] * ns.g * (ns.nstar[j] + ns.dn[j])
                  for j in ns.pop)
        )
        for i in ns.pop
    ),
}

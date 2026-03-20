"""
models/hawkes.py
================
Model specification dict for a nonlinear Hawkes process
with N_pop=2 interacting populations, embedded in the MSRJD path integral.

Action (before MF expansion):
  S = sum_i { ñ_i ṅ_i  +  (e^{ñ_i} - 1) φ_i(v_i)
            + ṽ_i [(τ ∂_t + 1) v_i - E_i - Σ_j w_{ij} (g ∗ ṅ_j)] }

MF expansion:
  ṅ_i = ṅ_i*  + δṅ_i      (physical field expanded around background)
  v_i = v_i*  + δv_i
  ñ_i, ṽ_i unchanged       (response fields — full integration variables)

MF saddle condition:
  δS/δñ_i |_{bg} = 0  ⟹  ṅ_i* + φ_i(v_i*) = 0  ⟹  ṅ_i* = -n_i*
"""

import sympy as sp
from sympy import symbols

# ---------------------------------------------------------------------------
# Population count
# ---------------------------------------------------------------------------
N_POP = 2
_pop  = list(range(N_POP))

# ---------------------------------------------------------------------------
# Background symbols
# ---------------------------------------------------------------------------
_nstar = [symbols(f'nstar{i+1}', positive=True) for i in _pop]  # background rate
_vstar = [symbols(f'vstar{i+1}')                for i in _pop]  # background voltage
_E     = [symbols(f'E{i+1}')                    for i in _pop]  # external drive
_w     = [[symbols(f'w{i+1}{j+1}') for j in _pop] for i in _pop]
_g     = symbols('g')    # synaptic kernel symbol

# ---------------------------------------------------------------------------
# Taylor coefficients for φ_i (nonlinear gain function)
# phi_k_i = φ_i^(k)(v_i*)  — evaluated at background voltage
# phi0_i ≡ nstar_i  by the MF Poisson saddle condition φ_i(v_i*) = nstar_i
# ---------------------------------------------------------------------------
_phi1 = [symbols(f'phi1_{i+1}') for i in _pop]
_phi2 = [symbols(f'phi2_{i+1}') for i in _pop]
_phi3 = [symbols(f'phi3_{i+1}') for i in _pop]
_phi4 = [symbols(f'phi4_{i+1}') for i in _pop]

# ---------------------------------------------------------------------------
# Model dict
# ---------------------------------------------------------------------------
HAWKES_MODEL: dict = {

    'name': 'Nonlinear Hawkes 2-population',

    # -----------------------------------------------------------------------
    # Index sets
    # -----------------------------------------------------------------------
    'index_sets': {
        'pop': _pop,          # [0, 1]
    },

    # -----------------------------------------------------------------------
    # Physical fluctuation fields  (expanded:  field = background + fluct)
    # -----------------------------------------------------------------------
    'physical_fields': [
        {
            'name':    'dn',          # δṅ_i  — spike-train fluctuation
            'indexed': True,
            'latex':   r'\delta\dot{n}',
            'description': 'spike-train fluctuation around MF background',
        },
        {
            'name':    'dv',          # δv_i  — voltage fluctuation
            'indexed': True,
            'latex':   r'\delta v',
            'description': 'membrane-voltage fluctuation around MF background',
        },
    ],

    # -----------------------------------------------------------------------
    # Response fields  (NOT expanded — full integration variables)
    # -----------------------------------------------------------------------
    'response_fields': [
        {
            'name':    'nt',          # ñ_i
            'indexed': True,
            'latex':   r'\tilde{n}',
            'description': 'response field conjugate to spike train',
        },
        {
            'name':    'vt',          # ṽ_i
            'indexed': True,
            'latex':   r'\tilde{v}',
            'description': 'response field conjugate to voltage',
        },
    ],

    # -----------------------------------------------------------------------
    # Scalar parameters  (appear as SR coefficients in ring)
    # -----------------------------------------------------------------------
    'parameters': [
        {'name': 'nstar', 'indexed': True, 'positive': True,
         'latex': r'n^*',
         'description': 'background firing rate (= phi0_i from saddle)'},
        {'name': 'vstar', 'indexed': True,
         'latex': r'v^*',
         'description': 'background voltage'},
        {'name': 'E',     'indexed': True,
         'latex': r'E',
         'description': 'external drive / reversal potential'},
        {'name': 'tau',   'indexed': False, 'positive': True,
         'latex': r'\tau',
         'description': 'membrane time constant'},
        {'name': 'w',
         # w is a 2D matrix — we handle it manually via lambda below
         'indexed': False,          # suppress auto-creation
         'latex': r'w',
         'description': 'synaptic weight matrix'},
    ],

    # -----------------------------------------------------------------------
    # Convolution kernels  (algebraic placeholders)
    # -----------------------------------------------------------------------
    'kernels': [
        {
            'name':  'g',
            'latex': 'g',
            'description': 'synaptic filter kernel g(t)',
        },
    ],

    # -----------------------------------------------------------------------
    # Differential operators  (algebraic placeholders for sector counting)
    # -----------------------------------------------------------------------
    'operators': [
        {
            'name':  'Dt',
            'latex': 'Dt',
            'description': r'∂_t  (algebraic stand-in; display as delta_Dp)',
        },
    ],

    # -----------------------------------------------------------------------
    # Nonlinear functions with Taylor expansion info
    # taylor_coeffs[i][k] = phi_i^(k)(v_i*)  (SymPy expr)
    # -----------------------------------------------------------------------
    'functions': [
        {
            'name':    'phi',
            'indexed': True,
            'description': 'nonlinear gain function φ_i(v_i* + δv_i)',
            # coeffs[population_index][taylor_order_k]
            # phi0_i = nstar_i enforced by MF Poisson saddle (φ_i(v_i*) = nstar_i)
            'taylor_coeffs': [
                [_nstar[i], _phi1[i], _phi2[i], _phi3[i], _phi4[i]]
                for i in _pop
            ],
        },
    ],

    # -----------------------------------------------------------------------
    # Mean-field substitutions  (run after symbols are built)
    # These callables receive the namespace ns and return SymPy expressions.
    # -----------------------------------------------------------------------
    'mf_substitutions': [
        {
            'name': 'w',
            # w is a 2×2 matrix of symbols
            'value': lambda ns: [
                [symbols(f'w{i+1}{j+1}') for j in ns.pop]
                for i in ns.pop
            ],
            'description': 'synaptic weight matrix w_{ij}',
        },
        {
            'name': 'ndot_bg',
            # MF saddle: δS/δñ_i = ṅ_i* + φ_i(v_i*) = 0  ⟹  ṅ_i* = -φ_i(v_i*)
            # φ_i(v_i*) = phi0_i, so ṅ_i* = -phi0_i = -nstar_i
            'value': lambda ns: [-ns.nstar[i] for i in ns.pop],
            'description': 'background spike rate from MF saddle condition',
        },
    ],

    # -----------------------------------------------------------------------
    # MF background conditions  (substitutions applied after action expansion)
    # These enforce the MF saddle-point equations symbolically:
    #   (1) phi0_i = nstar_i  already enforced via Taylor coefficients above
    #   (2) vstar_i = E_i + Σ_j w_ij * g * nstar_j  (voltage background EOM)
    # -----------------------------------------------------------------------
    'mf_bg_conditions': lambda ns: {
        ns.vstar[i]: ns.E[i] + sum(ns.w[i][j] * ns.g * ns.nstar[j]
                                   for j in ns.pop)
        for i in ns.pop
    },

    # -----------------------------------------------------------------------
    # Background field convention
    # -----------------------------------------------------------------------
    'background_rate_convention': (
        'ṅ_i* = -φ_i(v_i*)  [positive Poisson term sign: +ñ_i ṅ_i + (e^ñ-1)φ]'
    ),

    # -----------------------------------------------------------------------
    # Action  (callable receiving the namespace ns)
    # -----------------------------------------------------------------------
    'action': lambda ns: sum(
        # S1: ñ_i ṅ_i  with  ṅ_i = ṅ_i* + δṅ_i
        ns.nt[i] * (ns.ndot_bg[i] + ns.dn[i])
        # S2: (e^{ñ_i} - 1) φ_i(v_i* + δv_i)
        + ns.exp_m1(ns.nt[i]) * ns.phi(i, ns.dv[i])
        # S3: ṽ_i [(τ ∂_t + 1) δv_i  +  v_i* - E_i  -  Σ_j w_{ij} g * (n_j* + δṅ_j)]
        + ns.vt[i] * (
            (ns.tau * ns.Dt + 1) * ns.dv[i]          # voltage operator on fluctuation
            + ns.vstar[i] - ns.E[i]                   # background voltage residual
            - sum(ns.w[i][j] * ns.g * (ns.nstar[j] + ns.dn[j])
                  for j in ns.pop)                     # synaptic input
        )
        for i in ns.pop
    ),
}

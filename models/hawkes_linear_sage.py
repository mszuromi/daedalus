"""
models/hawkes_linear_sage.py
============================
Model specification for the LINEAR Hawkes process (SageMath version).

Same MSR-JD action as the nonlinear version, but with linear transfer
function φ_i(v) = v.  This means:
  - φ'_i = 1  (constant gain, independent of operating point)
  - φ''_i = 0  (no nonlinear interaction vertices from phi)
  - Loop diagrams arise ONLY from the exp(ñ) Poisson nonlinearity.

Action (before MF expansion):
  S = sum_i { ñ_i ṅ_i  -  (e^{ñ_i} - 1) φ_i(v_i)
            + ṽ_i [(τ ∂_t + 1) v_i  -  E_i  -  Σ_j w_{ij} (g * ṅ_j)] }

with φ_i(v) = v.

MF: n*_i = v*_i = E_i + Σ_j w_{ij} n*_j  →  n* = (I - W)^{-1} E
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

    'name': 'Linear Hawkes 2-population',

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
    # Scalar parameters  (SR symbols)
    # No 'a' parameter — phi is fixed as the identity.
    # -----------------------------------------------------------------------
    'parameters': [
        {'name': 'nstar', 'indexed': True,  'domain': 'positive',
         'description': 'background firing rate  nstar_i = phi_i(vstar_i) = vstar_i'},
        {'name': 'vstar', 'indexed': True,
         'description': 'background voltage'},
        {'name': 'E',     'indexed': True,
         'description': 'external drive'},
        {'name': 'tau',   'indexed': False, 'domain': 'positive',
         'description': 'membrane time constant'},
    ],

    # -----------------------------------------------------------------------
    # Kernels
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
    # phi_i(v) = v  (identity / linear transfer function)
    # The framework still Taylor-expands this formally, producing:
    #   phi0_i = phi_i(0) = 0  →  substituted to nstar_i by MF conditions
    #   phi1_i = 1
    #   phi2_i = phi3_i = ... = 0
    # -----------------------------------------------------------------------
    'functions': [
        {
            'name':         'phi',
            'indexed':      True,
            'deriv_prefix': 'phi',
            'latex':        r'\varphi',
            'description':  'linear gain function φ_i(v) = v',
            'expression':   lambda i, v: function(f'phi_{i+1}')(v),
        },
    ],

    # -----------------------------------------------------------------------
    # Namespace-level substitutions
    # -----------------------------------------------------------------------
    'mf_substitutions': [
        {
            'name':  'w',
            'value': lambda ns: [[SR.var(f'w{i+1}{j+1}') for j in ns.pop]
                                 for i in ns.pop],
        },
        {
            'name':  'ndot_bg',
            # MF Poisson saddle:  ṅ_i* = +φ_i(v_i*) = +nstar_i
            'value': lambda ns: [+ns.nstar[i] for i in ns.pop],
        },
    ],

    # -----------------------------------------------------------------------
    # MF background conditions
    # -----------------------------------------------------------------------
    'mf_bg_conditions': lambda ns: {
        **{SR.var(f'phi0_{i+1}'): ns.nstar[i]        for i in ns.pop},
        **{ns.vstar[i]: ns.E[i] + sum(ns.w[i][j] * ns.g * ns.nstar[j]
                                      for j in ns.pop)
           for i in ns.pop},
    },

    # -----------------------------------------------------------------------
    # Specializations
    #
    # phi linear:  φ_i(v) = v  →  phi1_i = 1, phi2_i = phi3_i = ... = 0
    # g = δ(t):  instantaneous synaptic coupling
    # -----------------------------------------------------------------------
    'specializations': lambda ns: {
        # phi1_i = 1 (unit gain)
        **{SR.var(f'phi1_{i+1}'): SR(1) for i in ns.pop},
        # phi2 and all higher derivatives are zero
        **{SR.var(f'phi{k}_{i+1}'): SR(0)
           for k in range(2, ns._taylor_order + 1)
           for i in ns.pop},
        ns.g: ns.delta_D,
    },

    # -----------------------------------------------------------------------
    # Concrete transfer function  (for numerical evaluation)
    # phi_i(v) = v
    # -----------------------------------------------------------------------
    'phi_concrete': lambda ns, i, v: v,

    # -----------------------------------------------------------------------
    # Mean-field self-consistency equations
    # n*_i = v*_i = E_i + Σ_j w_{ij} n*_j   (linear, solvable as n* = (I-W)^{-1} E)
    # -----------------------------------------------------------------------
    'mf_equations': lambda ns: [
        ns.nstar[i] == ns.E[i] + sum(ns.w[i][j] * ns.nstar[j] for j in ns.pop)
        for i in ns.pop
    ],

    # -----------------------------------------------------------------------
    # Background rate convention
    # -----------------------------------------------------------------------
    'background_rate_convention': (
        'ṅ_i* = +φ_i(v_i*) = v_i*  '
        '[negative Poisson term:  +ñ_i ṅ_i - (e^{ñ_i}-1) φ_i]'
    ),

    # -----------------------------------------------------------------------
    # Action
    # -----------------------------------------------------------------------
    'action': lambda ns: sum(
        # S1: ñ_i (ṅ_i* + δṅ_i)
        ns.nt[i] * (ns.ndot_bg[i] + ns.dn[i])
        # S2: -(e^{ñ_i} - 1) φ_i(δv_i)
        - (exp(ns.nt[i]) - 1) * ns.phi(i, ns.dv[i])
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

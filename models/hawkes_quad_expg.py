"""
models/hawkes_quad_expg.py
==========================
**Quadratic** Hawkes 2-population model with gain ``a`` and exponential
synaptic filter ``g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)``.

Transfer function:
    phi_i(v) = a * v^2                (quadratic)

    => phi^{(0)}(v_i*)  =  a v_i*^2  (= n_i* by the Poisson saddle)
       phi^{(1)}(v_i*)  =  2 a v_i*
       phi^{(2)}(v_i*)  =  2 a
       phi^{(k)}(v_i*)  =  0  for  k >= 3

Unlike the linear variants, ``phi^{(2)} != 0``.  This surfaces a
``n~ dv^2`` cubic interaction vertex in the MSR-JD action, which
enables 1-loop diagrams from the quadratic nonlinearity -- distinct
from the `e^{n~}` Poisson-nonlinearity loops that the linear models
already had.  With this model the notebook's ``max_ell = 1``
enumerator will produce a richer set of 1-loop topologies whose
evaluation stress-tests the loop-subgraph integrator.

Synaptic filter (unit integral, same as hawkes_linear_expg):
    g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)
    g_hat(omega) = 1 / (1 + i * omega * tau_g)

Action (before MF expansion):
    S = sum_i { n~_i n_dot_i - (e^{n~_i} - 1) phi_i(v_i)
              + v~_i [(tau d_t + 1) v_i - E_i
                      - sum_j w_{ij} (g * n_dot_j)] }

MF saddle:
    n*_i = phi_i(v*_i) = a * (v*_i)^2
    v*_i = E_i + sum_j w_{ij} * int(g) * n*_j
         = E_i + sum_j w_{ij} * n*_j                (kernel integrates to 1)
Combined (nonlinear, fsolve):
    n*_i = a * ( E_i + sum_j w_{ij} * n*_j )^2
"""

from sage.all import SR, exp, function, I

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

    'name': 'Quadratic Hawkes 2-population (phi = a v^2, exp filter tau_g)',

    # -----------------------------------------------------------------------
    # Index sets
    # -----------------------------------------------------------------------
    'index_sets': {'pop': _pop},

    # -----------------------------------------------------------------------
    # Response fields  (full integration variables)
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
    # Scalar parameters
    #
    #   a      -- gain of the quadratic transfer function phi(v) = a v^2
    #   tau_g  -- exponential synaptic filter timescale
    # -----------------------------------------------------------------------
    'parameters': [
        {'name': 'nstar', 'indexed': True,  'domain': 'positive',
         'description': 'background firing rate  n*_i = phi_i(v*_i) = a * (v*_i)^2'},
        {'name': 'vstar', 'indexed': True,
         'description': 'background voltage'},
        {'name': 'E',     'indexed': True,
         'description': 'external drive'},
        {'name': 'tau',   'indexed': False, 'domain': 'positive',
         'description': 'membrane time constant'},
        {'name': 'a',     'indexed': False,
         'description': 'quadratic transfer-function gain: phi(v) = a v^2'},
        {'name': 'tau_g', 'indexed': False, 'domain': 'positive',
         'description': 'synaptic exponential filter timescale'},
        {'name': 'w',     'indexed': True,
         'description': 'synaptic weight matrix w_{ij}'},
    ],

    # -----------------------------------------------------------------------
    # Kernels
    # -----------------------------------------------------------------------
    'kernels': [
        {'name': 'g', 'sage_name': 'z_g', 'latex_name': 'g',
         'description': 'exponential synaptic filter, unit integral'},
    ],

    # -----------------------------------------------------------------------
    # Operators
    # -----------------------------------------------------------------------
    'operators': [
        {'name': 'Dt', 'sage_name': 'Dt', 'latex_name': r'\partial_t',
         'description': r'd/dt  (algebraic placeholder for the time-derivative operator)'},
    ],

    # -----------------------------------------------------------------------
    # Nonlinear functions
    # phi_i(v) = a v^2  (quadratic gain)
    # Taylor around v = v*_i:  phi = a v*^2 + 2 a v* dv + a dv^2 + 0 + ...
    # The framework Taylor-expands formally; the derivative symbols
    # phi0_i, phi1_i, phi2_i, ... land in the action, then ``mf_bg_conditions``
    # shifts phi0_i -> nstar_i (Poisson saddle) and ``specializations`` fixes
    # the higher derivatives to their concrete values.
    # -----------------------------------------------------------------------
    'functions': [
        {
            'name':         'phi',
            'indexed':      True,
            'deriv_prefix': 'phi',
            'latex':        r'\varphi',
            'description':  'quadratic gain function phi_i(v) = a v^2',
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
            # MF Poisson saddle:  n_dot_i* = +phi_i(v_i*) = +nstar_i
            'value': lambda ns: [+ns.nstar[i] for i in ns.pop],
        },
    ],

    # -----------------------------------------------------------------------
    # MF background conditions
    # -----------------------------------------------------------------------
    'mf_bg_conditions': lambda ns: {
        **{SR.var(f'phi0_{i+1}'): ns.nstar[i] for i in ns.pop},
        **{ns.vstar[i]: ns.E[i] + sum(ns.w[i][j] * ns.g * ns.nstar[j]
                                      for j in ns.pop)
           for i in ns.pop},
    },

    # -----------------------------------------------------------------------
    # Specializations
    #
    # phi quadratic:  phi1_i = 2 a v*_i,  phi2_i = 2 a,  phi_k_i = 0 for k >= 3
    #
    # We symbolically express phi1_i in terms of ``a`` and ``ns.vstar[i]``;
    # ``mf_bg_conditions`` then further rewrites ns.vstar[i] into ``E_i +
    # sum_j w_{ij} g n*_j``, so the final action has only (E, w, nstar, a,
    # tau, tau_g) as symbolic parameters + the g kernel symbol.
    #
    # Synaptic kernel g is NOT specialized in the time domain -- the
    # notebook's cell 8 substitutes its frequency-space image via the
    # 'kernel_ft_image' hook below.
    # -----------------------------------------------------------------------
    'specializations': lambda ns: {
        **{SR.var(f'phi1_{i+1}'): 2 * SR.var('a') * ns.vstar[i]
           for i in ns.pop},
        **{SR.var(f'phi2_{i+1}'): 2 * SR.var('a') for i in ns.pop},
        **{SR.var(f'phi{k}_{i+1}'): SR(0)
           for k in range(3, ns._taylor_order + 1)
           for i in ns.pop},
    },

    # -----------------------------------------------------------------------
    # Kernel frequency-space image  (used by notebook cell 8)
    #
    #     g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)
    #     g_hat(omega) = 1 / (1 + i omega tau_g)
    # -----------------------------------------------------------------------
    'kernel_ft_image': lambda ns, omega: {
        ns.g: SR(1) / (SR(1) + I * omega * ns.tau_g),
    },

    # -----------------------------------------------------------------------
    # Concrete transfer function  (for numerical MF evaluation in cell 23)
    # phi_i(v) = a * v^2
    # -----------------------------------------------------------------------
    'phi_concrete': lambda ns, i, v: SR.var('a') * v**2,

    # -----------------------------------------------------------------------
    # Mean-field self-consistency equations
    # n*_i = phi_i(v*_i) = a * (v*_i)^2
    # v*_i = E_i + sum_j w_{ij} * n*_j
    # Combined (nonlinear -- fsolve in cell 23):
    # n*_i = a * (E_i + sum_j w_{ij} * n*_j)^2
    # -----------------------------------------------------------------------
    'mf_equations': lambda ns: [
        ns.nstar[i] == SR.var('a') * (
            ns.E[i] + sum(ns.w[i][j] * ns.nstar[j] for j in ns.pop)
        )**2
        for i in ns.pop
    ],

    # -----------------------------------------------------------------------
    # Background rate convention
    # -----------------------------------------------------------------------
    'background_rate_convention': (
        'n_dot_i* = +phi_i(v_i*) = a * (v_i*)^2  '
        '[negative Poisson term:  +n~_i n_dot_i - (e^{n~_i}-1) phi_i]'
    ),

    # -----------------------------------------------------------------------
    # Action
    # -----------------------------------------------------------------------
    'action': lambda ns: sum(
        # S1: n~_i (n_dot_i* + dn_i)
        ns.nt[i] * (ns.ndot_bg[i] + ns.dn[i])
        # S2: -(e^{n~_i} - 1) phi_i(dv_i)
        - (exp(ns.nt[i]) - 1) * ns.phi(i, ns.dv[i])
        # S3: v~_i [(tau d_t + 1) dv_i + v_i* - E_i
        #          - sum_j w_{ij} g (n*_j + dn_j)]
        + ns.vt[i] * (
            (ns.tau * ns.Dt + 1) * ns.dv[i]
            + ns.vstar[i] - ns.E[i]
            - sum(ns.w[i][j] * ns.g * (ns.nstar[j] + ns.dn[j])
                  for j in ns.pop)
        )
        for i in ns.pop
    ),
}

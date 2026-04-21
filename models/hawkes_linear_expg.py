"""
models/hawkes_linear_expg.py
============================
Linear Hawkes 2-population model with a **tunable gain** `a` and an
**exponential synaptic filter** `g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)`.

Transfer function:
    phi_i(v) = a * v          (linear with gain a)

Synaptic filter (unit integral):
    g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)
    g_hat(omega) = 1 / (1 + i * omega * tau_g)

Action (before MF expansion):
    S = sum_i { n~_i n_dot_i - (e^{n~_i} - 1) phi_i(v_i)
              + v~_i [(tau d_t + 1) v_i - E_i - sum_j w_{ij} (g * n_dot_j)] }

MF saddle:
    n*_i = phi_i(v*_i) = a * v*_i
    v*_i = E_i + sum_j w_{ij} * int(g) * n*_j = E_i + sum_j w_{ij} * n*_j
Combined:
    n*_i = a * (E_i + sum_j w_{ij} * n*_j)   =>   n* = a (I - a W)^{-1} E
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

    'name': 'Linear Hawkes 2-population (gain a, exp filter tau_g)',

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
    # Scalar parameters  (SR symbols)
    #
    #   a      — gain of the linear transfer function phi(v) = a*v
    #   tau_g  — exponential synaptic filter timescale
    # -----------------------------------------------------------------------
    'parameters': [
        {'name': 'nstar', 'indexed': True,  'domain': 'positive',
         'description': 'background firing rate  nstar_i = phi_i(vstar_i) = a * vstar_i'},
        {'name': 'vstar', 'indexed': True,
         'description': 'background voltage'},
        {'name': 'E',     'indexed': True,
         'description': 'external drive'},
        {'name': 'tau',   'indexed': False, 'domain': 'positive',
         'description': 'membrane time constant'},
        {'name': 'a',     'indexed': False,
         'description': 'linear transfer-function gain: phi(v) = a v'},
        {'name': 'tau_g', 'indexed': False, 'domain': 'positive',
         'description': 'synaptic exponential filter timescale'},
        # 2D synaptic weight matrix.  `indexed=True` + a 2D list in
        # `fundamental` makes the generic parameter-substitution loop
        # in the notebook expand this into w_{i+1}{j+1} symbols
        # (w11, w12, w21, w22) and substitute each from fundamental['w'].
        {'name': 'w',     'indexed': True,
         'description': 'synaptic weight matrix w_{ij}'},
    ],

    # -----------------------------------------------------------------------
    # Kernels
    # g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)  — unit integral
    # Handled by the notebook via the 'kernel_ft_image' hook below, which
    # supplies g_hat(omega) = 1 / (1 + i omega tau_g) as a post-Fourier
    # substitution.  The kernel symbol ns.g therefore remains unspecialized
    # in the time-domain kernel matrix and is replaced only once the action
    # has been Fourier transformed.
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
    # phi_i(v) = a * v  (linear with gain a)
    # Taylor expansion yields phi0_i = 0 (formally), phi1_i = a, higher = 0.
    # -----------------------------------------------------------------------
    'functions': [
        {
            'name':         'phi',
            'indexed':      True,
            'deriv_prefix': 'phi',
            'latex':        r'\varphi',
            'description':  'linear gain function phi_i(v) = a v',
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
            # MF Poisson saddle:  n_dot_i* = +phi_i(v_i*) = +a*v_i* = +nstar_i
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
    # phi linear with gain a:
    #     phi1_i = a,   phi_k_i = 0  for k >= 2
    #
    # Synaptic kernel g:  NOT specialized in the time domain.  It is left
    # as a symbolic constant in the time-domain kernel matrix and the
    # notebook substitutes its frequency-domain image post-FT via the
    # 'kernel_ft_image' hook below.
    # -----------------------------------------------------------------------
    'specializations': lambda ns: {
        **{SR.var(f'phi1_{i+1}'): SR.var('a') for i in ns.pop},
        **{SR.var(f'phi{k}_{i+1}'): SR(0)
           for k in range(2, ns._taylor_order + 1)
           for i in ns.pop},
    },

    # -----------------------------------------------------------------------
    # Kernel frequency-space image  (NEW hook used by the notebook cell 8)
    #
    #     g(t) = (1/tau_g) exp(-t/tau_g) Theta(t)
    #     g_hat(omega) = int_0^inf (1/tau_g) exp(-t/tau_g) exp(-i omega t) dt
    #                  = 1 / (1 + i omega tau_g)
    # -----------------------------------------------------------------------
    'kernel_ft_image': lambda ns, omega: {
        ns.g: SR(1) / (SR(1) + I * omega * ns.tau_g),
    },

    # -----------------------------------------------------------------------
    # Concrete transfer function  (for numerical evaluation)
    # phi_i(v) = a * v
    # -----------------------------------------------------------------------
    'phi_concrete': lambda ns, i, v: SR.var('a') * v,

    # -----------------------------------------------------------------------
    # Mean-field self-consistency equations
    # n*_i = phi_i(v*_i) = a * v*_i
    # v*_i = E_i + sum_j w_{ij} * n*_j    (kernel integrates to 1)
    # Combined:
    # n*_i = a * (E_i + sum_j w_{ij} * n*_j)
    # -----------------------------------------------------------------------
    'mf_equations': lambda ns: [
        ns.nstar[i] == SR.var('a') * (
            ns.E[i] + sum(ns.w[i][j] * ns.nstar[j] for j in ns.pop)
        )
        for i in ns.pop
    ],

    # -----------------------------------------------------------------------
    # Background rate convention
    # -----------------------------------------------------------------------
    'background_rate_convention': (
        'n_dot_i* = +phi_i(v_i*) = a * v_i*  '
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
        # S3: v~_i [(tau d_t + 1) dv_i + v_i* - E_i - sum_j w_{ij} g (n*_j + dn_j)]
        + ns.vt[i] * (
            (ns.tau * ns.Dt + 1) * ns.dv[i]
            + ns.vstar[i] - ns.E[i]
            - sum(ns.w[i][j] * ns.g * (ns.nstar[j] + ns.dn[j])
                  for j in ns.pop)
        )
        for i in ns.pop
    ),
}

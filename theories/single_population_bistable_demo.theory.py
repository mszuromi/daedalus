"""
Single-population BISTABLE demo theory — multi-root MF showcase.

Linear-rate-like Hawkes voltage dynamics with a SIGMOIDAL transfer
function ``phi(v) = tanh(g_gain·v)``.  At sufficient feedback strength
``w·g_gain > 1`` the MF equation ``v = Em + w·tanh(g·v)`` has THREE
fixed points — two stable outer branches and one unstable middle
branch — making this the canonical test for the multi-root + linear-
stability machinery in ``pipeline._mean_field_dae``.

Single population (size 1), single state pair (n, v).  Default
parameters chosen so the three roots are well-separated:
  ``Em = 0, tau = 1, w = 2, g_gain = 2``  →
  ``n* = -0.9993`` (stable),  ``n* = 0`` (unstable),  ``n* = +0.9993``
  (stable).

Use with ``compute_cumulants(..., fixed_point_index=N)`` for N in
{0, 1, 2} to compute cumulants around each branch separately.
"""
from pipeline.theory import TheoryBuilder


def build():
    return (
        TheoryBuilder('Single population Bistable Demo')
        .population('E', size=1, description='Excitatory')
        .physical_field('n', population='E', description='spike train (sigmoidal rate)')
        .physical_field('v', population='E', description='voltage')
        .parameter('Em',     default=[0.0], indexed_by=['E'], domain='real')
        .parameter('tau',    default=[1.0], indexed_by=['E'], domain='positive')
        .parameter('w',      default=[[2.0]], indexed_by=['E', 'E'], domain='real')
        .parameter('g_gain', default=[2.0], indexed_by=['E'], domain='positive')
        .define_function('phi', args=['v'],
                         expression='tanh(g_gain[i]*v)', population='E')
        .define_kernel('g', time_expr='dirac_delta(t)', latex_name='g')
        .set_action_text('''
            sum( nt[i]*n[i]
            - (exp(nt[i])-1)*phi[i](v[i])
            + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i]
            - sum(w[i,j]*Conv(g, n[j]) for j in E))
            for i in E)
        ''')
        .equation(
            lhs='(tau[i]*Dt + 1) * v[i]',
            rhs='Em[i] + sum(w[i, j]*n[j] for j in E)',
            population='E',
        )
        .equation(
            lhs='n[i]',
            rhs='phi[i](v[i])',
            population='E',
        )
        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'Em':     [0.0],
    'tau':    [1.0],
    'w':      [[2.0]],
    'g_gain': [2.0],
}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('nE', 1), ('nE', 1)],
    'tau_max': 8.0,
    'tau_step': 0.5,
    'fixed_point_index_default': 0,
    'seed_box_default': {'n': (-1.5, 1.5), 'v': (-3.0, 3.0)},
}

"""
Single-population LINEAR conductance theory — stripped-down test fixture.

Same conductance synapse ``-w·v·Conv(g, n)`` as
single_population_conductance_test.theory.py, but with **linear** rate
``phi(v) = a·v`` instead of quadratic.  Linearising kills all the
``nt^k·dv^l`` interaction vertices that come from expanding
``-(exp(nt)-1)·phi(v)``, leaving only the ConvVertexType as the
non-local interaction in the (≥3)-degree sectors.  Lets us probe
the conductance-vertex code path on a diagram set that's an order of
magnitude smaller than the cubic-phi theory.
"""
from pipeline.theory import TheoryBuilder


def build():
    return (
        TheoryBuilder('Single population Linear Conductance Test')
        .population('E', size=1, description='Excitatory')
        .physical_field('n', population='E', description='spike train')
        .physical_field('v', population='E', description='voltage')
        .parameter('Em', default=[0.5], indexed_by=['E'], domain='positive')
        .parameter('tau', default=[10], indexed_by=['E'], domain='positive')
        .parameter('taug', default=[[5]], indexed_by=['E', 'E'], domain='positive')
        .parameter('a', default=[0.3], indexed_by=['E'], domain='positive')
        .parameter('w', default=[[0.4]], indexed_by=['E', 'E'], domain='positive')
        .define_function('phi', args=['v'], expression='a[i]*v', population='E')
        .define_kernel('g', time_expr='(1/taug[i,j])*exp(-t/taug[i,j])*heaviside(t)',
                       latex_name='g', indexed_by=['E', 'E'])
        .set_action_text('''
            sum( nt[i]*n[i]
            - (exp(nt[i])-1)*phi[i](v[i])
            + vt[i]*((tau[i]*Dt + 1)*v[i] - Em[i]
            - sum(w[i,j]*v[i]*Conv(g[i,j],n[j]) for j in E))
            for i in E)
        ''')
        .set_mf_equation('vstar', '(Em[i] / (1 - sum(w[i,j]*nstar[j] for j in E)))')
        .set_mf_equation('nstar', 'phi[i](vstar[i])')
        .build()
    )


DEFAULT_FUNDAMENTAL = {}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('nE', 1)],
    'tau_max': 20.0,
    'tau_step': 2.5,
}

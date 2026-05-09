"""
Linear Hawkes 2-population — text-driven theory file.

Same shell as ``quadratic_hawkes.theory.py`` but with a linear
transfer function.  Useful as a comparison reference (the linear
case has a closed-form mean-field solution, so it doubles as a
sanity check on the saddle solver).
"""
from pipeline.theory import TheoryBuilder


def build():
    """Linear Hawkes 2-pop with exp synaptic kernel.

      φ(v) = a · v        (linear transfer function)
      g(t) = (1/τ_g) · exp(-t/τ_g) · Θ(t)
    """
    return (
        TheoryBuilder('Linear Hawkes 2-pop', n_populations=2)

        .response_field('nt', indexed=True, latex=r'\tilde n')
        .response_field('vt', indexed=True, latex=r'\tilde v')
        .physical_field('dn', indexed=True, natural_name='n',
                        latex=r'\delta\dot n')
        .physical_field('dv', indexed=True, natural_name='v',
                        latex=r'\delta v')

        .parameter('nstar', indexed=True, domain='positive',
                   mean_field=True, natural_name='n')
        .parameter('vstar', indexed=True,
                   mean_field=True, natural_name='v')
        .parameter('E',     indexed=True)
        .parameter('tau',   default=10.0, domain='positive')
        .parameter('a',     default=1.0)
        .parameter('tau_g', default=2.5,  domain='positive')
        .parameter('w',     indexed='matrix')

        .define_function('phi', args=['v'], expression='a * v',
                         latex=r'\varphi')
        .define_kernel('g', freq_image='1 / (1 + I*omega*tau_g)',
                       latex_name='g')

        .set_action_text('''
            nt[i] * (nstar[i] + dn[i])
            - (exp(nt[i]) - 1) * phi(dv[i])
            + vt[i] * (
                (tau * Dt + 1) * dv[i]
                + vstar[i] - E[i]
                - sum(w[i, j] * g * (nstar[j] + dn[j]) for j in pop)
            )
        ''')

        .set_mf_equation('vstar',
            'E[i] + sum(w[i, j] * g * nstar[j] for j in pop)')
        .set_mf_equation('nstar', 'phi(vstar[i])')

        .build()
    )


DEFAULT_FUNDAMENTAL = {
    'E':     [0.78, 0.81],
    'w':     [[0.30, 0.25],
              [0.30, 0.35]],
    'tau':   10.0,
    'a':     1.0,
    'tau_g': 2.5,
}


METADATA = {
    'description':   'Linear Hawkes 2-pop with exp synaptic kernel.',
    'k_default':     2,
    'ell_default':   0,
    'recommended_external_fields': [('n', 1), ('n', 2)],
    'tau_max':       50.0,
    'tau_step':      0.5,
}

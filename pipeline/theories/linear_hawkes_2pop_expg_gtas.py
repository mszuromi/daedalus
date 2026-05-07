"""
Linear Hawkes, 2 populations, exp synaptic kernel, with GTaS external
correlated noise (Bernoulli + Gaussian shifts).

  phi(v) = a · v
  g(t) = (1/tau_g) · exp(-t/tau_g) · Theta(t)
  External rate process per cell:  Bernoulli(p_part)-marked Poisson(λ_X)
                                   with i.i.d. Gaussian shifts per event.

Equivalent to ``models/hawkes_linear_expg_gtas.py``.

The GTaS template adds:
  * physical field ``dm[i]`` and response field ``mt[i]``
  * parameters w_X, lambda_X, p_part, mu_shift_diff, sigma_shift_diff_sq, mstar
  * feedforward action term ``- vt[i] · w_X · g · (b_X + dm[i])``
  * MSR pairing ``+ mt[i] · dm[i]``
  * saddle shift for vstar
  * mstar = b_X MF equation
  * correlated_noises['X'] block with κ⁽²⁾ (cross + auto) and
    κ⁽³⁾, κ⁽⁴⁾ (auto-only — Poisson statistics for the marginal).
"""
from pipeline.theory import TheoryBuilder
from pipeline.theory_templates import (
    HawkesAction, ExpSynapticKernel, GTaSNoise,
)


_t = (
    TheoryBuilder(
        'Linear Hawkes 2-pop + GTaS external (phi = a·v, exp filter τ_g, '
        'Bernoulli + Gaussian shifts)',
        n_populations=2,
    )
    .response_field('nt', indexed=True, latex=r'\tilde{n}',
                    description='response field conjugate to spike train')
    .response_field('vt', indexed=True, latex=r'\tilde{v}',
                    description='response field conjugate to voltage')
    .physical_field('dn', indexed=True, latex=r'\delta\dot{n}',
                    description='spike-train fluctuation')
    .physical_field('dv', indexed=True, latex=r'\delta v',
                    description='voltage fluctuation')
    .parameter('nstar', indexed=True, domain='positive',
               description='background firing rate')
    .parameter('vstar', indexed=True, description='background voltage')
    .parameter('E',     indexed=True,
               description='external constant drive')
    .parameter('tau',   default=10.0, domain='positive')
    .parameter('a',     default=1.0,
               description='phi(v) = a · v')
    .parameter('tau_g', default=2.5, domain='positive',
               description='synaptic exponential filter timescale')
    .parameter('w',     indexed='matrix',
               description='synaptic weight matrix w_{ij}')
    .kernel('g', sage_name='z_g', latex_name='g',
            description='exponential synaptic filter, unit integral')
    .use_action_template(HawkesAction(phi='linear'))
    .use_synaptic_kernel(ExpSynapticKernel(timescale_param='tau_g'))
    # GTaSNoise auto-declares the dm/mt fields, the GTaS-specific
    # parameters, the saddle shift, and the κ^(2-4) cumulant block.
    .add_gtas_noise(GTaSNoise())
)

HAWKES_MODEL = _t.build()

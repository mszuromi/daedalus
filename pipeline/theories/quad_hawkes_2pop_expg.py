"""
Quadratic Hawkes, 2 populations, exponential synaptic kernel.

  phi(v) = a · v²
  g(t) = (1/tau_g) · exp(-t/tau_g) · Theta(t)
  no external GTaS noise

Equivalent to ``models/hawkes_quad_expg.py``.

The quadratic phi makes phi'(v*) = 2·a·v* nonzero (unlike linear), so
nstar appears in the (n_tilde, n_phys) bigrade vertices that the
expander generates.  This is the model that exercises the most
features of the diagrammatic machinery.
"""
from pipeline.theory import TheoryBuilder
from pipeline.theory_templates import HawkesAction, ExpSynapticKernel


_t = (
    TheoryBuilder('Quadratic Hawkes 2-population (phi = a·v², exp filter τ_g)',
                  n_populations=2)
    .response_field('nt', indexed=True, latex=r'\tilde{n}',
                    description='response field conjugate to spike train')
    .response_field('vt', indexed=True, latex=r'\tilde{v}',
                    description='response field conjugate to voltage')
    .physical_field('dn', indexed=True, latex=r'\delta\dot{n}',
                    description='spike-train fluctuation')
    .physical_field('dv', indexed=True, latex=r'\delta v',
                    description='voltage fluctuation')
    .parameter('nstar', indexed=True, domain='positive',
               description='background firing rate  n*_i = a · (v*_i)²')
    .parameter('vstar', indexed=True,
               description='background voltage')
    .parameter('E',     indexed=True,
               description='external constant drive')
    .parameter('tau',   default=10.0, domain='positive',
               description='membrane time constant')
    .parameter('a',     default=0.44,
               description='quadratic transfer-function gain: phi(v) = a · v²')
    .parameter('tau_g', default=5.0, domain='positive',
               description='synaptic exponential filter timescale')
    .parameter('w',     indexed='matrix',
               description='synaptic weight matrix w_{ij}')
    .kernel('g', sage_name='z_g', latex_name='g',
            description='exponential synaptic filter, unit integral')
    .use_action_template(HawkesAction(phi='quadratic'))
    .use_synaptic_kernel(ExpSynapticKernel(timescale_param='tau_g'))
)

HAWKES_MODEL = _t.build()

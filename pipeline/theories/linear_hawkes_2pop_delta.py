"""
Linear Hawkes, 2 populations, instantaneous (delta) synaptic kernel.

Equivalent to the hand-written ``models/hawkes_linear_sage.py``.

  phi(v) = v               (no gain parameter)
  synaptic kernel  g(t) = δ(t)
  no external GTaS noise

This is the simplest possible Hawkes model — useful as a sanity-check
target for the templates and as a "hello-world" example for a new
user wanting to see what a Theory declaration looks like.
"""
from pipeline.theory import TheoryBuilder
from pipeline.theory_templates import HawkesAction, DeltaKernel


_t = (
    TheoryBuilder('Linear Hawkes 2-population (delta synapse)',
                  n_populations=2)
    # ── Cortical fields ────────────────────────────────────────
    .response_field('nt', indexed=True, latex=r'\tilde{n}',
                    description='response field conjugate to spike train')
    .response_field('vt', indexed=True, latex=r'\tilde{v}',
                    description='response field conjugate to voltage')
    .physical_field('dn', natural_name='n', indexed=True, latex=r'\delta\dot{n}',
                    description='spike-train fluctuation')
    .physical_field('dv', natural_name='v', indexed=True, latex=r'\delta v',
                    description='voltage fluctuation')
    # ── Parameters ─────────────────────────────────────────────
    .parameter('nstar', mean_field=True, natural_name='n', indexed=True, domain='positive',
               description='background firing rate')
    .parameter('vstar', mean_field=True, natural_name='v', indexed=True,
               description='background voltage')
    .parameter('E',     indexed=True,
               description='external constant drive')
    .parameter('tau',   default=10.0, domain='positive',
               description='membrane time constant')
    .parameter('w',     indexed='matrix',
               description='synaptic weight matrix w_{ij}')
    # Kernel — instantaneous synapse (no frequency image, just a symbol)
    .kernel('g', sage_name='z_g', latex_name='g',
            description='synaptic kernel (instantaneous, δ(t))')
    # Action template — phi(v) = v (linear with unit gain)
    .use_action_template(HawkesAction(phi='linear_unit_gain'))
    .use_synaptic_kernel(DeltaKernel())
)

HAWKES_MODEL = _t.build()

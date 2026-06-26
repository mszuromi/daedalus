"""
Throwaway test theory: scalar quartic Langevin in MSR-JD form.

  dx/dt = mu*x - g*x^3 + sqrt(2D)*xi
  S = ∫ dt  xt*((Dt - mu)*x + g*x^3) - D*xt^2

Single field x (size=1), single response field xt.  Used to
exercise the 1×1 K_ft path in the propagator + Phase J integrator.
"""
from api.theory import TemporalTheoryBuilder


def build():
    return (
        TemporalTheoryBuilder('Quartic Double Well Toy')
        .population('A', size=1, description='single scalar field')
        .physical_field('x', population='A', description='order parameter')
        .parameter('mu', default=[-1.0], indexed_by=['A'], domain='real')
        .parameter('g',  default=[1.0],  indexed_by=['A'], domain='positive')
        .parameter('D',  default=[0.1],  indexed_by=['A'], domain='positive')
        .set_action_text('''
            sum(
                xt[i]*((Dt - mu[i])*x[i] + g[i]*x[i]^3) - D[i]*xt[i]^2
                for i in A
            )
        ''')
        .set_mf_equation('xstar', '0')   # x=0 is the stable saddle for mu<0
        .build()
    )


DEFAULT_FUNDAMENTAL = {}
METADATA = {
    'k_default': 1,
    'ell_default': 0,
    'recommended_external_fields': [('x', 1)],
    'tau_max': 10.0,
    'tau_step': 1.0,
}

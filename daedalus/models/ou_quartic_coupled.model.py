"""
OU Quartic, two coupled fields (white Gaussian noise) — text-driven model file.

Two Ornstein–Uhlenbeck variables, each with a cubic restoring nonlinearity,
coupled linearly and driven by independent white Gaussian noise,

    dx/dt = -mu1*x - eps1*x^3 + J1*y + xi,    <xi(t) xi(t')>   = 2 D1 delta(t - t'),
    dy/dt = -mu2*y - eps2*y^3 + J2*x + eta,   <eta(t) eta(t')> = 2 D2 delta(t - t').

The two-field sibling of ``ou_quartic``: a cross correlator <x y> and a
two-mode propagator, so loop diagrams carry two distinct rates (the
chain-simplex path of Phase J).
"""
from api.model import TemporalModelBuilder


def build():
    return (
        TemporalModelBuilder('OU Quartic (two coupled fields)', n_populations=0)
        .physical_field('x', indexed=True, description='variable1')
        .physical_field('y', indexed=True, description='variable2')
        .parameter('mu1', default=1.0, domain='real')
        .parameter('mu2', default=1.0, domain='real')
        .parameter('eps1', default=0.05, domain='positive')
        .parameter('eps2', default=0.05, domain='positive')
        # |J1*J2| < |mu1*mu2| keeps the origin linearly stable: at the defaults
        # the linearization at (0, 0) has eigenvalues -0.5 and -1.5
        # (J1 = J2 = mu1 = mu2 = 1 would sit on the bifurcation).
        .parameter('J1', default=0.5, domain='real')
        .parameter('J2', default=0.5, domain='real')
        .parameter('D1', default=1.0, domain='positive')
        .parameter('D2', default=1.0, domain='positive')
        .set_action_text('xt*((Dt+mu1)*x + eps1*x^3 - J1*y) - D1*xt^2 '
                         '+ yt*((Dt+mu2)*y + eps2*y^3 - J2*x) - D2*yt^2')
        .mf_residual(lhs='(Dt+mu1)*x', rhs='-eps1*x^3 + J1*y')
        .mf_residual(lhs='(Dt+mu2)*y', rhs='-eps2*y^3 + J2*x')
        # mu_i < 0 gives up to 9 saddles in the (x, y) plane: the stability
        # filter makes ``fixed_point_index`` range over the stable wells only.
        .stability_analysis(True)
        .build()
    )


DEFAULT_FUNDAMENTAL = {}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('dx', 1), ('dx', 1)],
    'tau_max': 50.0,
    'tau_step': 0.5,
    'fixed_point_index_default': 0,
}

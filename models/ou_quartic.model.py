"""
OU Quartic (white Gaussian noise) — text-driven model file.

The simplest nonlinear stochastic process in the framework: an
Ornstein–Uhlenbeck variable with a cubic restoring nonlinearity, driven by
*white* Gaussian noise,

    dx/dt = -mu*x - eps*x^3 + xi,    <xi(t) xi(t')> = 2 D delta(t - t').

This is the white-noise sibling of ``ou_quartic_colored`` — a SINGLE physical
field, no finite-tau_c Markovian embedding — so every diagram runs on the bare
white-noise system and the loop integrals stay cheap.  At mu>0 the mean-field
saddle is x*=0 (single well); mu<0 is the genuine double well (see
``ou_quartic_double_well``).
"""
from api.model import TemporalModelBuilder


def build():
    return (
        TemporalModelBuilder('OU Quartic (white noise)')
        .population('pop', size=1)
        .physical_field('x', population='pop', description='variable')
        .parameter('mu', default=1.0, domain='positive')
        .parameter('eps', default=0.02, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .set_action_text(
            'sum(xt[i]*((Dt+mu)*x[i] + eps*x[i]^3) - D*xt[i]^2 for i in pop)')
        .equation(lhs='(Dt+mu)*x[i]', rhs='-eps*x[i]^3', population='pop')
        .build()
    )


# Mildly-perturbative single-well regime: loop parameter g_eff ≈ eps·D/mu² =
# 0.02 here, so tree + loops converge fast and Phase J is cheap.
DEFAULT_FUNDAMENTAL = {'mu': 1.0, 'eps': 0.02, 'D': 1.0}


METADATA = {
    'k_default': 2,
    'ell_default': 0,
    'recommended_external_fields': [('dx', 1), ('dx', 1)],
    'tau_max': 8.0,
    'tau_step': 0.5,
    'fixed_point_index_default': 0,
}

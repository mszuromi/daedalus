"""Coupled 2-species reaction-diffusion in d=1 (the spatial multi-field
demo theory).

Two diffusing species ``a(x,t)``, ``b(x,t)`` with a matrix reaction
coupling (an off-diagonal predator-prey pair → a complex eigenvalue pair
→ damped spatio-temporal oscillation) and stabilizing cubic
nonlinearities, driven by independent white noise:

    da/dt = -mua*a - g*b  + Da ∇²a - ga*a³ + sqrt(2 Ta) ξ_a
    db/dt = -mub*b + h*a  + Db ∇²b - gb*b³ + sqrt(2 Tb) ξ_b

The default has EQUAL diffusion (Da = Db = 0.8) so the spectral-
assignment tree + loop driver applies directly.  Set Da ≠ Db (and a
Dyson order via the notebook's ``Config.dyson_order``) to exercise the
unequal-diffusion Dyson dressing.  The matched simulator is
``models/coupled_rd_1d_sim.simulate_coupled_rd_1d``.
"""
from pipeline.theory import SpatialTheoryBuilder


def build():
    b = (SpatialTheoryBuilder('coupled-rd-2species-1d')
         .physical_field('a', spatial_dim=1)
         .physical_field('b', spatial_dim=1)
         .parameter('mua', default=1.5, domain='positive')
         .parameter('mub', default=1.2, domain='positive')
         .parameter('Da', default=0.8, domain='positive')
         .parameter('Db', default=0.8, domain='positive')
         .parameter('g',  default=0.4)
         .parameter('h',  default=0.3)
         .parameter('ga', default=0.3)
         .parameter('gb', default=0.3)
         .parameter('Ta', default=1.0, domain='positive')
         .parameter('Tb', default=0.7, domain='positive'))
    act = ('at*((Dt+mua-Da*Laplacian)*a + g*b + ga*a^3) '
           '+ bt*((Dt+mub-Db*Laplacian)*b - h*a + gb*b^3) '
           '- Ta*at^2 - Tb*bt^2')
    return (b.set_action_text(act)
            .equation(lhs='(Dt+mua-Da*Laplacian)*a + g*b + ga*a^3', rhs='0')
            .equation(lhs='(Dt+mub-Db*Laplacian)*b - h*a + gb*b^3', rhs='0')
            .boundary('infinite').initial('stationary').build())


DEFAULT_FUNDAMENTAL = {
    'mua': 1.5, 'mub': 1.2, 'Da': 0.8, 'Db': 0.8,
    'g': 0.4, 'h': 0.3, 'ga': 0.3, 'gb': 0.3, 'Ta': 1.0, 'Tb': 0.7,
}
METADATA = {
    'k_default': 2,
    'ell_default': 1,
    'recommended_external_fields': [('da', 1), ('da', 1)],
    'spatial_grid': [-8.0, 8.0, 65],
    'tau_max': 0.0,
    'tau_step': 1.0,
}

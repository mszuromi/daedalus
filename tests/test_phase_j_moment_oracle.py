"""Phase J against an independent oracle: the exact perturbative moment hierarchy of a polynomial Ito SDE
(tests/_ito_moments.py, vendored from sft-wick, BSD-3).  The oracle shares no code with Daedalus: no diagrams, no
propagators, no Wick contractions; it works from the generator on polynomials.

It returns MOMENTS about the saddle, Daedalus returns CUMULANTS, so the oracle's lower-order means are subtracted.
Both cases have more than one field with non-gradient coupling, where no Boltzmann solution exists.
"""
import numpy as np
import pytest

pytest.importorskip('sage.all')
from api.theory import TemporalTheoryBuilder          # noqa: E402
import api.compute as AC                              # noqa: E402
import tests._ito_moments as im                       # noqa: E402

T_STATIONARY = 150.0


def _cov_ell(model, max_ell, par=None):
    res = AC.compute_cumulants(model, k=2, max_ell=max_ell, external_fields=[('dx', 1), ('dx', 1)],
                               parameters=par, tau_grid=np.array([0.0, 1.0]), use_cache=False,
                               parallel=False, verbose=False)
    return {int(l): complex(a[0]).real for l, a in res['C_tau_by_ell'].items()}


def _coupled_cubic_model():
    return (TemporalTheoryBuilder('coupled cubic (moment oracle)', n_populations=0)
            .physical_field('x', indexed=True).physical_field('y', indexed=True)
            .parameter('mu1', default=1.0, domain='positive').parameter('mu2', default=1.7, domain='positive')
            .parameter('eps', default=0.05, domain='positive')
            .parameter('J1', default=0.5, domain='real').parameter('J2', default=0.3, domain='real')
            .parameter('D1', default=1.0, domain='positive').parameter('D2', default=0.6, domain='positive')
            .set_action_text('xt*((Dt+mu1)*x + eps*x^3 - J1*y) - D1*xt^2 + yt*((Dt+mu2)*y - J2*x) - D2*yt^2')
            .equation(lhs='(Dt+mu1)*x', rhs='-eps*x^3 + J1*y')
            .equation(lhs='(Dt+mu2)*y', rhs='J2*x')
            .stability_analysis(True).build())


def test_coupled_cubic_one_loop_against_moment_hierarchy():
    mu1, mu2, J1, J2, D1, D2, eps = 1.0, 1.7, 0.5, 0.3, 1.0, 0.6, 0.05
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(np.array([[-mu1, J1], [J2, -mu2]]))
    sde.drift.append((0, -1.0, (3, 0), (1,)))                    # dx/dt ⊃ -eps x^3, tag = power of eps
    sde.add_constant_diffusion(np.diag([2 * D1, 2 * D2]))
    xx = im.unit(2, 0, 0)
    out = im.solve(sde, [(xx, (0,)), (xx, (1,))], [T_STATIONARY])
    tree, one_loop = out[(xx, (0,))][0], out[(xx, (1,))][0] * eps   # <x> = 0 by x -> -x symmetry: moment = cumulant
    got = _cov_ell(_coupled_cubic_model(), 1, {'mu1': mu1, 'mu2': mu2, 'J1': J1, 'J2': J2, 'D1': D1, 'D2': D2, 'eps': eps})
    assert got[0] == pytest.approx(tree, rel=2e-6, abs=0.0)       # includes the deliberate Ito tau = 0 nudge
    assert got[1] == pytest.approx(one_loop, rel=2e-6, abs=0.0)


def _quadratic_drift_model(nst, q):
    return (TemporalTheoryBuilder('quadratic drift two field (moment oracle)', n_populations=0)
            .physical_field('x', indexed=True).physical_field('y', indexed=True)
            .parameter('tau', default=2.0, domain='positive').parameter('taug', default=1.0, domain='positive')
            .parameter('w', default=0.4, domain='real').parameter('nst', default=nst, domain='positive')
            .parameter('q', default=q, domain='real')
            .set_action_text('xt*((tau*Dt+1)*x - w*y) + yt*((taug*Dt+1)*y - nst*x - q*x^2) - (nst/2)*yt^2')
            .equation(lhs='(tau*Dt+1)*x', rhs='w*y')
            .equation(lhs='(taug*Dt+1)*y', rhs='nst*x + q*x^2')
            .stability_analysis(True).build())


def test_quadratic_drift_connected_one_loop_against_moment_hierarchy():
    """Two quadratic-drift vertices: the oracle's eps^2 moment contains <x>^2 (disconnected), Daedalus's cumulant does not."""
    tau, taug, w, nst = 2.0, 1.0, 0.4, 0.246224870792
    q = nst / 2
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(np.array([[-1 / tau, w / tau], [nst / taug, -1 / taug]]))
    Q = np.zeros((2, 2, 2)); Q[1, 0, 0] = q / taug
    sde.add_quadratic_drift(Q, tag=(1,))
    B = np.zeros((2, 2)); B[1, 1] = nst / taug ** 2
    sde.add_constant_diffusion(B)
    x1, xx = im.unit(2, 0), im.unit(2, 0, 0)
    out = im.solve(sde, [(x1, (1,)), (xx, (0,)), (xx, (2,))], [T_STATIONARY])
    mean1, tree, m2 = out[(x1, (1,))][0], out[(xx, (0,))][0], out[(xx, (2,))][0]
    connected_one_loop = m2 - mean1 ** 2
    got = _cov_ell(_quadratic_drift_model(nst, q), 1)
    assert got[0] == pytest.approx(tree, rel=2e-6, abs=0.0)
    assert got[1] == pytest.approx(connected_one_loop, rel=2e-6, abs=0.0)

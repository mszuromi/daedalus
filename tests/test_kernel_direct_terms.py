"""
Kernel direct terms at vertex legs (item 56).

A conductance kernel whose frequency image is ``c + (strictly proper)``
(``g = c delta + a pole sum``: the all-pass Pade (n, n) image of a bare hard
delay) is folded into the dressed line it sits on, ``c G + g_smooth * G``, on
both Phase J routes; a kernel edge (a leg time) holds only a strictly proper
kernel and refuses it by name.  End-to-end checks (the delay driver, Markov
embeddings): ``tests/test_delay_nonlinear.py``.
"""
from __future__ import annotations

import cmath
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.engine.integration.time_domain import final_integral as FI     # noqa: E402

def test_freq_image_direct_split():
    from sage.all import I
    img = lambda w: (1 - I * w) / (1 + I * w)                         # noqa: E731
    assert FI._modes_from_freq_image(img) is None        # not strictly proper
    c, modes = FI._modes_from_freq_image(img, direct=True)
    # (1 - i w)/(1 + i w) = -1 + 2/(1 + i w):  g = -delta + 2 e^{-t}
    assert c == -1
    assert len(modes) == 1
    C, lam = modes[0]
    assert abs(C - 2) < 1e-14 and abs(lam + 1) < 1e-14
    # a strictly proper image: no direct term, the same modes as before
    img2 = lambda w: 3 / (2 + I * w)                                  # noqa: E731
    c2, m2 = FI._modes_from_freq_image(img2, direct=True)
    assert c2 == 0 and m2 == FI._modes_from_freq_image(img2)
    # the attachment helper: only a frequency image with c != 0
    assert FI._kernel_direct_split({'kernel_ft_fn': img2}) is None
    assert FI._kernel_direct_split({'kernel_ft_fn': img})[0] == -1
    assert FI._kernel_direct_split({'kernel_ft_fn': img,
                                    'kernel_td_fn': lambda t: t}) is None
    # a polynomial part of degree >= 1 is not a direct term
    assert FI._modes_from_freq_image(lambda w: 1 + I * w, direct=True) is None


@pytest.mark.parametrize('D', [0.0, 0.7])
def test_dressed_line_with_a_direct_term(D):
    """(g * G)(t) for g = c delta + c1 e^{mu t}, G = D delta + C e^{lam t}:
    c D delta + c C e^{lam t} + D c1 e^{mu t} + C c1 (e^{lam t} - e^{mu t}) /
    (lam - mu); the delta part c D only on a line with one."""
    c, c1, mu = -1.0, 2.0, -1.0
    C, lam = 0.8 - 0.3j, -0.4 + 1.1j
    ems = FI.EdgeModeSum(ri=0, pi=0, delta_coeff=complex(D), modes=((C, lam),))
    out = FI._dress_edge_modes(ems, [(c1, mu)], direct=c)
    assert complex(out.delta_coeff) == c * D
    for t in (0.0, 0.3, 1.7, 4.0):
        ref = (c * C * cmath.exp(lam * t) + D * c1 * cmath.exp(mu * t)
               + C * c1 * (cmath.exp(lam * t) - cmath.exp(mu * t)) / (lam - mu))
        got = sum(Cm * cmath.exp(lm * t) for (Cm, lm) in out.modes)
        assert abs(got - ref) < 1e-14 * max(1.0, abs(ref))
    # direct = 0 is the old dressing, bit for bit
    a = FI._dress_edge_modes(ems, [(c1, mu)])
    b = FI._dress_edge_modes(ems, [(c1, mu)], direct=0)
    assert a == b and complex(a.delta_coeff) == 0


def test_realize_kernel_direct_term():
    from sage.all import I
    from daedalus.api._realization import realize_kernel, RealizationRefused
    att = {'symbol': 'ap', 'kernel_ft_fn': lambda w: (1 - I * w) / (1 + I * w)}
    with pytest.raises(RealizationRefused) as exc:
        realize_kernel(att, {})
    assert exc.value.reason == 'kernel_not_strictly_proper'
    ent, field = realize_kernel(att, {}, direct=True)
    assert field == 'QQ' and ent.d == -1.0 and ent.n_states == 1
    for s in (0.2, 1.0 + 2.0j, 5.0j):
        assert abs(ent.transfer(s) - (1 - s) / (1 + s)) < 1e-14
    # a pure constant: no states, d only
    ent0, _f = realize_kernel({'symbol': 'k', 'kernel_ft_fn': lambda w: 3 + 0 * w}, {},
                              direct=True)
    assert ent0.n_states == 0 and ent0.d == 3.0
    # a polynomial part of degree >= 1 is refused either way
    with pytest.raises(RealizationRefused) as exc:
        realize_kernel({'symbol': 'p', 'kernel_ft_fn': lambda w: 1 + I * w}, {}, direct=True)
    assert exc.value.reason == 'kernel_not_strictly_proper'


def test_statespace_kernel_edge_refuses_and_dressed_line_takes_a_direct_term():
    """``_ModelOps``: a kernel edge declines a direct term by name; a dressed
    line is the series connection with C = [d2 c1, c2], whose impulse
    response is d2 G(t) + (g_smooth * G)(t)."""
    from sage.all import I
    from daedalus.engine.integration.time_domain import statespace as SS
    from daedalus.api._realization import EntryRealization
    import types
    a1, b1, c1 = np.array([[-0.5]]), np.array([1.0]), np.array([0.8])
    ent_g = EntryRealization(a=a1, b=b1, c=c1, d=0.0, n_states=1)
    real = types.SimpleNamespace(A=a1, entries={(0, 0): ent_g})
    ops = SS._ModelOps(real, {})
    att = {'symbol': 'ap', 'kernel_ft_fn': lambda w: (1 - I * w) / (1 + I * w)}
    assert ops.kernel_entry(att) == 'kernel_direct_term'
    # the run-level predictions count it (as a dressed line): the run is
    # covered, the pre-pass charges states, not pole tuples
    kr = ops.kernel_route(att)
    assert not isinstance(kr, str) and kr.d == -1.0 and kr.n_states == 1
    # ... but only where Phase J dresses it: a local c delta kernel (a time
    # image) and a time image c delta + smooth stay declined, as before
    from sage.all import SR, dirac_delta, exp
    from daedalus.engine.core.vertices import KernelTimeImage
    t = SR.var('dt56tt')
    for img in (2 * dirac_delta(t), dirac_delta(t) + exp(-t)):
        assert ops.kernel_route({'symbol': 'loc', 'kernel_td_fn': KernelTimeImage(img, t)}) \
            == 'kernel_direct_term'
    dr = ops.dressed_entry(0, 0, att)
    assert not isinstance(dr, str)
    from scipy.linalg import expm
    # G(t) = 0.8 e^{-t/2}; g = -delta + 2 e^{-t}:
    # (g * G)(t) = -0.8 e^{-t/2} + 1.6 (e^{-t/2} - e^{-t}) / (1/2)
    for t in (0.0, 0.4, 2.5):
        got = float(np.real(dr.c @ expm(dr.a * t) @ dr.b))
        ref = -0.8 * np.exp(-t / 2) + 3.2 * (np.exp(-t / 2) - np.exp(-t))
        assert abs(got - ref) < 1e-13

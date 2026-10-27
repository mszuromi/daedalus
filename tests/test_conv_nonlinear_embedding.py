"""
``Conv(K, X)`` with ``X`` a NONLINEAR function of the fields (``Conv(k, v^2/a)``,
``Conv(k, phi(v))``): the SDE front end embeds it as auxiliary linear fields
(``SDE._embed_nonlinear_convs``), so the result is the hand-embedded model's.

The kernel route integrated such a term wrong with status 'ok' (it delayed ONE
copy of a fluctuation leg of ``X``, ``x (g * x)``, where every leg of the product
sits at the same earlier time): one-loop terms were off by 10-25 %, trees exact.
Checked here against the hand-written Markov embedding (the filter states as
fields): a single pole, separated poles, a repeated pole (alpha) and a kernel
with a numerator; k = 1 and 2, one and two loops; a point-process rate with a
kernel indexed by the summed population.  A hand-written action with such a term
is refused by name (``tests/test_frontend_small_bugs.py``).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import compute_cumulants                                   # noqa: E402
from daedalus.api.noise import Poisson                                       # noqa: E402
from daedalus.api.sde import SDE, SDEError                                   # noqa: E402

KW = dict(parallel=False, verbose=False, use_cache=False)
IMG = {'single': '1/(1 + I*omega*s{n})',
       'sep': '1/((1 + I*omega*s{n})*(1 + I*omega*r*s{n}))',
       'rep': '1/(1 + I*omega*s{n})^2',
       'num': '(1 + I*omega*q)/((1 + I*omega*s{n})*(1 + I*omega*r*s{n}))'}


def _run(m, k, ell, ext, taus=(0.0,)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = compute_cumulants(m.build(), k=k, max_ell=ell, external_fields=ext,
                              tau_grid=np.array(taus), **KW)
    assert r['status'] == 'ok'
    return r


def _loops(r, ell):
    return np.array([np.real(np.asarray(r['C_tau_by_ell'][q])) for q in range(1, ell + 1)])


def _close(a, b, rtol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert np.max(np.abs(a - b)) <= rtol * np.max(np.abs(b)), (a, b)


# ── a two-population rate model, synapses on v^2 ─────────────────────
P = dict(tE=0.015867, tI=0.0089374, bE=2.23679, bI=-3.83811, aE=9, aI=10, wEE=0.725655,
         wEI=-0.429340, wIE=0.874274, wII=-0.0220371, xE=0.331705, xI=0.256509, bX=5.640064,
         lam=27.77948, pc=0.1, sEE=0.001, sEI=0.005, sIE=0.0005, sII=0.003, r=1.7, q=0.0007)
SRC = {'EE': 'vE^2/aE', 'EI': 'vI^2/aI', 'IE': 'vE^2/aE', 'II': 'vI^2/aI'}


def _two_pop(name):
    m = SDE(name)
    for k, v in P.items():
        m = m.parameter(k, default=v)
    return m.field('vE').field('vI')


def _finish(m, term):
    return (m.noise('X', Poisson(rate='lam', size='pc'), into={'vE': 'xE/tE', 'vI': 'xI/tI'})
            .drift('vE', f"(-(vE - bE) + wEE*{term('EE')} + wEI*{term('EI')} + xE*(bX - lam*pc))/tE")
            .drift('vI', f"(-(vI - bI) + wIE*{term('IE')} + wII*{term('II')} + xI*(bX - lam*pc))/tI"))


def _kernel_two_pop(mode):
    m = _two_pop(f'conv nonlinear {mode}')
    for n in SRC:
        m = m.define_kernel(f'k{n}', freq_image=IMG[mode].format(n=n))
    return _finish(m, lambda n: f'Conv(k{n}, {SRC[n]})')


def _embedded_two_pop(mode):
    """The filter states written by hand: ra' = (X - ra)/s, rb' = (ra - rb)/(r s)
    (or /s for the repeated pole); the numerator 1 + i omega q reads q rb'."""
    m = _two_pop(f'conv embedded {mode}')
    for n in SRC:
        m = m.field(f'ra{n}').drift(f'ra{n}', f'({SRC[n]} - ra{n})/s{n}')
        if mode != 'single':
            tau2 = f'(r*s{n})' if mode in ('sep', 'num') else f's{n}'
            m = m.field(f'rb{n}').drift(f'rb{n}', f'(ra{n} - rb{n})/{tau2}')

    def out(n):
        if mode == 'single':
            return f'ra{n}'
        if mode == 'num':
            return f'(rb{n} + q*(ra{n} - rb{n})/(r*s{n}))'
        return f'rb{n}'
    return _finish(m, out)


@pytest.mark.parametrize('mode,k', [('single', 1), ('sep', 1), ('rep', 1), ('num', 1),
                                    ('single', 2),
                                    pytest.param('rep', 2, marks=pytest.mark.slow)])
def test_two_population_one_loop_equals_the_embedding(mode, k):
    """k = 1 for every kernel type; k = 2 (two fields, 10 with the filter
    states at the repeated pole: ~3 min) for the single pole by default."""
    ext, taus = {1: ([('dvE', 1)], (0.0,)), 2: ([('dvE', 1), ('dvI', 1)], (0.0, 0.01))}[k]
    a = _run(_kernel_two_pop(mode), k, 1, ext, taus)
    b = _run(_embedded_two_pop(mode), k, 1, ext, taus)
    assert a['mf_values']['vEstar'][0] == pytest.approx(b['mf_values']['vEstar'][0], rel=1e-13)
    _close(np.real(np.asarray(a['C_tau_by_ell'][0])), np.real(np.asarray(b['C_tau_by_ell'][0])))
    _close(_loops(a, 1), _loops(b, 1))


def test_embedding_shows_in_the_action_and_notes():
    m = _kernel_two_pop('rep')
    txt = m.action_text()
    assert 'Conv(' not in txt
    assert 'kEE_emb1_2t[i]*(((sEE)*Dt + (1))*kEE_emb1_2[i] - kEE_emb1_1[i])' in txt
    assert any('Markov embedding' in n for n in m.notes)
    # derivations are idempotent: the second one builds the same model
    assert m.action_text() == txt
    # an affine argument keeps the kernel route
    s = (SDE('affine').parameter('s', default=0.2).parameter('lam', default=1.0)
         .define_kernel('k', freq_image='1/(1 + I*omega*s)').field('v')
         .drift('v', '-v + Conv(k, 2*v + 1)').noise('X', Poisson(rate='lam'), into={'v': 1}))
    assert 'Conv(k, 2*v[i] + 1)' in s.action_text()


# ── a one-field model at two loops ────────────────────────────────────
def _one_field(name):
    return (SDE(name).parameter('tau', default=1.0).parameter('w', default=0.1)
            .parameter('b', default=0.5).parameter('s', default=0.4).parameter('r', default=1.7)
            .parameter('q', default=0.3).parameter('lam', default=3.0).parameter('p', default=0.3)
            .field('v').noise('X', Poisson(rate='lam', size='p'), into={'v': '1/tau'}))


@pytest.mark.parametrize('mode', ['single', 'rep', 'num'])
def test_one_field_two_loops_equal_the_embedding(mode):
    a = (_one_field(f'k {mode}').define_kernel('k', freq_image=IMG[mode].format(n=''))
         .drift('v', '(-v + b + w*Conv(k, v^2))/tau'))
    b = _one_field(f'e {mode}').field('ua').drift('ua', '(v^2 - ua)/s')
    if mode == 'single':
        b = b.drift('v', '(-v + b + w*ua)/tau')
    else:
        tau2 = 's' if mode == 'rep' else '(r*s)'
        out = 'ub' if mode == 'rep' else '(ub + q*(ua - ub)/(r*s))'
        b = b.field('ub').drift('ub', f'(ua - ub)/{tau2}').drift('v', f'(-v + b + w*{out})/tau')
    for k, ext in ((1, [('dv', 1)]), (2, [('dv', 1), ('dv', 1)])):
        ra, rb = _run(a, k, 2, ext, (0.0, 0.5)), _run(b, k, 2, ext, (0.0, 0.5))
        _close(_loops(ra, 2), _loops(rb, 2))


# ── a spike train whose rate filters phi(v) with a kernel of the summed population ─
def _rate_net(embedded):
    s = (SDE('rate net embedded' if embedded else 'rate net').population('E', size=2)
         .parameter('Em', default=[-1.0, -1.3], indexed_by=['E'])
         .parameter('s', default=[0.3, 0.5], indexed_by=['E'], domain='positive')
         .parameter('w', default=[[0.2, -0.3], [0.25, 0.1]], indexed_by=['E', 'E'])
         .parameter('tau', default=1.0, domain='positive')
         .define_function('phi', args=['v'], expression='exp(v)', population='E')
         .define_kernel('h', freq_image='1/(1 + I*omega)')
         .field('v', population='E'))
    if embedded:
        s = (s.field('u', population='E').equation('u', lhs='(s*Dt + 1)*u', rhs='phi(v)')
             .point_process('n', 'exp(Em + sum(w[i,j]*u[j] for j in E))', population='E'))
    else:
        s = (s.define_kernel('g', freq_image='1/(1 + I*omega*s[i])', indexed_by=['E'])
             .point_process('n', 'exp(Em + sum(w[i,j]*Conv(g[j], phi(v[j])) for j in E))',
                            population='E'))
    return s.equation('v', lhs='(tau*Dt + 1)*v', rhs='Em + 0.5*Conv(h, n)')


def test_population_kernel_in_a_rate_equals_the_embedding():
    ext = [('dv', 1), ('dv', 2)]
    a = _run(_rate_net(False), 2, 1, ext, (0.0, 1.0))
    b = _run(_rate_net(True), 2, 1, ext, (0.0, 1.0))
    _close(np.real(np.asarray(a['C_tau_by_ell'][0])), np.real(np.asarray(b['C_tau_by_ell'][0])))
    _close(_loops(a, 1), _loops(b, 1))


def test_refusals():
    with pytest.raises(SDEError, match='indexed by one population only'):
        (SDE('refuse').population('E', size=2).parameter('s', default=0.2)
         .field('v', population='E')
         .define_kernel('g', freq_image='1/(1 + I*omega*s)', indexed_by=['E', 'E'])
         .equation('v', lhs='Dt*v', rhs='-v + sum(Conv(g[i,j], v[j]^2) for j in E)')
         .action_text())
    with pytest.raises(SDEError, match='only a time_expr'):
        (SDE('refuse 1').parameter('s', default=0.2).field('v')
         .define_kernel('g', time_expr='exp(-t/s)/s*heaviside(t)')
         .drift('v', '-v + Conv(g, v^2)').action_text())
    with pytest.raises(SDEError, match='delayed kernel'):
        (SDE('refuse 2').parameter('s', default=0.2).parameter('d', default=0.5).field('v')
         .define_kernel('g', freq_image='1/(1 + I*omega*s)', delay='d')
         .drift('v', '-v + Conv(g, v^2)').action_text())

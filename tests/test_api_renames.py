"""The 0.4.0 spelling changes (audit CON-06): every old spelling keeps working with exactly one DeprecationWarning
and builds the byte-identical model; the removed ``dd.Kernel`` alias of ``PairingKernel`` is the only one that
goes."""
import glob
import os
import re
import warnings

import numpy as np
import pytest

pytest.importorskip('sage.all')

import daedalus as dd                                                  # noqa: E402
from api.model import ModelBuilder                                     # noqa: E402
from api.noise import (Gaussian, Meixner, NormalInverseGaussian,       # noqa: E402
                       NoiseError, VarianceGamma)
from api.sde import SDE                                                # noqa: E402


def _dep(fn):
    """``(result, [DeprecationWarning messages])`` of ``fn()``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = fn()
    return out, [str(w.message) for w in rec if issubclass(w.category, DeprecationWarning)]


# ── 1. Kernel ────────────────────────────────────────────────────────────

def test_dd_kernel_is_the_shape_builder_and_the_pairing_alias_is_gone():
    import api.delay as D
    import api.symbolic_out as so
    assert dd.Kernel is D.Kernel and dd.PairingKernel is so.PairingKernel
    assert not hasattr(so, 'Kernel') and 'Kernel' not in so.__all__
    assert dd.Kernel.exponential('tau') is not None


# ── 2. TemporalModelBuilder.equation -> mf_residual ──────────────────────

def _builder(method):
    b = (ModelBuilder('rename mf', n_populations=0).physical_field('x')
         .parameter('mu', default=1.0, domain='positive').parameter('D', default=0.2, domain='positive')
         .parameter('g', default=0.1))
    getattr(b, method)(lhs='(Dt + mu)*x', rhs='-g*x^3')
    return b.set_action_text('xt*(Dt(x) + mu*x + g*x^3) - D*xt^2').build()


def test_equation_is_a_deprecated_alias_of_mf_residual():
    new, w_new = _dep(lambda: _builder('mf_residual'))
    old, w_old = _dep(lambda: _builder('equation'))
    assert w_new == [] and len(w_old) == 1 and 'mf_residual' in w_old[0]
    assert new['spec_signature'] == old['spec_signature']
    assert new['equations'] == old['equations'] and new['equations']


def test_tracked_models_build_the_same_with_the_old_spelling():
    """The tracked model files use mf_residual now: their text with the old spelling builds the same model."""
    import importlib.util
    root = os.path.join(os.path.dirname(__file__), '..', 'models')
    n = 0
    for path in sorted(glob.glob(os.path.join(root, '*.model.py'))):
        text = open(path).read()
        if '.mf_residual(' not in text:
            continue
        built = []
        for variant in (text, re.sub(r'\.mf_residual\(', '.equation(', text)):
            ns = {'__name__': 'tracked_model'}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exec(compile(variant, path, 'exec'), ns)
                built.append(ns['build']())
        a, b = built
        assert a['spec_signature'] == b['spec_signature'], path
        assert a['equations'] == b['equations'] and a['parameters'] == b['parameters'], path
        n += 1
    assert n >= 3


def test_model_file_text_round_trips_with_either_spelling():
    from api.model_serialize import load_spec_from_file, render_model_file
    import tempfile
    src = ("from api.model import ModelBuilder\n\n\ndef build():\n    return (ModelBuilder('rt mf')\n"
           "        .physical_field('x')\n        .parameter('mu', default=1.0, domain='positive')\n"
           "        .{m}(lhs='(Dt + mu)*x', rhs='1.0')\n        .set_action_text('xt*(Dt(x) + mu*x - 1.0) - 0.1*xt^2')\n"
           "        .build())\n")
    outs = []
    for m in ('mf_residual', 'equation'):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, 'rt.model.py')
            open(f, 'w').write(src.format(m=m))
            outs.append(render_model_file(load_spec_from_file(f)))
    assert outs[0] == outs[1] and '.mf_residual(' in outs[0] and '.equation(' not in outs[0]


# ── 3. point_process(size=) -> jump_size= ────────────────────────────────

def _pp(**kw):
    s = SDE('rename pp').parameter('a', default=2.0).parameter('J', default=0.5).field('v')
    return s.point_process('n', 'a', **kw).drift('v', '-v + n').build()


def test_point_process_size_is_a_deprecated_alias_of_jump_size():
    new, w_new = _dep(lambda: _pp(jump_size='J'))
    old, w_old = _dep(lambda: _pp(size='J'))
    assert w_new == [] and len(w_old) == 1 and 'jump_size' in w_old[0]
    assert new['spec_signature'] == old['spec_signature']
    assert _pp()['spec_signature'] == _pp(jump_size='1')['spec_signature']
    from api.sde import SDEError
    with pytest.raises(SDEError, match='jump_size= only'):
        _pp(size='J', jump_size='J')


# ── 4. parameter(indexed=) with several populations ──────────────────────

def _pops(n, **p):
    b = ModelBuilder('rename idx', n_populations=0)
    for name in ('E', 'I')[:n]:
        b = b.population(name, size=2)
    b = b.parameter('w', default=[1.0, 1.0], **p).parameter('mu', default=1.0, domain='positive')
    return b


def test_indexed_without_indexed_by_warns_with_several_populations_only():
    def build(b):
        b = b.physical_field('x', population='E')
        b.mf_residual(lhs='(Dt + mu)*x[i]', rhs='1.0', population='E')
        return b.set_action_text('sum(xt[i]*(Dt*x[i] + mu*x[i] - 1.0) - 0.1*xt[i]^2 for i in E)').build()
    _, w1 = _dep(lambda: build(_pops(1, indexed=True)))
    _, w2 = _dep(lambda: build(_pops(2, indexed=True)))
    _, w3 = _dep(lambda: build(_pops(2, indexed_by=['E'])))
    assert w1 == [] and w3 == []                       # the single-population shorthand; indexed_by
    assert len(w2) == 1 and "parameter 'w'" in w2[0] and 'indexed_by' in w2[0]


# ── 6. noise vocabulary: loc= ────────────────────────────────────────────

@pytest.mark.parametrize('cls, args, old, new', [
    (NormalInverseGaussian, (2, 0.5, 1.2), 'mu', 'loc'),
    (Meixner, (1, 0.3, 1), 'mu', 'loc'),
    (VarianceGamma, (0.1, 0.3, 0.5), 'drift', 'loc'),
])
def test_location_keyword_is_loc_and_the_old_one_warns_once(cls, args, old, new):
    a, w_a = _dep(lambda: cls(*args, **{new: 0.1}))
    b, w_b = _dep(lambda: cls(*args, **{old: 0.1}))
    c, w_c = _dep(lambda: cls(*args, 0.1))
    assert w_a == [] and w_c == [] and len(w_b) == 1 and new in w_b[0]
    assert str(a.cgf()) == str(b.cgf()) == str(c.cgf()) and a.params == b.params == c.params
    assert 'loc' in a.params and old not in a.params
    with pytest.raises(NoiseError, match=f'give {new}= only'):
        cls(*args, **{new: 0.1, old: 0.2})
    # the default location is 0
    assert str(cls(*args).cgf()) == str(cls(*args, **{new: 0}).cgf())


def test_yaml_noise_blocks_accept_the_old_keys_with_a_warning():
    from api.config_run import _noise_object
    desc = {'type': 'NormalInverseGaussian', 'alpha': 2, 'beta': 0.5, 'delta': 1.2, 'mu': 0.1, 'into': 'x'}
    old, w = _dep(lambda: _noise_object(dict(desc)))
    new, w2 = _dep(lambda: _noise_object({**{k: v for k, v in desc.items() if k != 'mu'}, 'loc': 0.1}))
    assert len(w) == 1 and w2 == [] and str(old.cgf()) == str(new.cgf())


def test_sampling_uses_the_location():
    """The location shifts the sample mean by ``loc * dt`` (the samplers read it)."""
    rng = np.random.default_rng(0)
    for cls, args in ((NormalInverseGaussian, (2, 0.5, 1.2)), (Meixner, (1, 0.3, 1)),
                      (VarianceGamma, (0.1, 0.3, 0.5))):
        x0 = cls(*args).sample(1.0, 20000, np.random.default_rng(0))
        x3 = cls(*args, loc=3.0).sample(1.0, 20000, np.random.default_rng(0))
        assert abs(np.mean(x3) - np.mean(x0) - 3.0) < 0.05, cls

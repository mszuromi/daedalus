"""
tests/test_save_path_slug.py
============================
Output paths must never be built from ``str(params_dict)``.

A Sage / sympy substitution dict stringifies to ``{Em1: 1.0, tau1: 10.0,
...}`` (symbol keys → no quotes) or ``{'mu': 0.1, ...}``.  Used as a
directory name this produces junk dirs like
``{Em1: 1.0, ..., vstar2: 1.0}/`` and ``{mu: 0.1, ..., xstar1: 0.0}/`` at
the repo root — one per working point — which the ``{*}/`` ``.gitignore``
rule was a band-aid for.

These tests pin:
  * :func:`pipeline.save.params_slug` — the canonical, stable, readable,
    filesystem-safe encoding of a parameter dict;
  * :func:`pipeline.save._sanitize_output_path` — the backstop that
    rewrites a leaked dict-repr path component in place;
  * ``save_npz`` / ``save_csv`` materialise a slug directory, never a
    ``{...}/`` one, and the saved arrays still round-trip.

The save module is loaded standalone (it needs only numpy) so this test
runs under plain ``pytest`` as well as ``sage -python -m pytest``.

Run:  pytest tests/test_save_path_slug.py -q
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

# Load api/save.py in isolation — avoids importing the whole
# (Sage-heavy) ``pipeline`` package just to test the path helpers.
_SAVE_PY = os.path.join(os.path.dirname(__file__), '..', 'api', 'save.py')
_spec = importlib.util.spec_from_file_location('pipeline_save_under_test',
                                               _SAVE_PY)
save = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(save)


# The two exact fingerprints from the bug report.
_FP_HAWKES = ('{Em1: 0.8, Em2: 0.78, tau1: 10.0, tau2: 9.0, '
              'vstar1: 1.0473, vstar2: 0.9894}')
_FP_OU = '{mu: 0.1, eps: 0.1, D: 1.0, xstar1: 0.0}'

_UNSAFE = set('{}: ,\'"')


def _is_fs_safe(s: str) -> bool:
    return bool(s) and not (_UNSAFE & set(s)) and '/' not in s and '\\' not in s


# ── params_slug ───────────────────────────────────────────────────────

def test_params_slug_sorted_keys_and_kv_format():
    # Keys sorted by str(key); ``key=value``; integral floats truncated.
    assert save.params_slug({'mu': 0.1, 'eps': 0.1, 'D': 1.0}) == \
        'D=1__eps=0.1__mu=0.1'


def test_params_slug_is_filesystem_safe():
    slug = save.params_slug({'Em1': 0.8, 'tau1': 10.0, 'vstar2': 0.9894})
    assert _is_fs_safe(slug)


def test_params_slug_is_order_independent():
    a = save.params_slug({'a': 1.0, 'b': 2.0, 'c': 3.0})
    b = save.params_slug({'c': 3.0, 'a': 1.0, 'b': 2.0})
    assert a == b == 'a=1__b=2__c=3'


def test_params_slug_truncates_floats():
    # 6 significant figures, trailing zeros dropped.
    assert save.params_slug({'x': 6.7073170731}) == 'x=6.70732'
    assert save.params_slug({'x': 10.0}) == 'x=10'
    assert save.params_slug({'x': 0.0}) == 'x=0'


def test_params_slug_handles_vector_and_matrix_values():
    slug = save.params_slug({'E': [0.8, 0.78], 'w': [[0.0, 0.25], [0.2, 0.0]]})
    assert _is_fs_safe(slug)
    assert slug == 'E=0.8-0.78__w=0-0.25-0.2-0'


def test_params_slug_long_input_is_bounded_and_hashed():
    big = {f'param_number_{i}': float(i) for i in range(200)}
    slug = save.params_slug(big, max_len=180)
    assert len(slug) <= 180
    assert _is_fs_safe(slug)
    # deterministic
    assert slug == save.params_slug(big, max_len=180)


def test_params_slug_non_dict_falls_back_to_sanitized_string():
    assert save.params_slug('weird name!') == 'weird_name'


# ── dict-repr recovery + path sanitizing ──────────────────────────────

def test_slug_from_dict_repr_recovers_both_fingerprints():
    s_ou = save._slug_from_dict_repr(_FP_OU)
    assert s_ou == 'D=1__eps=0.1__mu=0.1__xstar1=0'
    assert _is_fs_safe(s_ou)

    s_hk = save._slug_from_dict_repr(_FP_HAWKES)
    assert _is_fs_safe(s_hk)
    # symbol-keyed repr has no quotes; keys survive verbatim.
    assert 'Em1=0.8' in s_hk and 'vstar2=0.9894' in s_hk


def test_looks_like_dict_repr():
    assert save._looks_like_dict_repr(_FP_OU)
    assert save._looks_like_dict_repr("{'mu': 0.1}")
    assert not save._looks_like_dict_repr('saved_theories')
    assert not save._looks_like_dict_repr('k2_ell1')
    assert not save._looks_like_dict_repr('{nobraces')


def test_sanitize_output_path_rewrites_dict_component():
    raw = os.path.join(_FP_OU, 'run.npz')
    with pytest.warns(UserWarning, match='junk directory'):
        fixed = save._sanitize_output_path(raw)
    assert fixed == os.path.join('D=1__eps=0.1__mu=0.1__xstar1=0', 'run.npz')
    # the offending braces/colons are gone from every component.
    assert _is_fs_safe(os.path.dirname(fixed))


def test_sanitize_output_path_leaves_clean_paths_untouched():
    clean = os.path.join('pipeline_outputs', 'ou_quartic', 'k2_ell1.npz')
    # no warning, identity return.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert save._sanitize_output_path(clean) == clean


# ── end-to-end: save_npz / save_csv never create a ``{...}/`` dir ──────

def _minimal_result():
    grid = np.array([-1.0, 0.0, 1.0])
    curve = np.array([0.5 + 0j, 1.0 + 0j, 0.5 + 0j])
    return {
        'config': {'k': 2, 'max_ell': 0, 'fundamental': {'mu': 0.1},
                   'external_fields': [], 'model_name': 'slug_test'},
        'tau_grid': grid,
        'C_tau': curve,
        'C_tau_by_ell': {0: curve},
        'mf_values': {'nstar': [0.5]},
    }


def test_save_npz_slugifies_dict_repr_directory(tmp_path):
    # Caller (buggy) builds the path from str(params): {mu: 0.1, ...}/out.npz
    bad = os.path.join(str(tmp_path), _FP_OU, 'out.npz')
    with pytest.warns(UserWarning, match='junk directory'):
        out = save.save_npz(_minimal_result(), bad)

    # No junk ``{...}/`` directory was created anywhere under tmp_path.
    junk = [d for d in os.listdir(tmp_path) if d.startswith('{')]
    assert junk == [], f'junk dirs created: {junk}'

    # The slug directory exists and holds a loadable npz.
    slug_dir = os.path.join(str(tmp_path), 'D=1__eps=0.1__mu=0.1__xstar1=0')
    assert os.path.isdir(slug_dir)
    assert os.path.isfile(out)
    with np.load(out) as z:
        assert int(z['k'][0]) == 2
        np.testing.assert_allclose(z['tau_grid'], [-1.0, 0.0, 1.0])


def test_save_csv_slugifies_dict_repr_directory(tmp_path):
    bad = os.path.join(str(tmp_path), _FP_HAWKES, 'out.csv')
    with pytest.warns(UserWarning, match='junk directory'):
        out = save.save_csv(_minimal_result(), bad)
    assert [d for d in os.listdir(tmp_path) if d.startswith('{')] == []
    assert os.path.isfile(out)
    assert '{' not in out and ': ' not in out


def test_save_npz_clean_path_unchanged(tmp_path):
    good = os.path.join(str(tmp_path), 'pipeline_outputs', 'run_k2.npz')
    out = save.save_npz(_minimal_result(), good)
    assert out == good
    assert os.path.isfile(good)

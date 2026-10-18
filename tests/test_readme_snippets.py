"""The README's Python snippets, executed as written (audit W5: a README example that breaks or
returns an incomplete result should fail a test, not a reader).

The blocks are read from README.md itself, so editing a snippet changes what runs here.  Only
the placeholder repository path is substituted.  The Hawkes snippet's last line (a 200-path
simulation) is left out here: its sample paths are not checked by anything in the snippet.
"""
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _blocks():
    text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    return re.findall(r'```python\n(.*?)```', text, re.S)


def _block(marker):
    hits = [b for b in _blocks() if marker in b]
    assert len(hits) == 1, f'README: {len(hits)} python blocks contain {marker!r}'
    return (hits[0].replace("'/path/to/daedalus/notebooks'", repr(os.path.join(ROOT, 'notebooks')))
            .replace("'/path/to/daedalus'", repr(ROOT)))


def _exec(code, name):
    ns = {}
    exec(compile(code, f'README.md ({name})', 'exec'), ns)
    return ns


def test_readme_python_blocks_are_the_ones_tested_here():
    """A new README snippet must get a test (or be listed here)."""
    markers = ['dd.run(model, dd.Config(', "SDE('Hawkes')", 'dd.build_cache(']
    blocks = _blocks()
    assert len(blocks) == len(markers), [b.splitlines()[0] for b in blocks]
    for b in blocks:
        assert sum(m in b for m in markers) == 1, b


def test_readme_script_example_runs_complete():
    ns = _exec(_block('dd.run(model, dd.Config('), 'script example')
    res = ns['res']
    assert isinstance(res, dict) and res.get('status', 'ok') != 'incomplete', res.get('status')


def test_readme_sde_hawkes_example_is_ok():
    code = _block("SDE('Hawkes')")
    head, sep, _sim = code.partition('rec = m.simulate(')
    assert sep, 'the Hawkes snippet no longer ends with the simulator line'
    res = _exec(head, 'Hawkes example')['res']
    assert res['status'] == 'ok', res['status']
    assert not res.get('phase_j_incomplete')


def test_readme_build_cache_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _exec(_block('dd.build_cache('), 'build_cache dry run')
    assert list(tmp_path.iterdir()) == []

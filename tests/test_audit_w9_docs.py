"""Audit wave 9: the public text points at things that exist (CON-05, CON-08)."""
from __future__ import annotations

import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tracked():
    try:
        out = subprocess.run(['git', '-C', ROOT, 'ls-files'], capture_output=True,
                             text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip('not a git checkout')
    return [f for f in out.split('\n') if f]


_TEXT = ('.py', '.md')
_SKIP = ('notebooks/', 'tests/fixtures', 'changelog.d/', 'scratch/', 'scripts/')


def _read(f):
    with open(os.path.join(ROOT, f), errors='ignore') as fh:
        return fh.read()


def test_docs_and_code_cite_no_missing_design_documents():
    tracked = set(_tracked())
    pat = re.compile(r'(?<![\w/])((?:docs|notebooks/spatial)/[\w./-]*\.(?:py|md|ipynb))(?![\w])')
    missing = []
    for f in sorted(tracked):
        if not f.endswith(_TEXT) or f.startswith(_SKIP) or f == 'CHANGELOG.md':
            continue
        for m in pat.finditer(_read(f)):
            p = m.group(1).rstrip('.')
            if p not in tracked and '<' not in p:
                missing.append(f'{f}: {p}')
    assert not missing, missing


def test_docs_use_the_current_package_names():
    bad = []
    for f in _tracked():
        if not (f.startswith('docs/') and f.endswith('.md')):
            continue
        s = _read(f)
        for m in re.finditer(r'(?<![\w/])(msrjd|pipeline)/\w', s):
            bad.append(f'{f}: {m.group(0)}')
        if re.search(r'\b(?:Temporal|Spatial)?TheoryBuilder\b', s):
            bad.append(f'{f}: TheoryBuilder')
    assert not bad, bad


def test_no_internal_ticket_numbers_in_library_text():
    pat = re.compile(r'\b(?:[Ii]tem|[Bb]ug) ?#?\d+\b')
    hits = []
    for f in _tracked():
        if not f.endswith('.py') or not f.startswith(('api/', 'engine/', 'simulations/', 'models/')) \
                and f != 'daedalus.py':
            continue
        for i, line in enumerate(_read(f).split('\n'), 1):
            if pat.search(line):
                hits.append(f'{f}:{i}: {line.strip()[:80]}')
    assert not hits, hits[:10]

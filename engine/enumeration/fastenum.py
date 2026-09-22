"""
engine.enumeration.fastenum
===========================
Loader for the compiled hot loops in ``_fastenum.pyx``.

The extension is built on first import with ``pyximport`` (build cache in
``<repo>/.pyxbld``, gitignored) so no install step is needed; it requires a C
compiler and takes a few seconds once.  It exports ``tree_candidates`` (the
edge-addition stage for one tree), ``aux_cert`` (certificates) and
``orientation_patterns`` (the causal-orientation search); see the .pyx header.  If Cython or the compiler is missing, or ``DAEDALUS_FASTENUM=0``,
``available`` is False and the callers fall back to the pure-Python code with
identical output.  ``reason`` says why when it is unavailable.
"""
from __future__ import annotations

import os
import sys
import warnings

available = False
reason = ''
aux_cert = None
orientation_patterns = None
tree_candidates = None

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get('DAEDALUS_FASTENUM', '1') == '0':
    reason = 'disabled by DAEDALUS_FASTENUM=0'
else:
    try:
        import pyximport
        pyximport.install(build_dir=os.path.join(_ROOT, '.pyxbld'), language_level=3,
                          inplace=False, build_in_temp=True)
        from engine.enumeration import _fastenum as _ext      # noqa: E402
        aux_cert = _ext.aux_cert
        orientation_patterns = _ext.orientation_patterns
        tree_candidates = _ext.tree_candidates
        available = True
    except Exception as e:                                     # pragma: no cover
        reason = f'{type(e).__name__}: {e}'
        warnings.warn(f'fastenum: compiled enumeration unavailable ({reason}); '
                      'using the pure-Python path.')

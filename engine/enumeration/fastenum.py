"""
engine.enumeration.fastenum
===========================
Loader for the compiled hot loops in ``_fastenum.pyx``.

The extension is built on first import with ``pyximport`` (build cache in
``<repo>/.pyxbld``, gitignored) so no install step is needed; it requires a C
compiler and takes a few seconds once.  It exports the compiled stages
``tree_filter_sparse6`` (E1), ``tree_candidates`` (E2), ``prediagram_certs``
and ``orientation_patterns`` (E3), and the certificate ``cert``: nauty's
``densenauty`` from ``_fastnauty.pyx``, linked against Sage's bundled
libnauty, when that builds (``cert_backend == 'nauty'``), else ``aux_cert``
(Sage's ``search_tree`` on a DenseGraph, ``cert_backend == 'aux'``).  See the
.pyx headers.  Certificates from the three backends (nauty, aux, Sage's
``canonical_label``) are different bytes for the same isomorphism class.  If Cython or the compiler is missing, or ``DAEDALUS_FASTENUM=0``,
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
prediagram_certs = None
tree_filter_sparse6 = None
cert = None
cert_backend = ''
nauty_reason = ''

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
        prediagram_certs = _ext.prediagram_certs
        tree_filter_sparse6 = _ext.tree_filter_sparse6
        cert = aux_cert
        cert_backend = 'aux'
        try:
            from engine.enumeration import _fastnauty as _nauty  # noqa: E402
            cert = _nauty.nauty_cert
            cert_backend = 'nauty'
        except Exception as e:                                 # pragma: no cover
            nauty_reason = f'{type(e).__name__}: {e}'
        _ext.set_cert_fn(cert)
        available = True
    except Exception as e:                                     # pragma: no cover
        reason = f'{type(e).__name__}: {e}'
        warnings.warn(f'fastenum: compiled enumeration unavailable ({reason}); '
                      'using the pure-Python path.')

"""A3c tolerance recorder (pytest plugin, scratch only; never imported by tracked code).

Load with:  -p tolrec_plugin -o enable_assertion_pass_hook=true   (tools/ on PYTHONPATH)
Env:        TOLREC_OUT=<path.jsonl>   (one JSON object per recorded comparison)

Records, for every comparison made FROM A FILE UNDER tests/:
  * pytest.approx (scalar element comparisons; aggregated per call site by the reader),
  * numpy.allclose / numpy.isclose / numpy.testing.assert_allclose / math.isclose,
  * every PASSING plain `assert` (pytest_assertion_pass: the explanation string carries the
    intermediate values, e.g. 'assert 3.2e-14 < 1e-09').
Each record: site (file:line), test nodeid, kind, achieved abs / rel error (max over elements),
tolerance arguments, passed flag. Wraps and passes through: behaviour of the tests is unchanged.
"""
import json
import math
import os
import sys
import threading

_OUT = os.environ.get('TOLREC_OUT', 'tolrec.jsonl')
_LOCK = threading.Lock()
_FH = None
_CUR = {'nodeid': None}
_TESTS_MARK = os.sep + 'tests' + os.sep


def _fh():
    global _FH
    if _FH is None:
        _FH = open(_OUT, 'a', buffering=1)
    return _FH


def _emit(rec):
    rec['nodeid'] = _CUR['nodeid']
    try:
        line = json.dumps(rec, default=str)
    except Exception:
        return
    with _LOCK:
        _fh().write(line + '\n')


def _site():
    """file:line of the direct caller, skipping pytest / numpy / this plugin's frames; None unless that
    caller lives under tests/ (comparisons made inside production code are not test tolerances)."""
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if ('tolrec_plugin' in fn or os.sep + '_pytest' + os.sep in fn
                or os.sep + 'numpy' + os.sep in fn):
            f = f.f_back
            continue
        if _TESTS_MARK in fn:
            i = fn.rfind(_TESTS_MARK)
            return fn[i + 1:] + ':' + str(f.f_lineno)
        return None
    return None


def _errs(a, b):
    import numpy as np
    try:
        a = np.asarray(a, dtype=complex)
        b = np.asarray(b, dtype=complex)
        a, b = np.broadcast_arrays(a, b)
        if a.size == 0:
            return 0.0, 0.0, 0.0
        d = np.abs(a - b)
        fin = np.isfinite(d)
        if not fin.any():
            return None, None, None
        d = d[fin]
        bb = np.abs(b)[fin]
        absd = float(d.max())
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(bb > 0, d / bb, np.where(d == 0, 0.0, np.inf))
        return absd, float(rel.max()), float(bb.max())
    except Exception:
        return None, None, None


def _install():
    import numpy as np
    import numpy.testing as npt

    real_allclose = np.allclose
    real_isclose = np.isclose
    real_assert_allclose = npt.assert_allclose
    real_misclose = math.isclose

    def _sig(*names):
        return names

    def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        out = real_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        s = _site()
        if s:
            ab, rl, sc = _errs(a, b)
            _emit({'site': s, 'kind': 'np.allclose', 'abs_err': ab, 'rel_err': rl, 'scale': sc,
                   'rtol': rtol, 'atol': atol, 'passed': bool(out)})
        return out

    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        out = real_isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        s = _site()
        if s:
            ab, rl, sc = _errs(a, b)
            try:
                p = bool(np.all(out))
            except Exception:
                p = None
            _emit({'site': s, 'kind': 'np.isclose', 'abs_err': ab, 'rel_err': rl, 'scale': sc,
                   'rtol': rtol, 'atol': atol, 'passed': p})
        return out

    def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='',
                        verbose=True, **kw):
        s = _site()
        ok = True
        try:
            return real_assert_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan,
                                        err_msg=err_msg, verbose=verbose, **kw)
        except AssertionError:
            ok = False
            raise
        finally:
            if s:
                ab, rl, sc = _errs(actual, desired)
                _emit({'site': s, 'kind': 'assert_allclose', 'abs_err': ab, 'rel_err': rl,
                       'scale': sc, 'rtol': rtol, 'atol': atol, 'passed': ok})

    def misclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
        out = real_misclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        s = _site()
        if s:
            ab, rl, sc = _errs(a, b)
            _emit({'site': s, 'kind': 'math.isclose', 'abs_err': ab, 'rel_err': rl, 'scale': sc,
                   'rtol': rel_tol, 'atol': abs_tol, 'passed': bool(out)})
        return out

    np.allclose = allclose
    np.isclose = isclose
    npt.assert_allclose = assert_allclose
    math.isclose = misclose

    # pytest.approx: patch the scalar element comparison (sequences / arrays / mappings all
    # reduce to ApproxScalar.__eq__ per element).
    from _pytest import python_api as pa
    real_eq = pa.ApproxScalar.__eq__

    def approx_eq(self, actual):
        out = real_eq(self, actual)
        s = _site()
        if s:
            try:
                tol = self.tolerance
            except Exception:
                tol = None
            ab, rl, sc = _errs(actual, self.expected)
            _emit({'site': s, 'kind': 'approx', 'abs_err': ab, 'rel_err': rl, 'scale': sc,
                   'rtol': self.rel, 'atol': self.abs, 'eff_tol': tol, 'passed': bool(out)})
        return out

    pa.ApproxScalar.__eq__ = approx_eq


def pytest_configure(config):
    _install()


def pytest_runtest_setup(item):
    _CUR['nodeid'] = item.nodeid


def pytest_assertion_pass(item, lineno, orig, expl):
    fn = str(item.path)
    i = fn.rfind(_TESTS_MARK)
    # lineno is the assert's line in the test module (the module the test function lives in)
    _emit({'site': (fn[i + 1:] if i >= 0 else fn) + ':' + str(lineno), 'kind': 'assert_pass',
           'orig': orig[:400], 'expl': expl[:1200]})

"""
tests/test_propagator_timeout.py
================================
The SIGALRM budget of the exact propagator inverse
(``daedalus.api._propagator._run_with_timeout``).  The timeout can be raised
inside a Cython callback that ignores exceptions (Sage's ``py_is_real``,
called back from pynac: printed as "Exception ignored", the callback then
returns a default value).  Such a callback is modelled here by a loop that
catches and drops the timeout.  The budget must still take effect promptly,
and a value computed after a swallowed raise is never returned.
"""
import os
import signal
import sys
import time

import pytest

pytest.importorskip('sage.all')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from daedalus.api import _propagator as PR                        # noqa: E402

pytestmark = pytest.mark.skipif(not PR._HAS_SIGALRM, reason='needs SIGALRM')


@pytest.fixture(autouse=True)
def _fast_refire(monkeypatch):
    monkeypatch.setattr(PR, '_TIMEOUT_REFIRE_SEC', 0.05)


def _assert_clean():
    # no timer left armed, the previous handler back
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is not None
    assert getattr(signal.getsignal(signal.SIGALRM), '__name__', '') != '_handler'


def test_a_swallowed_timeout_is_raised_again_promptly():
    swallowed = []

    def fn():
        while True:
            try:
                time.sleep(0.005)
            except PR._PolynomialPathTimeout:
                if not swallowed:               # the first raise lands in a silent callback
                    swallowed.append(time.perf_counter())
                    continue
                raise

    t0 = time.perf_counter()
    with pytest.raises(PR._PolynomialPathTimeout):
        PR._run_with_timeout(fn, 0.2)
    assert swallowed and time.perf_counter() - t0 < 2.0
    _assert_clean()


def test_a_result_after_a_swallowed_timeout_is_discarded():
    def fn():
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            try:
                time.sleep(0.005)
            except PR._PolynomialPathTimeout:
                pass                            # every raise swallowed
        return 'a value a silenced callback may have corrupted'

    with pytest.raises(PR._PolynomialPathTimeout, match='discarded'):
        PR._run_with_timeout(fn, 0.1)
    _assert_clean()


def test_within_budget_returns_the_value_and_leaves_no_timer():
    assert PR._run_with_timeout(lambda a, b=0: a + b, 5, 2, b=3) == 5
    _assert_clean()
    time.sleep(0.1)                             # nothing fires later

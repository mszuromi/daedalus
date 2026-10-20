"""tests._cputime.CpuClock: CPU seconds of the process and its children, not wall seconds."""
import subprocess
import sys
import time

from tests._cputime import CpuClock


def test_sleep_costs_no_cpu_but_a_busy_loop_does():
    c = CpuClock()
    time.sleep(0.6)
    assert c.elapsed() < 0.3                     # wall 0.6 s, CPU ~0
    c.reset()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.4:
        sum(i * i for i in range(1000))
    assert c.elapsed() > 0.25


def test_a_busy_child_process_is_counted():
    c = CpuClock()
    p = subprocess.Popen([sys.executable, '-c',
                          'import time\nt=time.perf_counter()\n'
                          'while time.perf_counter()-t<1.0: sum(i*i for i in range(1000))\n'
                          'time.sleep(5)'])
    time.sleep(1.4)                               # the child is still alive (a persistent helper)
    try:
        assert c.elapsed() > 0.5
    finally:
        p.kill()
        p.wait()

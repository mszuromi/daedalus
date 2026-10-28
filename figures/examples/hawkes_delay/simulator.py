"""Independent simulator for the delayed exponential Hawkes network (no diagrams), numba.

   tau dv_i/dt = -(v_i - Em) + sum_j w_ij F2_j,  taug dF1_j/dt = -F1_j + n_j(t - D),  taug dF2_j/dt = -F2_j + F1_j
(F2 = alpha-filtered, delayed spike train with unit area). Spike counts per step ~ Poisson(a exp(v_i) dt) with v_i taken
at the step start (Ito). The linear filters are updated exactly over each step; spikes wait nD = D/dt steps in a
delay line. Returns binned spike counts and bin-averaged voltages per replica."""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=False)
def run(n_rep, n_steps, n_burn, dt, Em, tau, taug, a, w, binsz, seed, nD=0):
    n_b = n_steps // binsz
    counts = np.zeros((n_rep, 2, n_b)); volts = np.zeros((n_rep, 2, n_b))
    for r in prange(n_rep):
        np.random.seed(seed + r)
        v = np.array([Em, Em]); F1 = np.zeros(2); F2 = np.zeros(2); buf = np.zeros((max(nD, 1), 2))   # spike delay line, nD steps
        ev = np.exp(-dt/taug); ev_v = np.exp(-dt/tau)
        for t in range(n_burn + n_steps):
            spk = np.zeros(2)
            for i in range(2):
                spk[i] = np.random.poisson(a*np.exp(v[i])*dt)
            # linear filters: exact over the step for the homogeneous part, spikes enter as delta impulses at the step start
            inp = np.zeros(2)
            for i in range(2):
                inp[i] = w[i, 0]*F2[0] + w[i, 1]*F2[1]
            for i in range(2):
                v[i] = Em + (v[i] - Em)*ev_v + inp[i]*(1 - ev_v)
            if nD > 0:
                slot = t % nD; arr = buf[slot].copy()            # written at step t - nD: spikes delayed by nD*dt = D
                for j in range(2): buf[slot, j] = spk[j]
            else:
                arr = spk.copy()
            for j in range(2):
                F1n = F1[j]*ev + arr[j]/taug
                F2[j] = F2[j]*ev + F1[j]*(dt/taug)*ev
                F1[j] = F1n
            if t >= n_burn:
                b = (t - n_burn)//binsz
                if b < n_b:
                    for i in range(2):
                        counts[r, i, b] += spk[i]; volts[r, i, b] += v[i]/binsz
    return counts, volts

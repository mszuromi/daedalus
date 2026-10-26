#### Self-consistent Gaussian (Hartree) prototype

New research module `api/hartree.py`. It is not wired into `compute_cumulants` or `dd`. `hartree(model, parameters)`
returns the self-consistent Gaussian of a temporal model: the mean, covariance and dressed drift that solve
$\langle f\rangle = 0$ and $A\Sigma + \Sigma A^{\mathsf T} + \langle Q\rangle = 0$ with $A = \langle\partial f\rangle$.
This is the Hartree truncation of the 2PI effective action. The module also gives the lag correlator $C(\tau)$ and the
spectrum. It covers several fields and colored noise (through the Markovian embedding). It refuses kernels, spatial
models and derivative vertices.

`docs/hartree_ou_quartic_comparison.py` compares, for OU quartic, the exact Boltzmann / Fokker-Planck answer, the loop
series at 1–3 loops, Hartree, the loop series re-expanded around a variational mass, and simulation. It also covers the
O(N) large-N limit and the double well. The results are in `docs/hartree_prototype.md`. No existing number moves.

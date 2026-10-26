.. _models-noise:

The noise library
=================

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: the coloured-Gaussian feature; the in-flight Gaussian-input session.

.. todo::

   The noise library: write the parts that wait for the coloured-Gaussian feature; the in-flight Gaussian-input session once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: choosing and combining noise sources.
   - Contents: one CGF per source; Gaussian; Poisson and compound Poisson with the jump laws (Dirac, Exponential,
     Gamma, GaussianJump, Laplace, Uniform, Bernoulli, Binomial, FinitePMF, Geometric, …); Lévy sources (Gamma and
     inverse-Gaussian processes, VarianceGamma, NIG, CGMY, tempered stable, Meixner, bilateral gamma, Skellam,
     negative binomial); ``Cumulants`` and ``CGF`` for a user law; ``SumSource``; named composites (Merton, Kou,
     ``levy_driven_ou``); state-dependent amplitudes; cumulant truncation (``CumulantTruncationWarning``); samplers
     (``source.sample``); per-source interpretation and the Marcus rule (exact affine terms, truncation advisory);
     **coloured Gaussian noise** (``Gaussian(kernel=...)``, ``GaussianVector(cov=...)``) and the Markovian embedding;
     correlated noise across fields; **Gaussian input jitter** (in flight).
   - Sources: ``api/noise.py``, ``api/noise_samplers.py``, ``docs/noise_library.md``, ``docs/correlated_noise_capabilities.md``,
     ``api/colored_to_markovian.py``, tutorial 02.
   - Examples: Lévy-driven OU vs exact cumulants; coloured-noise OU (``OU_quartic_correlated_colored`` notebook).
   - Auto: **yes**: the table of sources and jump laws (name, parameters, CGF, sampler available) can be generated from
     the classes (``NoiseSource`` / ``JumpDistribution`` subclasses) by a small Sphinx extension or a pre-build script.
   - Status: library now; coloured-noise section **after the coloured-Gaussian feature**; jitter **after the in-flight
     Gaussian-input session**. Effort: M + S + S.

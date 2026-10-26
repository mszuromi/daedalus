.. _concepts-action:

From a stochastic equation to the MSR-JD action
===============================================

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: the minimum theory to read a Daedalus result correctly.
   - Audience: users with an SDE background but no field theory; physicists for the notation.
   - Contents: response field and the action :math:`S[x,\tilde x]=\int \tilde x(\dot x - f(x)) - W(\tilde x;x)`; the noise
     as a conditional cumulant generating function (Gaussian, Poisson, Lévy; why every source in the library is one
     CGF); the Itô discretisation and why the Jacobian is trivial; point processes and jump processes in the same
     language; multiplicative noise and the interpretation corrections (Stratonovich, Marcus) as extra drift terms;
     what "the action derived by origin" in ``check()`` shows.
   - Sources: ``api/sde.py`` (action derivation), ``api/noise.py`` (CGFs), ``docs/noise_library.md``,
     ``docs/correlated_noise_capabilities.md``.
   - Examples: OU, OU with Poisson kicks, a Hawkes process: three actions side by side (from ``m.latex()``).
   - Auto: the action listings can be produced by ``m.latex()`` at build time (doctest). Status: now. Effort: M.

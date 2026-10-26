.. hand-written: the plan is kept below, and the doctest is real (it runs in CI).

.. _getting_started-first_calculation:

Your first calculation
======================

The stationary variance of the quartic Ornstein-Uhlenbeck process

.. math::

   \mathrm{d}x = (-\mu x - \varepsilon x^3)\,\mathrm{d}t + \sqrt{2D}\,\mathrm{d}W,

to one loop. The tree level is the linear (OU) variance :math:`D/\mu`; the one-loop correction is
:math:`-3\varepsilon D^2/\mu^3`, here :math:`-0.06`.

.. testcode::

   from api.sde import SDE
   from api.noise import Gaussian
   from api import compute_cumulants

   m = (SDE('OU quartic')
        .parameter('mu', default=1.0, domain='positive')
        .parameter('eps', default=0.02, domain='positive')
        .parameter('D', default=1.0, domain='positive')
        .field('x')
        .drift('x', '-mu*x - eps*x^3')
        .noise('xi', Gaussian(var='2*D'), couples={'x': 1}))
   res = compute_cumulants(m.build(), k=2, max_ell=1, external_fields=[('x', 1), ('x', 1)],
                           tau_grid=np.array([0.0]), verbose=False)
   print(res['status'])
   for ell, c in res['C_tau_by_ell'].items():
       print(ell, round(float(c[0].real), 12))

.. testoutput::

   ok
   0 1.0
   1 -0.06

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: one complete run in ten lines, then what each line did.
   - Audience: new users.
   - Contents: the OU quartic model with the SDE front end; ``m.check()`` and what the report means;
     ``compute_cumulants`` vs ``dd.run(model, dd.Config(...))``; tree, one loop, two loops; plotting; comparing
     with ``m.simulate`` (or the exact series :math:`-3\varepsilon D^2/\mu^3`); next steps.
   - Sources: README snippets, :tutorial:`00_quick_start`, :src:`models/ou_quartic.model.py`.
   - Examples: the doctested snippet above (tree and one loop, a few seconds); a figure from the example notebook.
   - Status: now. Effort: S.
   - Note: ``couples=`` becomes ``into=`` with :pr:`107`.

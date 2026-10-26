.. _cookbook-index:

Cookbook
========

.. admonition:: Depends on unmerged work
   :class: warning

   Parts of this chapter wait for: :pr:`35`; :pr:`102`; :pr:`106`.

.. todo::

   Cookbook: write the parts that wait for :pr:`35`; :pr:`102`; :pr:`106` once they merge.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   Each recipe: the question, the model (shortest front-end form), the run, the result (figure or
   table), the check (exact, oracle or simulation), pitfalls, links to the notebook. Doctested where under ~10 s;
   otherwise the code is pulled from an executed notebook (``literalinclude`` of a script that a slow test runs).


   .. list-table::
      :header-rows: 1

      * - #
        - recipe
        - draws from
        - status
      * - R1
        - OU quartic to two loops vs the exact series
        - tutorial 00, example ``OU_quartic_white``
        - now
      * - R2
        - A φ⁶ nonlinearity
        - ``OU_sextic_white``
        - now
      * - R3
        - Double well: choosing the saddle
        - ``models/ou_quartic_double_well``
        - now
      * - R4
        - Two fields with correlated coloured noise
        - ``OU_quartic_correlated_colored``
        - now; front-end form after the coloured-Gaussian feature
      * - R5
        - Multiplicative noise: Itô vs Stratonovich
        - README mixed noise
        - now
      * - R6
        - Lévy-driven OU vs exact cumulants
        - tutorial 02
        - now
      * - R7
        - Linear Hawkes: exact check
        - tutorial 03, ``models/linear_hawkes``
        - now
      * - R8
        - Nonlinear Hawkes with an alpha kernel vs simulation
        - ``Hawkes_quadratic_alpha``
        - now
      * - R9
        - Delayed feedback (delayed OU, delayed-alpha Hawkes)
        - tutorial 04, delay examples (:pr:`35`)
        - after :pr:`35` for links
      * - R10
        - Two-compartment neuron (a DAE)
        - ``Dendritic_nonlinear``
        - now
      * - R11
        - Reaction network vs the master equation
        - ``dd.master_equation``, ``dd.JumpProcess``
        - after :pr:`102`
      * - R12
        - Spectrum and correlation time of a network
        - spectra
        - now
      * - R13
        - Covariance matrix and sensitivity to a coupling
        - covariance
        - now
      * - R14
        - Stochastic Allen-Cahn in 1-D
        - ``Allen_Cahn_phi4``
        - now
      * - R15
        - Reaction-diffusion in 2-D and two species
        - ``RD_2D``, ``RD_2_species``
        - now
      * - R16
        - Strong coupling: loops vs resummation
        - :pr:`106` validation
        - after :pr:`106`


   Effort: S per recipe (M for R11, R15, R16); about 4 sessions in total.

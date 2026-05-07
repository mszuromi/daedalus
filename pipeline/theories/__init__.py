"""
Standardized theory files for the canonical MSR-JD pipeline examples.

Each module here exposes a single ``HAWKES_MODEL`` dict produced by
``pipeline.theory.TheoryBuilder`` + the ``pipeline.theory_templates``
templates.  The intent is twofold:

  1. Demonstrate the high-level builder API by reproducing every
     existing model file in one place, side-by-side.
  2. Serve as ground-truth examples — each generated dict is verified
     to produce the same FieldTheory.expand sectors as the
     hand-written model in ``models/``.

Modules
-------

  * linear_hawkes_2pop_delta  — phi(v)=v,        delta synapse
                                (≡ models/hawkes_linear_sage.py)
  * linear_hawkes_2pop_expg   — phi(v)=a·v,      exp(τ_g) synapse
                                (≡ models/hawkes_linear_expg.py)
  * quad_hawkes_2pop_expg     — phi(v)=a·v²,    exp(τ_g) synapse
                                (≡ models/hawkes_quad_expg.py)
  * linear_hawkes_2pop_expg_gtas
                              — phi(v)=a·v + GTaS external noise
                                (≡ models/hawkes_linear_expg_gtas.py)
  * quad_hawkes_2pop_expg_gtas
                              — phi(v)=a·v² + GTaS external noise
                                (≡ models/hawkes_quad_expg_gtas.py)

Usage::

    from pipeline.theories.linear_hawkes_2pop_expg_gtas import HAWKES_MODEL
    # ... feed to pipeline.compute_cumulants(...) as usual.
"""

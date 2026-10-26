.. hand-written: not generated from the outline.

Daedalus
========

Daedalus computes the statistics of stochastic models (Itô SDEs, SPDEs, point processes and jump processes) by
evaluating the Feynman diagrams of their MSR-JD field theory automatically. From a model written as equations, it
returns the mean field, connected cumulants of any order with their loop corrections, spectra and integrated
correlators, covariance matrices and their parameter derivatives, and symbolic forms where they exist. A model
analyzer checks each question before anything is computed, and a result either is complete, says what it left
out, or is refused.

This is the manual of version |release|.

.. admonition:: This manual is being written
   :class: warning

   The chapters below are stubs that hold the plan of each chapter. :doc:`status` lists the parts that wait for
   features that have not merged yet. Until a chapter is written, the
   `README <https://github.com/mszuromi/daedalus#readme>`_, the tutorials and the example notebooks are the
   documentation.

.. admonition:: Chapter plan (replace with the text)
   :class: note

   - Purpose: what the package computes, for which models, and what it does not do; the one-paragraph pitch.
   - Audience: everyone; first page of the site.
   - Contents: stationary (and switch-on) statistics of Itô SDEs, SPDEs, point and jump processes; the outputs; the
     trust model (analyzer gates, never-silent statuses, oracles); what it is not (not a simulator, not a
     non-perturbative solver unless resummation is asked for); version and citation box.
   - Sources: README pitch, ``pyproject.toml`` description, ``CITATION.cff``.
   - Examples: one figure, the OU quartic tree, one loop and two loops against simulation (from
     :example:`SDE/OU_quartic_white`).

.. toctree::
   :caption: Start here
   :maxdepth: 1

   front/how_to_read
   front/conventions_card
   getting_started/installation
   getting_started/first_calculation
   getting_started/documentation_map

.. toctree::
   :caption: Concepts
   :maxdepth: 1

   concepts/action
   concepts/saddle
   concepts/propagator
   concepts/diagrams
   concepts/asymptotics
   concepts/conventions

.. toctree::
   :caption: Describing a model
   :maxdepth: 1

   models/sde_front_end
   models/operators
   models/paper_syntax
   models/noise
   models/kernels_delays
   models/point_processes
   models/jump_processes
   models/populations
   models/action_by_hand
   models/model_files

.. toctree::
   :caption: Running and outputs
   :maxdepth: 1

   running/run_and_config
   running/results
   running/cumulants_moments
   running/spectra
   running/covariance
   running/symbolic
   running/switch_on
   running/resummation

.. toctree::
   :caption: Spatial models (SPDEs)
   :maxdepth: 1

   spatial/spde_models
   spatial/grids_outputs
   spatial/coupled_accuracy

.. toctree::
   :caption: Accuracy and trust
   :maxdepth: 1

   trust/analyzer
   trust/statuses
   trust/invariance
   trust/validation
   trust/scope

.. toctree::
   :caption: Performance
   :maxdepth: 1

   performance/caches
   performance/memory
   performance/parallelism
   performance/cost

.. toctree::
   :caption: Cookbook
   :maxdepth: 1

   cookbook/index

.. toctree::
   :caption: Reference
   :maxdepth: 1

   reference/api
   reference/config_options
   reference/yaml
   reference/environment
   reference/exceptions
   reference/analyzer_findings
   reference/troubleshooting
   reference/glossary
   reference/changelog
   reference/citing
   reference/developers
   status

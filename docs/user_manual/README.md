# The Daedalus user manual (Sphinx source)

Build it with SageMath's Python, from the repository root (SageMath 10.9 ships Sphinx, furo and sphinx-copybutton,
so nothing else needs installing; in the conda environment, `python` is Sage's and
`pip install -r docs/user_manual/requirements.txt` adds them if they are missing):

```bash
sage -python -m sphinx -W --keep-going -b html docs/user_manual docs/user_manual/_build/html
sage -python -m sphinx -W --keep-going -b doctest docs/user_manual docs/user_manual/_build/doctest
sage -python -m sphinx -M latexpdf docs/user_manual docs/user_manual/_build -W --keep-going   # PDF, needs xelatex
```

The `docs` workflow runs the first two on every pull request that touches the manual or the package (warnings fail
the build) and the PDF on `v*` tags and on demand.

## Writing a chapter

- Each chapter is one `.rst` file. A stub holds the chapter's plan in a "Chapter plan" box: write the text, then
  delete the box. A "Depends on unmerged work" box (and its `.. todo::`, collected on the status page) stays until
  the feature it names has merged and its part is written.
- Code examples are tests: use `.. testcode::` / `.. testoutput::` (or `.. doctest::`), keep each under about 10 s
  (`max_ell=1` at most, small grids), and pass `verbose=False`. An example too slow for CI is taken from a file a
  test or a notebook already runs, and says so.
- Refer to code by its name, never by a line number: `:func:`api.spectra.spectra``, `:class:`daedalus.Config``,
  or a file with `:src:`api/noise.py``. Link a tutorial with `:tutorial:`00_quick_start``, an example notebook with
  `:example:`SDE/OU_quartic_white``, a pull request with `:pr:`106``.
- Every number in the text comes from a doctest, a notebook or a test that the text names.
- The manual explains and specifies; the tutorials and example notebooks demonstrate. Link them instead of copying.
- Style: model-agnostic framing (Hawkes and spiking examples are fine); only models that are in the repository;
  no internal ticket numbers.

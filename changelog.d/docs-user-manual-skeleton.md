#### User manual skeleton (Sphinx)

`docs/user_manual/` is the source of a new user manual, built with Sphinx (HTML, and a PDF through xelatex) under
SageMath, with the API reference generated from the docstrings and executed examples. Every chapter is a stub that
holds its plan; a status page lists the parts that wait for unmerged features. A `docs` workflow builds the HTML and
runs the manual's doctests on every pull request that touches the manual or the package, failing on any warning,
and builds the PDF on release tags. No code changes; no numbers move.

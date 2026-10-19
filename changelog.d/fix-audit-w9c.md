#### Public text points at things that exist; no internal ticket numbers in library comments (audit wave W9, part 3)

Text only: no code path, number or API changes.

- **Dangling references fixed (CON-05).** About 25 docs and code sites cited design documents, spike scripts and
  example notebooks that are not in the repository; they now say "the internal design notes (not distributed)" (or
  point at `CHANGELOG.md`). The design docs under `docs/` use the current package names (`engine/`, `api/`,
  `ModelBuilder`, `tests/oracles/symbolic.py`, retired modules marked as retired) instead of `msrjd/`, `pipeline/` and
  `TheoryBuilder`. Docstrings that described behaviour the code no longer has are corrected: the time-domain
  `pipeline` module no longer says loops are skipped, and `SymbolicCumulant.callable` no longer promises Horner forms.
  Simulator docstrings no longer name git-ignored local model files.
- **Internal ticket numbers removed from comments and docstrings (CON-08).** About 125 "item N" / "bug N" references
  in `api/`, `engine/`, `daedalus.py` and `models/` are gone (the sentence keeps its content). `CHANGELOG.md` is left
  as written (fragments are collated at release).
- `tests/test_audit_w9_docs.py` fails when a tracked file cites a missing `docs/` file or old package name, or when a
  library comment gets a ticket number again.

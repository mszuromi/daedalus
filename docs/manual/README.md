# The Daedalus Code Manual

A book-length, pedagogical manual for the Daedalus automated MSR-JD
Feynman-diagram pipeline. It documents every stage of the pipeline — from the
text you type to describe a model, through the symbolic algebra, diagram
enumeration, and loop integration, to the cumulants and plots that come out —
for a reader who knows the physics but is new to the tooling (SageMath, `nauty`,
`numba`, `scipy`).

## Files

| Path | What it is |
|------|------------|
| `daedalus_manual.tex` | The **master** file. Compile this. It `\input`s every chapter. |
| `sections/*.tex` | One file per chapter (numbered) plus the appendices (`A`–`D`). |
| `_briefs/*.md` | Per-subsystem technical briefs used as drafting source material. Not part of the compiled book; kept for reference and future updates. |
| `AUDIT_FINDINGS.md` | Discrepancies found while cross-checking each chapter against the code. Mirrored into Appendix D of the book. |

## Compiling

### On Overleaf (recommended for viewing)

1. Upload the **whole `docs/manual/` folder** to a new Overleaf project (drag the
   folder in, or upload a zip of it).
2. Set the main document to `daedalus_manual.tex` (Menu → Main document).
3. Compile. Everything used is standard CTAN (no custom packages): `report`,
   `amsmath`, `tcolorbox`, `tikz`, `listings`, `hyperref`, `booktabs`, `geometry`.

### Locally

```sh
cd docs/manual
pdflatex -interaction=nonstopmode daedalus_manual.tex
pdflatex -interaction=nonstopmode daedalus_manual.tex   # second pass: ToC + cross-refs
```

(Or `latexmk -pdf daedalus_manual.tex`, which runs the passes for you.)

## Editing

Each chapter is independent. To revise one, edit its `sections/NN-slug.tex` and
recompile the master. The custom macros and callout boxes (`\code`, `\file`,
`\term`, `note`/`gotcha`/`defn` boxes, the math shorthands) are all defined in
the preamble of `daedalus_manual.tex`.

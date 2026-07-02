# CascadeMind Paper Source

The canonical camera-ready source is:

- `latest-paperfeb12026/semeval2026_final.tex`
- `latest-paperfeb12026/references.bib`
- `latest-paperfeb12026/acl.sty`
- `latest-paperfeb12026/acl_natbib.bst`

Generated PDFs, logs, arXiv zips, and START submission bundles are intentionally not tracked in this repository.

## Build

Use a clean LaTeX installation from the canonical source directory:

```bash
cd paper/latest-paperfeb12026
latexmk -pdf -interaction=nonstopmode -halt-on-error semeval2026_final.tex
```

Manual fallback:

```bash
cd paper/latest-paperfeb12026
pdflatex semeval2026_final.tex
bibtex semeval2026_final
pdflatex semeval2026_final.tex
pdflatex semeval2026_final.tex
```

## Camera-Ready Facts

- Official Track A result: `72.75%`.
- Listed Track A rank in the organizer overview table: `10th`.
- Track A development triples: `200`.
- Track A test triples: `400`.
- Synthetic training triples used for symbolic-weight calibration: `1900`.
- Public code URL: `https://github.com/epoch-learn/CascadeMind`.

Post-hoc reruns and archived local result files should stay labeled as diagnostic analysis. They do not change the official shared-task standing.

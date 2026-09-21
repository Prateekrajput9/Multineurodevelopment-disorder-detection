# MSE report

## Compile

Upload `mse_report_overleaf.zip` to Overleaf (New Project → Upload Project) and
compile `main.tex` with pdfLaTeX. No BibTeX step is needed; the references are
inline.

Locally: `pdflatex main.tex` twice (the second pass resolves references).

## Where the numbers come from

Every result in `main.tex` is a macro defined in `generated/numbers.tex`, and
every results table is in `generated/tab_*.tex`. Both are written by

    python report/build_tables.py

from `results/stage1/results_summary.json`. If you re-run any experiment,
re-run that script and recompile; do not type numbers into `main.tex`.

`python report/check_report.py` checks that every macro and citation resolves,
that environments balance, and lists every digit typed by hand in the text.

## Placeholders to fill

- Title page (IIT Indore template layout): `[Discipline]`, `[Student Name]`,
  `[roll no.]`, `[Prof. Supervisor Name]`, `[Designation, Department]`,
  co-supervisor, and `[Department Name]` at the bottom. The template's
  default discipline was Electrical Engineering; use yours.
- None in the timeline: it runs from August 2026, through the mid-semester
  evaluation on 25 September, to the final evaluation and submission at the
  end of November.

## If the PDF is longer than 7 pages (excluding the title page)

Cut in this order:
1. Remove Table 4 (embedding statistics); the key numbers are already in the
   text beside the t-SNE figure.
2. Reduce the t-SNE figure from `0.66\textwidth` to `0.55\textwidth`.
3. Shorten the "Decomposition methods" paragraph of Section 3 to two sentences.

## References

Only three: the replicated paper and the two dataset sources. Prior studies
in the literature survey are named by author and year and cited as reviewed
in the replicated paper [1].

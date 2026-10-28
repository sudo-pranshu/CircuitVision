# Report source

The submitted mini-project report (Word, institute template format) is generated from these files:

- `content.js`: all report text, tables, figure and citation keys. **Edit text here.** Real CGHD results are marked `[[PENDING: real CGHD run]]`; replace them only with numbers from the baseline_v1 evidence files (`test/metrics.json`, `downstream/summary.json`, `tables.md`).
- `build.js`: docx layout (A4, Times New Roman 12 pt, 1.5 spacing, justified, margins 1.25/1/1/1 in, Roman then Arabic page numbers, numbered captions, references in citation order).
- `finish.py`: inserts real Word TOC / list-of-figures / list-of-tables fields and computes their page numbers from a PDF render.
- `make_figs.py`, `fig/`: figures, generated from the code and `results/`.

Build: `bash report/build.sh` (needs `npm i -g docx`, LibreOffice and poppler). The output `.docx`/`.pdf` are not committed.
In Word, after editing, update the contents pages with Ctrl+A then F9.

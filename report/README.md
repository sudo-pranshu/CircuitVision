# Report source

The submitted mini-project report (Word, institute template format) is generated from these files:

- `content.js`: all report text, tables, figure and citation keys. **Edit text here.** Real CGHD results come only from the baseline_v1 evidence files, copied to `results/baseline_v1/`.
- `build.js`: docx layout (A4, Times New Roman 12 pt, 1.5 spacing, justified, 0.75 in margins on all sides, Roman then Arabic page numbers, numbered captions, references in citation order).
- `finish.py`: inserts real Word TOC / list-of-figures / list-of-tables fields and computes their page numbers from a PDF render.
- `make_figs.py`, `fig/`: figures, generated from the code and `results/`.

Build: `bash report/build.sh` (needs `npm i -g docx`, LibreOffice and poppler). The final `CircuitVision_Mini_Project_Report.docx` and `.pdf` (46 pages, 2026-11-04) are committed as the submission deliverable; rebuild them after any text change.
In Word, after editing, update the contents pages with Ctrl+A then F9.

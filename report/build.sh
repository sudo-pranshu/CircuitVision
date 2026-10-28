#!/usr/bin/env bash
# Build report/CircuitVision_Mini_Project_Report.docx with page numbers resolved from a PDF render.
# Needs: node + docx (npm), LibreOffice (soffice), poppler (pdftotext, pdfinfo).
set -euo pipefail
cd "$(dirname "$0")"
OUT=CircuitVision_Mini_Project_Report
[ -s pages.json ] || echo "{}" > pages.json
for pass in 1 2 3; do
  node build.js raw.docx pages.json
  python3 finish.py fields raw.docx "$OUT.docx"
  soffice --headless --convert-to pdf "$OUT.docx" >/dev/null 2>&1
  python3 finish.py pages "$OUT.pdf" raw.docx.entries.json pages_new.json
  if cmp -s pages_new.json pages.json; then echo "page numbers stable after pass $pass"; break; fi
  mv pages_new.json pages.json
done
rm -f raw.docx raw.docx.entries.json pages_new.json

#!/usr/bin/env bash
# Render docs/task4_report.html to a PDF using headless Chrome (Task 4 report,
# no page limit). Mirrors scripts/make_proposal_pdf.sh's render/verify pattern:
# no pandoc/LaTeX/wkhtmltopdf/weasyprint available on this machine, so headless
# Chrome is the only PDF route. `mdls` is non-functional here (Spotlight isn't
# indexing), so page count falls back to the macOS-native `file` utility.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_PATH="${ROOT_DIR}/docs/task4_report.html"
PDF_PATH="${ROOT_DIR}/submissions/Task4_Spatial_Filtering.pdf"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

render() {
  local extra_flag="${1:-}"
  "${CHROME}" \
    ${extra_flag} \
    --headless --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=5000 \
    --print-to-pdf="${PDF_PATH}" \
    "file://${HTML_PATH}"
}

echo "Rendering ${HTML_PATH} -> ${PDF_PATH}"
render ""

if [ ! -s "${PDF_PATH}" ]; then
  echo "PDF missing or empty after default headless render, retrying with --headless=old"
  render "--headless=old"
fi

MDLS_OUT="$(mdls -name kMDItemNumberOfPages "${PDF_PATH}" 2>&1 || true)"
echo "mdls output: ${MDLS_OUT}"
PAGE_COUNT="$(echo "${MDLS_OUT}" | grep -o '[0-9]\+' || true)"

if [ -z "${PAGE_COUNT}" ]; then
  # mdls returns "could not find <path>" on this machine even right after write
  # (Spotlight/mdworker not indexing this volume/path) -- fall back to the
  # macOS-native BSD `file` utility, which parses the PDF trailer directly.
  echo "mdls unavailable/non-functional here, falling back to 'file' utility."
  FILE_OUT="$(file "${PDF_PATH}")"
  echo "file output: ${FILE_OUT}"
  PAGE_COUNT="$(echo "${FILE_OUT}" | grep -o '[0-9]\+ pages\?' | grep -o '[0-9]\+' || true)"
fi

if [ -z "${PAGE_COUNT}" ]; then
  # `file`'s libmagic PDF rule only scans a limited byte prefix and misses the
  # page count on larger, multi-object Chrome-generated PDFs (observed here).
  # Fall back to counting `/Type /Page` object headers directly via a small
  # regex scan -- no pypdf/PyMuPDF installed, so plain byte-level regex it is.
  echo "file utility gave no page count either, falling back to a raw PDF object scan."
  PAGE_COUNT="$("${ROOT_DIR}/.venv/bin/python" -c "
import re
data = open('${PDF_PATH}', 'rb').read()
print(len(re.findall(rb'/Type\s*/Page[^s]', data)))
")"
fi

if [ -z "${PAGE_COUNT}" ] || [ "${PAGE_COUNT}" -eq 0 ]; then
  echo "ERROR: could not determine page count via mdls, file, or raw object scan." >&2
  exit 1
fi

echo "OK: PDF is ${PAGE_COUNT} page(s). No page limit for this report (unlike the 2-page proposal)."

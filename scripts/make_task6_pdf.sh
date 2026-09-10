#!/usr/bin/env bash
# Render docs/task6_report.html to a PDF using headless Chrome (Task 6 report,
# institutional lab-template format). Mirrors scripts/make_task3_pdf.sh's
# render/verify pattern: no pandoc/LaTeX/wkhtmltopdf/weasyprint available on
# this machine, so headless Chrome is the only PDF route.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_PATH="${ROOT_DIR}/docs/task6_report.html"
PDF_PATH="${ROOT_DIR}/docs/task6_report.pdf"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

render() {
  local extra_flag="${1:-}"
  "${CHROME}" \
    ${extra_flag} \
    --headless --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=8000 \
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
  echo "mdls unavailable/non-functional here, falling back to 'file' utility."
  FILE_OUT="$(file "${PDF_PATH}")"
  echo "file output: ${FILE_OUT}"
  PAGE_COUNT="$(echo "${FILE_OUT}" | grep -o '[0-9]\+ pages\?' | grep -o '[0-9]\+' || true)"
fi

if [ -z "${PAGE_COUNT}" ]; then
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

echo "OK: PDF is ${PAGE_COUNT} page(s)."

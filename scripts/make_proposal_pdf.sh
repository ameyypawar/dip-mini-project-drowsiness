#!/usr/bin/env bash
# Render docs/proposal/proposal.html to a 2-page-max PDF using headless Chrome.
# No pandoc/LaTeX/wkhtmltopdf/weasyprint available on this machine — headless
# Chrome is the only PDF route. Verifies page count via macOS-native `mdls`
# (no pypdf/PyMuPDF installed).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_PATH="${ROOT_DIR}/docs/proposal/proposal.html"
PDF_PATH="${ROOT_DIR}/docs/proposal/DIP_Proposal_Drowsiness_Detection.pdf"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MAX_PAGES=2

render() {
  local extra_flag="${1:-}"
  "${CHROME}" \
    ${extra_flag} \
    --headless --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=3000 \
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
  echo "ERROR: could not determine page count via mdls or file." >&2
  exit 1
fi

echo "Detected page count: ${PAGE_COUNT}"

if [ "${PAGE_COUNT}" -gt "${MAX_PAGES}" ]; then
  echo "FAIL: PDF has ${PAGE_COUNT} pages, exceeds limit of ${MAX_PAGES}." >&2
  exit 1
fi

echo "OK: PDF is ${PAGE_COUNT} page(s), within limit of ${MAX_PAGES}."

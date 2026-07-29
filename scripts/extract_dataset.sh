#!/usr/bin/env bash
#
# extract_dataset.sh
#
# Extracts the MRL Eye Dataset archive into data/raw/.
#
# Source:  https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip
# Size:    ~342 MB compressed, 84,938 zip entries, 37 subject dirs (s0001-s0037)
#
# Idempotent / resumable: uses `unzip -n` (never overwrite existing files),
# so re-running after a partial extraction only extracts what's missing.
#
# Usage: scripts/extract_dataset.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ZIP_PATH="${REPO_ROOT}/data/raw/mrlEyes_2018_01.zip"
DEST_DIR="${REPO_ROOT}/data/raw"

if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "ERROR: zip not found at ${ZIP_PATH}" >&2
    echo "Download it first from https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip" >&2
    exit 1
fi

# --- Pre-flight free-space guard ---
# Remaining unextracted portion is roughly 157 MB; require 500 MB free to be safe.
REQUIRED_KB=$((500 * 1024))
AVAIL_KB="$(df -Pk "${DEST_DIR}" | awk 'NR==2 {print $4}')"
if [[ "${AVAIL_KB}" -lt "${REQUIRED_KB}" ]]; then
    echo "ERROR: insufficient free space in ${DEST_DIR}: ${AVAIL_KB} KB available, ${REQUIRED_KB} KB required." >&2
    exit 1
fi
echo "Free space check OK: ${AVAIL_KB} KB available at ${DEST_DIR}"

# --- Idempotent resume extraction ---
echo "Extracting ${ZIP_PATH} -> ${DEST_DIR} (skipping already-extracted files)..."
unzip -n "${ZIP_PATH}" -d "${DEST_DIR}"

# --- Post-check ---
DATASET_DIR="${DEST_DIR}/mrlEyes_2018_01"
SUBJECT_COUNT="$(find "${DATASET_DIR}" -mindepth 1 -maxdepth 1 -type d -name 's0*' | wc -l | tr -d ' ')"
PNG_COUNT="$(find "${DATASET_DIR}" -type f -name '*.png' | wc -l | tr -d ' ')"

echo "Subject directories found: ${SUBJECT_COUNT}"
echo "Total PNG files found:     ${PNG_COUNT}"

if [[ "${SUBJECT_COUNT}" -ne 37 ]]; then
    echo "ERROR: expected 37 subject directories (s0001-s0037), found ${SUBJECT_COUNT}." >&2
    exit 1
fi

echo "Extraction complete: 37/37 subject directories present, ${PNG_COUNT} PNG files."

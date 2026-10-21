#!/usr/bin/env bash
# Serve the project dashboard on localhost so the browser will grant camera access.
# Camera APIs are refused on file:// URLs, which is why this exists.
set -euo pipefail
PORT="${1:-8777}"
cd "$(dirname "$0")"
URL="http://localhost:${PORT}/dashboard.html"
echo "Dashboard: ${URL}"
echo "Press Ctrl-C to stop."
( sleep 1; command -v open >/dev/null && open "$URL" ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1

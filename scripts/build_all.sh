#!/usr/bin/env bash
# Rebuild one or all 7 volumes of "Intelligence from THE BOOK" via 2-pass XeLaTeX.
# Usage:
#   ./scripts/build_all.sh           # rebuild all 7 volumes
#   ./scripts/build_all.sh vol4 vol5 # rebuild selected volumes

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XELATEX="${XELATEX:-xelatex}"

if ! command -v "$XELATEX" >/dev/null 2>&1; then
  if [ -x /Library/TeX/texbin/xelatex ]; then
    XELATEX=/Library/TeX/texbin/xelatex
  else
    echo "Error: xelatex not found. Please install TeX Live (with ctex / Fandol fonts)." >&2
    exit 1
  fi
fi

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=(vol1 vol2 vol3 vol4 vol5 vol6 vol7)
fi

for v in "${TARGETS[@]}"; do
  VDIR="$ROOT_DIR/volumes/$v"
  if [ ! -f "$VDIR/main.tex" ]; then
    echo "Error: $VDIR/main.tex not found." >&2
    exit 1
  fi
  echo "==> Building $v (2-pass XeLaTeX)..."
  (
    cd "$VDIR"
    "$XELATEX" -interaction=nonstopmode main.tex >/dev/null
    "$XELATEX" -interaction=nonstopmode main.tex >/dev/null
  )
  OUT_PDF=$(ls "$ROOT_DIR/pdfs/Vol0${v#vol}-"*.pdf 2>/dev/null | head -n 1 || true)
  if [ -n "$OUT_PDF" ]; then
    cp "$VDIR/main.pdf" "$OUT_PDF"
    echo "    Updated: $(basename "$OUT_PDF")"
  else
    echo "    Built: $VDIR/main.pdf"
  fi
done

echo "Done."

#!/usr/bin/env bash
# Integrate the paper website into an existing clone while retaining all science code.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-}"
if [[ -z "$DEST" || ! -d "$DEST" ]]; then
  echo 'Usage: bash scripts/install_in_repo.sh /path/to/cloned/PHORA' >&2; exit 2
fi
mkdir -p "$DEST/docs/assets" "$DEST/.github/workflows"
cp "$ROOT/docs/index.html" "$DEST/docs/index.html"
cp "$ROOT/docs/.nojekyll" "$DEST/docs/.nojekyll"
cp "$ROOT/docs/METHODS_AND_DATA.md" "$DEST/docs/METHODS_AND_DATA.md"
cp "$ROOT/docs/CLINICAL_VISUALIZATION_METHODS.md" "$DEST/docs/CLINICAL_VISUALIZATION_METHODS.md"
cp "$ROOT/REPRODUCIBILITY_AUDIT.md" "$DEST/docs/REPRODUCIBILITY_AUDIT.md"
cp "$ROOT/docs/assets/site.css" "$DEST/docs/assets/site.css"
cp "$ROOT/docs/assets/site.js" "$DEST/docs/assets/site.js"
cp "$ROOT/docs/assets/case_data.js" "$DEST/docs/assets/case_data.js"
cp "$ROOT/docs/assets/case_data.json" "$DEST/docs/assets/case_data.json"
mkdir -p "$DEST/scripts"
cp "$ROOT/scripts/add_signed_maps.py" "$DEST/scripts/add_signed_maps.py"
cp -R "$ROOT/docs/assets/cases" "$DEST/docs/assets/"
cp "$ROOT/.github/workflows/research-website.yml" "$DEST/.github/workflows/research-website.yml"
echo 'PHORA research website integrated without overwriting scientific source files.'
echo 'Review git diff, commit the changes and push the repository.'

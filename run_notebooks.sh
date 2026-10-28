#!/usr/bin/env bash
# Execute the notebooks and save the results as .ipynb files in executed/.
# Usage: ./run_notebooks.sh                      # all notebooks
#        ./run_notebooks.sh notebooks/quickstart.md
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p executed
notebooks=("$@")
[ ${#notebooks[@]} -eq 0 ] && notebooks=(notebooks/*.md)

failed=()
for nb in "${notebooks[@]}"; do
    out="executed/$(basename "${nb%.md}").ipynb"
    echo "=== Running $nb -> $out"
    if ! jupytext --to ipynb --execute --output "$out" "$nb"; then
        failed+=("$nb")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}"
    exit 1
fi
echo "All notebooks ran successfully."

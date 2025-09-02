#!/bin/bash

set -e  # exit on any error

for notebook in notebooks/*.ipynb; do
  echo "Processing $notebook..."
  tmp=$(mktemp)

  jq '(.cells[] | select(.cell_type == "code") | .source) |= map(select(. != "%%capture\n"))' \
    "$notebook" > "$tmp"

  jupyter nbconvert --to script "$tmp" --output "$(basename "$notebook" .ipynb)" --output-dir notebooks

  rm "$tmp"
done

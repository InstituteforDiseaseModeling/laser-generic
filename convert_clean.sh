#!/bin/bash

# This script does an nbconvert on each ipynb file but just before that it has to remove
# all the %%capture lines because this cause nbconvert to skip those cells and there's
# no other nice way to convert those cells also.
set -e  # exit on any error

for notebook in notebooks/*.ipynb; do
  echo "Processing $notebook..."
  tmp=$(mktemp)

  jq '(.cells[] | select(.cell_type == "code") | .source) |= map(select(. != "%%capture\n"))' \
    "$notebook" > "$tmp"

  jupyter nbconvert --to script "$tmp" --output "$(basename "$notebook" .ipynb)" --output-dir notebooks

  rm "$tmp"
done

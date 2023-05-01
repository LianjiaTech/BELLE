#!/bin/bash

# Check if the input file is given as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 input_file.gz"
  exit 1
fi

# Check if the input file exists and is readable
if [ ! -f "$1" ] || [ ! -r "$1" ]; then
  echo "Invalid input file: $1"
  exit 2
fi

# Check if jq and zstd are installed
if ! command -v jq &> /dev/null; then
  echo "jq is not installed. Please install it first."
  exit 3
fi

if ! command -v zstd &> /dev/null; then
  echo "zstd is not installed. Please install it first."
  exit 4
fi

# Get the base name of the input file without extension
base_name=${1%.gz}

# Create a temporary file for filtered output
temp_file=$(mktemp)

# Filter out the entries with wiki_prob < 0.25 using jq
gunzip -c "$1" | jq -c 'select(.wiki_prob >= 0.25)' > "$temp_file"

# Compress the output file using zstd with .jsonl.zst extension
zstd -c "$temp_file" > "$base_name.jsonl.zst"

# Remove the temporary file
rm "$temp_file"

# Print a success message
echo "Done. Output file: $base_name.jsonl.zst"

# to run
# for file in $(ls /data/common_crawl/cc_classifier/2023-06/*.gz); do bash cc_classifier.sh "$file" & done
#!/bin/bash

json_dir="polyhaven_json"
png_dir="polyhaven_png_filter"

for png_file in "$png_dir"/*.png; do
    base_name=$(basename "$png_file" .png)
    json_file="$json_dir/$base_name.json"

    if [ ! -f "$json_file" ]; then
        echo "Deleting $png_file (No JSON found)"
        rm "$png_file"
    fi
done

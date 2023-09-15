#!/bin/bash

source_dir="polyhaven"
destination_dir="polyhaven_json"

mkdir -p "$destination_dir"

for folder in "$source_dir"/*; do
    if [ -d "$folder" ]; then
        for hdr_file in "$folder"/*.json; do
            if [ -f "$hdr_file" ]; then
                base_name=$(basename "${hdr_file%.json}")
                cp "$hdr_file" "$destination_dir/$base_name.json"
                echo "Copied $hdr_file to $destination_dir"
            fi
        done
    fi
done

echo "Copying HDR files complete."


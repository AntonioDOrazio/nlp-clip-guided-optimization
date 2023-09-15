#!/bin/bash

source_dir="polyhaven"
destination_dir="polyhaven_png"

mkdir -p "$destination_dir"

for folder in "$source_dir"/*; do
    if [ -d "$folder" ]; then
        for hdr_file in "$folder"/*.hdr; do
            if [ -f "$hdr_file" ]; then
                base_name=$(basename "${hdr_file%.hdr}")

                pfsin "$hdr_file" | pfstmo_reinhard05 | pfsgamma -g 2.2 | pfsout "$destination_dir/$base_name.png"

                echo "Converted $hdr_file to PNG"
            fi
        done
    fi
done

echo "Conversion and copying complete."

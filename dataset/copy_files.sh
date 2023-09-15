#!/bin/bash

source_dir="polyhaven"
destination_dir="polyhaven_hdr"

mkdir -p "$destination_dir"

for folder in "$source_dir"/*; do
    if [ -d "$folder" ]; then
        for hdr_file in "$folder"/*.hdr; do
            if [ -f "$hdr_file" ]; then
                base_name=$(basename "${hdr_file%.hdr}")

                cp "$hdr_file" "$destination_dir/$base_name.hdr"

                echo "Copied $hdr_file to $destination_dir"
            fi
        done
    fi
done

echo "Copying HDR files complete."


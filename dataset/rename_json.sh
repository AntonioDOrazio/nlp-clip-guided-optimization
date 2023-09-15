#!/bin/bash

main_folder="polyhaven"

for subfolder in "$main_folder"/*; do
    if [ -d "$subfolder" ]; then
        hdr_file=$(find "$subfolder" -maxdepth 1 -type f -name "*.hdr" | head -n 1)
        if [ -n "$hdr_file" ]; then
            hdr_filename=$(basename "$hdr_file" .hdr)

            mv "$subfolder/info.json" "$subfolder/$hdr_filename.json"

            echo "Renamed $subfolder/info.json to $subfolder/$hdr_filename.json"
        else
            echo "No .hdr file found in $subfolder"
        fi
    fi
done

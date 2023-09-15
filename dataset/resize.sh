#!/bin/bash

source_dir="polyhaven_png"
destination_dir="polyhaven_png_128"

mkdir -p "$destination_dir"

for png_file in "$source_dir"/*.png; do
    if [ -f "$png_file" ]; then
        base_name=$(basename "${png_file%.png}")

        convert "$png_file" -resize 256x128 "$destination_dir/$base_name.png"

        echo "Resized $png_file and saved to $destination_dir"
    fi
done

echo "Resizing complete."


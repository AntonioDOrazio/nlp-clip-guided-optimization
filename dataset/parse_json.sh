#!/bin/bash

input_folder="polyhaven_json"
output_file="polyhaven.json"
#!/bin/bash

result_data=()

for filename in "$input_folder"/*.json; do
    name=$(basename "$filename" | cut -d. -f1)

    tags=$(jq -r '.tags // []' "$filename")
    categories=$(jq -r '.categories // []' "$filename")

    tags_caption=$(jq -r '.tags // [] | map("\"" + . + "\"") | join(",")' "$filename")
    categories_caption=$(jq -r '.categories // [] | map("\"" + . + "\"") | join(",")' "$filename")

    caption="[${tags_caption},${categories_caption}]"
    result_data+=("{\"name\":\"$name\",\"tags\":$tags,\"categories\":$categories,\"caption\":$caption},")
done

output_data="[${result_data[@]}]"
result="${output_data%,*}${output_data##*,}"

echo "$result" > "$output_file"

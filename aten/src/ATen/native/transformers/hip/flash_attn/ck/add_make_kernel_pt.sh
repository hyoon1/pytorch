#!/bin/bash

# Check if the input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file_list.txt>"
    exit 1
fi

# Assign the input file to a variable
file_list=$1

# Check if the file exists
if [ ! -f "$file_list" ]; then
    echo "Error: File '$file_list' not found!"
    exit 1
fi

# Loop through each line in the file list
while IFS= read -r file; do
    # Check if the file exists in the current directory
    if [ -f "$file" ]; then
        file_name=$(basename "$file")
        # Match FlashAttention RDNA generated blobs: keep the global default for
        # other blobs, but use standard-CNaN conversion on gfx11/gfx12 instances.
        if [[ "$file_name" =~ ^fmha_(fwd|bwd).*_gfx1[12][^/]*\.cpp$ ]]; then
            tmp_file=$(mktemp)
            {
                echo "#include \"ck_tile/core/config.hpp\""
                echo "#undef CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT"
                echo "#ifdef CK_TILE_FLOAT_TO_BFLOAT16_STANDARD_CNAN"
                echo "#define CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT CK_TILE_FLOAT_TO_BFLOAT16_STANDARD_CNAN"
                echo "#else"
                echo "#define CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT CK_TILE_FLOAT_TO_BFLOAT16_STANDARD"
                echo "#endif"
                cat "$file"
            } > "$tmp_file"
            mv "$tmp_file" "$file"
        fi
        # Use sed to replace "make_kernel" with "make_kernel_pt" in place
        sed -i 's/make_kernel/make_kernel_pt/g' "$file"
        sed -i 's/\#include \"fmha_fwd.hpp\"/\#include \"fmha_fwd.hpp\"\n\#include \"launch_kernel_pt.hpp\"/g' "$file"
        sed -i 's/\#include \"fmha_bwd.hpp\"/\#include \"fmha_bwd.hpp\"\n\#include \"launch_kernel_pt.hpp\"/g' "$file"
        echo "Updated: $file"
    else
        echo "Skipping: $file (not found)"
    fi
done < "$file_list"

echo "Replacement completed."

#!/bin/bash

# Check if the script has the correct number of arguments
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <wav|txt>"
    exit 1
fi

type=$1

# Validate file type argument
if [[ "$type" != "wav" && "$type" != "txt" ]]; then
    echo "Invalid file type. Use 'wav' or 'txt'."
    exit 1
fi

# Define the source directory
SRC_DIR="/Users/owenfriend/Desktop/movie_data"

# Define the destination directories based on the file type
if [[ "$type" == "wav" ]]; then
    DEST_DIR="/Users/owenfriend/Desktop/movie_data/audio"
elif [[ "$type" == "txt" ]]; then
    DEST_DIR="/Users/owenfriend/Desktop/movie_data/segmentation"
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Crawl through the directory and process each subject folder
for subject_dir in "$SRC_DIR"/pc_*; do
    if [[ -d "$subject_dir" ]]; then
        # Extract the subject ID (e.g., pc_056 -> 056)
        if [[ $subject_dir =~ pc_([0-9]{3}) ]]; then
            subject_id="${BASH_REMATCH[1]}"
        else
            continue
        fi

        # Get a sorted list of files based on type and sort in reverse order
        files=($(find "$subject_dir" -type f -name "*.${type}" | sort -r))

        # Process sorted files
        for file in "${files[@]}"; do
            # Check if the file is not empty
            if [[ ! -s "$file" ]]; then
                echo "Skipping empty file: $file"
                continue
            fi

            # Apply conditions based on file type
            if [[ "$type" == "txt" ]]; then
                if [[ "$file" == *"mot"* && "$file" == *"coin"* ]]; then
                    new_filename="${subject_id}_coin.${type}"
                elif [[ "$file" == *"mot"* && "$file" == *"jinx"* ]]; then
                    new_filename="${subject_id}_jinx.${type}"
                else
                    continue
                fi
            elif [[ "$type" == "wav" ]]; then
                if [[ "$file" == *"rec"* && "$file" == *"coin"* ]]; then
                    new_filename="${subject_id}_coin.${type}"
                elif [[ "$file" == *"rec"* && "$file" == *"jinx"* ]]; then
                    new_filename="${subject_id}_jinx.${type}"
                else
                    continue
                fi
            fi

            # Copy the file to the appropriate destination directory with the new name
            cp "$file" "$DEST_DIR/$new_filename"
            echo "Copied $file to $DEST_DIR/$new_filename"
        done
    fi
done

echo "All $type files have been processed and stored in $DEST_DIR."

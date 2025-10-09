#!/bin/bash

# build_search_index.sh
# Automated script to build search index from collection.tsv
# Usage: ./build_search_index.sh [max_docs] [block_size]

set -e  # Exit on any error

# Default parameters
MAX_DOCS=${1:-100}        # Default: 100 documents
BLOCK_SIZE=${2:-128}      # Default: 128 block size
TEMP_DIR="tmp_postings"   # Directory for intermediate files
INDEX_DIR="index_output"  # Directory for final index

echo "=== Search Index Builder ==="
echo "Max documents: $MAX_DOCS"
echo "Block size: $BLOCK_SIZE"
echo "Temp directory: $TEMP_DIR"
echo "Index directory: $INDEX_DIR"
echo "================================"

# Check if collection.tsv exists
if [ ! -f "collection.tsv" ]; then
    echo "Error: collection.tsv not found!"
    echo "Please ensure collection.tsv is in the current directory."
    exit 1
fi

# Step 1: Compile Parse.cpp if needed
echo ""
echo "Step 1: Compiling Parse.cpp..."
if [ ! -f "Parse" ] || [ "Parse.cpp" -nt "Parse" ]; then
    g++ -std=c++17 -O2 -o Parse Parse.cpp
    echo "✓ Parse compiled successfully"
else
    echo "✓ Parse executable is up to date"
fi

# Step 2: Clean up previous runs
echo ""
echo "Step 2: Cleaning up previous runs..."
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "✓ Removed old temp directory"
fi
if [ -d "$INDEX_DIR" ]; then
    rm -rf "$INDEX_DIR"
    echo "✓ Removed old index directory"
fi

# Step 3: Run Parse to create intermediate postings
echo ""
echo "Step 3: Running Parse on first $MAX_DOCS documents..."
echo "This may take a while for large collections..."
./Parse collection.tsv "$TEMP_DIR" 128 "$MAX_DOCS"

if [ $? -ne 0 ]; then
    echo "Error: Parse failed!"
    exit 1
fi

# Check if temp files were created
if [ ! -d "$TEMP_DIR" ] || [ -z "$(ls -A $TEMP_DIR/postings_*.tmp 2>/dev/null)" ]; then
    echo "Error: No posting files created in $TEMP_DIR"
    exit 1
fi

echo "✓ Parse completed successfully"
echo "  Created $(ls $TEMP_DIR/postings_*.tmp | wc -l) posting files"

# Step 4: Compile BuildIndex.cpp if needed
echo ""
echo "Step 4: Compiling BuildIndex.cpp..."
if [ ! -f "BuildIndex" ] || [ "BuildIndex.cpp" -nt "BuildIndex" ]; then
    g++ -std=c++17 -O2 -o BuildIndex BuildIndex.cpp
    echo "✓ BuildIndex compiled successfully"
else
    echo "✓ BuildIndex executable is up to date"
fi

# Step 5: Run BuildIndex to create final index
echo ""
echo "Step 5: Building final index..."
echo "This will merge and compress the posting files..."
./BuildIndex "$TEMP_DIR" "$INDEX_DIR" "$BLOCK_SIZE"

if [ $? -ne 0 ]; then
    echo "Error: BuildIndex failed!"
    exit 1
fi

# Check if index files were created
if [ ! -f "$INDEX_DIR/invlists.bin" ] || [ ! -f "$INDEX_DIR/lexicon.tsv" ]; then
    echo "Error: Index files not created properly"
    exit 1
fi

echo "✓ BuildIndex completed successfully"

# Step 6: Show results
echo ""
echo "=== BUILD COMPLETE ==="
echo "Index files created in: $INDEX_DIR/"
echo ""
echo "Index file sizes:"
ls -lh "$INDEX_DIR"/*.bin "$INDEX_DIR"/*.tsv 2>/dev/null | awk '{print "  " $9 ": " $5}'
echo ""
echo "Lexicon entries: $(wc -l < $INDEX_DIR/lexicon.tsv)"
echo "Documents processed: $MAX_DOCS"
echo ""
echo "You can now use the index for searching!"
echo "================================"

# Optional: Clean up temp files (uncomment if you want to save disk space)
# echo ""
# echo "Cleaning up temporary files..."
# rm -rf "$TEMP_DIR"
# echo "✓ Temporary files removed"



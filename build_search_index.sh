#!/bin/bash

# build_search_index.sh
# Automated script to build search index from collection.tsv
# Usage: ./build_search_index.sh [max_docs] [block_size]

set -e  # Exit on any error

# Colors for better logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters - processes ENTIRE collection by default
BLOCK_SIZE=${1:-128}          # Default: 128 block size
TEMP_DIR="tmp_postings"       # Directory for intermediate files
INDEX_DIR="index_output"      # Directory for final index

# Timestamp function
timestamp() {
    date "+[%Y-%m-%d %H:%M:%S]"
}

# Log functions
log_info() {
    echo -e "${BLUE}$(timestamp) INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}$(timestamp) SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}$(timestamp) WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}$(timestamp) ERROR:${NC} $1"
}

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          Search Engine Index Builder - Full Collection        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

log_info "Configuration:"
echo "  • Source: collection.tsv"
echo "  • Documents: ALL (entire collection)"
echo "  • Block size: $BLOCK_SIZE"
echo "  • Temp directory: $TEMP_DIR"
echo "  • Index directory: $INDEX_DIR"
echo ""

# Check if collection.tsv exists
log_info "Checking for collection.tsv..."
if [ ! -f "collection.tsv" ]; then
    log_error "collection.tsv not found!"
    echo "  Please ensure collection.tsv is in the current directory."
    exit 1
fi

# Get collection stats
COLLECTION_SIZE=$(du -h collection.tsv | cut -f1)
COLLECTION_LINES=$(wc -l < collection.tsv | tr -d ' ')
log_success "Found collection.tsv (${COLLECTION_SIZE}, ~${COLLECTION_LINES} documents)"
echo ""

# Step 0: Clean up previous runs FIRST
echo "────────────────────────────────────────────────────────────────"
log_info "Step 1/5: Cleaning up previous builds..."
echo "────────────────────────────────────────────────────────────────"

CLEANUP_COUNT=0
if [ -d "$TEMP_DIR" ]; then
    TEMP_SIZE=$(du -sh "$TEMP_DIR" 2>/dev/null | cut -f1)
    rm -rf "$TEMP_DIR"
    log_success "Removed old temp directory ($TEMP_SIZE)"
    ((CLEANUP_COUNT++))
fi

if [ -d "$INDEX_DIR" ]; then
    INDEX_SIZE=$(du -sh "$INDEX_DIR" 2>/dev/null | cut -f1)
    rm -rf "$INDEX_DIR"
    log_success "Removed old index directory ($INDEX_SIZE)"
    ((CLEANUP_COUNT++))
fi

if [ $CLEANUP_COUNT -eq 0 ]; then
    log_info "No previous builds to clean up"
else
    log_success "Cleanup complete - removed $CLEANUP_COUNT directories"
fi
echo ""

# Step 1: Compile Parse.cpp if needed
echo "────────────────────────────────────────────────────────────────"
log_info "Step 2/5: Compiling Parse.cpp..."
echo "────────────────────────────────────────────────────────────────"

if [ ! -f "Parse" ] || [ "Parse.cpp" -nt "Parse" ]; then
    log_info "Compiling Parse.cpp with optimizations..."
    if g++ -std=c++17 -O3 -Wall -Wextra -o Parse Parse.cpp 2>&1; then
        log_success "Parse compiled successfully"
    else
        log_error "Parse compilation failed!"
        exit 1
    fi
else
    log_success "Parse executable is up to date"
fi
echo ""

# Step 2: Run Parse to create intermediate postings
echo "────────────────────────────────────────────────────────────────"
log_info "Step 3/5: Parsing collection and creating intermediate postings..."
echo "────────────────────────────────────────────────────────────────"
log_warning "This may take several minutes for large collections..."
echo ""

START_TIME=$(date +%s)
./Parse collection.tsv "$TEMP_DIR" 128

PARSE_END_TIME=$(date +%s)
PARSE_DURATION=$((PARSE_END_TIME - START_TIME))

if [ $? -ne 0 ]; then
    log_error "Parse failed!"
    exit 1
fi

# Check if temp files were created
if [ ! -d "$TEMP_DIR" ] || [ -z "$(ls -A $TEMP_DIR/postings_*.tmp 2>/dev/null)" ]; then
    log_error "No posting files created in $TEMP_DIR"
    exit 1
fi

NUM_POSTINGS=$(ls $TEMP_DIR/postings_*.tmp 2>/dev/null | wc -l | tr -d ' ')
TEMP_SIZE=$(du -sh "$TEMP_DIR" 2>/dev/null | cut -f1)
log_success "Parse completed in ${PARSE_DURATION}s"
echo "  • Created $NUM_POSTINGS posting files"
echo "  • Temp storage used: $TEMP_SIZE"
echo ""

# Step 3: Compile BuildIndex.cpp if needed
echo "────────────────────────────────────────────────────────────────"
log_info "Step 4/5: Compiling BuildIndex.cpp..."
echo "────────────────────────────────────────────────────────────────"

if [ ! -f "BuildIndex" ] || [ "BuildIndex.cpp" -nt "BuildIndex" ]; then
    log_info "Compiling BuildIndex.cpp with optimizations..."
    if g++ -std=c++17 -O3 -Wall -Wextra -o BuildIndex BuildIndex.cpp 2>&1; then
        log_success "BuildIndex compiled successfully"
    else
        log_error "BuildIndex compilation failed!"
        exit 1
    fi
else
    log_success "BuildIndex executable is up to date"
fi
echo ""

# Step 4: Run BuildIndex to create final index
echo "────────────────────────────────────────────────────────────────"
log_info "Step 5/5: Building final compressed index..."
echo "────────────────────────────────────────────────────────────────"
log_warning "Merging and compressing $NUM_POSTINGS posting files..."
echo ""

BUILD_START_TIME=$(date +%s)
./BuildIndex "$TEMP_DIR" "$INDEX_DIR" "$BLOCK_SIZE"
BUILD_END_TIME=$(date +%s)
BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))

if [ $? -ne 0 ]; then
    log_error "BuildIndex failed!"
    exit 1
fi

# Check if index files were created
if [ ! -f "$INDEX_DIR/invlists.bin" ] || [ ! -f "$INDEX_DIR/lexicon.tsv" ]; then
    log_error "Index files not created properly"
    exit 1
fi

log_success "BuildIndex completed in ${BUILD_DURATION}s"
echo ""

# Calculate total time
TOTAL_TIME=$((BUILD_END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_SEC=$((TOTAL_TIME % 60))

# Show final results
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    BUILD COMPLETE ✓                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

log_success "Index successfully created in: $INDEX_DIR/"
echo ""

echo "📊 Index Statistics:"
echo "────────────────────────────────────────────────────────────────"
echo "  File Sizes:"
ls -lh "$INDEX_DIR"/*.bin "$INDEX_DIR"/*.tsv 2>/dev/null | awk '{printf "    %-20s %8s\n", $9, $5}'
echo ""

LEXICON_ENTRIES=$(($(wc -l < $INDEX_DIR/lexicon.tsv) - 1))
INDEX_SIZE=$(du -sh "$INDEX_DIR" 2>/dev/null | cut -f1)
echo "  • Unique terms (lexicon): $LEXICON_ENTRIES"
echo "  • Documents processed: $COLLECTION_LINES"
echo "  • Total index size: $INDEX_SIZE"
echo "  • Compression ratio: $(echo "scale=2; $(du -sb collection.tsv | cut -f1) / $(du -sb $INDEX_DIR | cut -f1)" | bc)x"
echo ""

echo "⏱️  Build Time:"
echo "────────────────────────────────────────────────────────────────"
echo "  • Parse phase: ${PARSE_DURATION}s"
echo "  • Build phase: ${BUILD_DURATION}s"
echo "  • Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""

log_success "Search index is ready! Use QueryProcessing to search."
echo ""
echo "Example queries:"
echo "  ./QueryProcessing test"
echo "  ./QueryProcessing --and document search"
echo "  ./QueryProcessing -k 20 information retrieval"
echo ""

# Optional: Clean up temp files
read -p "Clean up temporary posting files? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    log_success "Temporary files removed (freed $TEMP_SIZE)"
else
    log_info "Temporary files kept in $TEMP_DIR"
fi
echo ""



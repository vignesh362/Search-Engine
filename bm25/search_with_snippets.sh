#!/bin/bash
# search_with_snippets.sh
# Complete search pipeline: Query → Results → Snippets
#
# Usage: ./search_with_snippets.sh "query terms" [top-k]

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"query terms\" [top-k]"
    echo "Example: $0 \"university ranking\" 10"
    exit 1
fi

QUERY="$1"
TOP_K="${2:-10}"  # Default to 10 results

echo "=========================================="
echo "Search Query: $QUERY"
echo "Top Results: $TOP_K"
echo "=========================================="
echo ""

# Step 1: Run query and get document IDs
echo "Step 1: Finding relevant documents..."
RESULTS=$(./QueryProcessing "$QUERY" -k "$TOP_K")

if [ -z "$RESULTS" ]; then
    echo "No results found."
    exit 0
fi

echo "$RESULTS"
echo ""

# Step 2: Extract just the document IDs (from lines like " 1. doc=6233096 score=...")
DOC_IDS=$(echo "$RESULTS" | grep "doc=" | sed 's/.*doc=\([0-9]*\).*/\1/' | paste -sd "," -)

if [ -z "$DOC_IDS" ]; then
    echo "No document IDs found."
    exit 0
fi

echo "Step 2: Generating snippets for documents: $DOC_IDS"
echo ""
echo "=========================================="
echo "RESULTS WITH SNIPPETS:"
echo "=========================================="
echo ""

# Step 3: Generate snippets
./SnippetExtractor --store-dir index_output \
    --docs "$DOC_IDS" \
    --query "$QUERY" \
    --windowsz 50 \
    --per 2

echo ""
echo "=========================================="
echo "Search complete!"
echo "=========================================="


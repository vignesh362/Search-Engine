#!/bin/bash
# Quick verification script to check if all required files exist

echo "================================================"
echo "Assignment #3: Pre-flight Check"
echo "================================================"
echo ""

check_file() {
    if [ -f "$1" ]; then
        echo "✓ $1"
        return 0
    else
        echo "✗ $1 (MISSING)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "✓ $1/"
        return 0
    else
        echo "✗ $1/ (MISSING)"
        return 1
    fi
}

all_ok=0

echo "Required Scripts:"
check_file "run_all.py" || all_ok=1
check_file "generate_all_runs.py" || all_ok=1
check_file "evaluate_all_runs.py" || all_ok=1
check_file "eval/evaluate_cli.py" || all_ok=1

echo ""
echo "Documentation:"
check_file "EVALUATION_GUIDE.md" || all_ok=1
check_file "IMPLEMENTATION_SUMMARY.md" || all_ok=1

echo ""
echo "Required Data Files:"
check_file "data/ms_marco/qrels.dev.tsv" || all_ok=1
check_file "data/ms_marco/qrels.eval.one.tsv" || all_ok=1
check_file "data/ms_marco/qrels.eval.two.tsv" || all_ok=1
check_file "data/ms_marco/msmarco_queries_dev_eval_embeddings.h5" || all_ok=1
check_file "data/ms_marco/msmarco_passages_embeddings_subset.h5" || all_ok=1

echo ""
echo "Required Indexes:"
check_file "bm25/bm25_subset_index.pkl" || all_ok=1
check_file "dense/faiss_hnsw_index.bin" || all_ok=1
check_file "dense/passage_ids.pkl" || all_ok=1

echo ""
echo "Output Directories:"
check_dir "runs" || all_ok=1
check_dir "eval" || all_ok=1

echo ""
echo "================================================"
if [ $all_ok -eq 0 ]; then
    echo "✓ All checks passed! Ready to run."
    echo ""
    echo "Next step:"
    echo "  python run_all.py"
else
    echo "✗ Some files are missing. Please check above."
    echo ""
    echo "Missing indexes? Build them first:"
    echo "  python bm25/build_bm25_from_subset.py"
    echo "  python dense/build_faiss_hnsw.py"
fi
echo "================================================"

exit $all_ok

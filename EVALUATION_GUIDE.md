# Assignment #3: Search Engine Evaluation

This directory contains scripts to generate and evaluate three search systems on MS MARCO data:
1. **BM25** - Traditional term-based ranking
2. **Dense (HNSW)** - Neural vector search with FAISS
3. **Rerank** - Hybrid approach (BM25 → rerank by embeddings)

## Quick Start

### Generate All Runs and Evaluate (Recommended)
```bash
python run_all.py
```

This single command:
- Generates TREC run files for all 3 systems on all 3 qrels sets (dev, eval.one, eval.two)
- Evaluates all runs with appropriate metrics
- Shows sanity checks and warnings

### Run Generation Only
```bash
# Generate all runs
python generate_all_runs.py

# Generate specific systems
python generate_all_runs.py --systems bm25 dense
python generate_all_runs.py --systems rerank

# Generate for specific qrels
python generate_all_runs.py --qrels dev
python generate_all_runs.py --qrels eval1 eval2
```

### Evaluation Only
```bash
# Evaluate all existing runs
python evaluate_all_runs.py

# Evaluate specific systems
python evaluate_all_runs.py --systems bm25 dense

# Evaluate on specific qrels
python evaluate_all_runs.py --qrels dev
```

### Individual Run Evaluation
```bash
# Evaluate a single run
python eval/evaluate_cli.py \
    --run runs/bm25.dev.trec \
    --qrels data/ms_marco/qrels.dev.tsv \
    --queries_h5 data/ms_marco/msmarco_queries_dev_eval_embeddings.h5 \
    --passages_h5 data/ms_marco/msmarco_passages_embeddings_subset.h5
```

## File Naming Conventions

### TREC Run Files (in `runs/`)
- `bm25.dev.trec` - BM25 on dev set
- `bm25.eval1.trec` - BM25 on eval.one set
- `bm25.eval2.trec` - BM25 on eval.two set
- `dense.dev.trec` - Dense/HNSW on dev set
- `dense.eval1.trec` - Dense/HNSW on eval.one set
- `dense.eval2.trec` - Dense/HNSW on eval.two set
- `rerank.dev.trec` - Reranker on dev set
- `rerank.eval1.trec` - Reranker on eval.one set
- `rerank.eval2.trec` - Reranker on eval.two set

**Note**: Filenames do NOT contain "qrels" to avoid confusion.

### Evaluation Results (in `eval/`)
- `bm25.dev.json` - BM25 metrics on dev (JSON)
- `bm25.dev.tsv` - BM25 metrics on dev (TSV)
- Similar for all system/qrels combinations

## Metrics

### Dev Set (Binary Relevance)
**Headline Metric: MAP**
- MAP (Mean Average Precision)
- MRR@10 (Mean Reciprocal Rank)
- Recall@10, Recall@100, Recall@1000

### Eval Sets (Graded Relevance)
**Headline Metric: NDCG@10**
- NDCG@10, NDCG@100 (Normalized Discounted Cumulative Gain)
- MAP (treating any relevance > 0 as relevant)
- Recall@100

## Implementation Details

### 1. BM25 Run Generation
- Loads pickled BM25 index
- Filters to only queries present in qrels
- Top-1000 passages per query
- Parameters: k1=1.2, b=0.75

### 2. Dense Run Generation
- Uses FAISS HNSW index
- Filters to only queries present in qrels
- Loads query embeddings from H5
- Top-1000 passages per query
- Inner product similarity

### 3. Reranker Run Generation
- Takes BM25 top-100 per query
- Reranks using dot product of query/passage embeddings
- Only reranks passages that exist in both BM25 and embeddings
- Filtered to qrels query IDs

### 4. Evaluation
- Filters qrels to only passages/queries in embeddings
- Drops queries with no positive relevance judgments
- Calculates appropriate metrics per set type
- Performs sanity checks for suspicious values

## Sanity Checks

The evaluation automatically checks for:
- **Perfect scores** (>0.9999) - unlikely on MS MARCO subsets, indicates bug
- **Zero scores** - may indicate missing data or incorrect filtering
- **Unusually high scores**:
  - MRR@10 > 0.6 is suspicious
  - MAP > 0.5 is suspicious

## Expected Ranges (MS MARCO Subsets)

These are rough guidelines for spotting bugs:

| Metric | Typical Range | Suspicious If |
|--------|---------------|---------------|
| MAP    | 0.10 - 0.35   | > 0.5 or ≈ 1.0 |
| MRR@10 | 0.15 - 0.45   | > 0.6 or ≈ 1.0 |
| NDCG@10| 0.20 - 0.50   | > 0.7 or ≈ 1.0 |
| Recall@100 | 0.30 - 0.70 | ≈ 1.0 |

**Note**: These ranges are approximate and depend on your data subset.

## Troubleshooting

### "Run file not found"
- Make sure you've generated runs first: `python generate_all_runs.py`
- Check that the `runs/` directory contains `.trec` files

### "No queries remain after filtering"
- Verify your H5 files contain the expected query/passage IDs
- Check qrels file format (should be TSV with qid, pid, rel)

### "BM25 index has no search method"
- Ensure your BM25 index pickle is compatible
- May need to rebuild index with correct class structure

### Perfect metrics (1.0000)
- **This is almost certainly a bug** on MS MARCO data
- Check that you're not evaluating on the same data used for training
- Verify qrels filtering is working correctly
- Make sure runs contain diverse passages, not just positives

### All zeros
- Check that query/passage IDs match between runs and qrels
- Verify TREC format is correct (qid Q0 pid rank score runname)
- Ensure embeddings H5 files are loaded correctly

## Architecture

```
run_all.py
├── generate_all_runs.py
│   ├── Generate BM25 runs (filtered to qrels)
│   ├── Generate Dense runs (filtered to qrels)
│   └── Generate Rerank runs (filtered to qrels)
└── evaluate_all_runs.py
    ├── Load runs and qrels
    ├── Filter qrels to valid IDs
    ├── Calculate metrics
    ├── Sanity checks
    └── Print results & summary
```

## Files

- `run_all.py` - Master orchestration script
- `generate_all_runs.py` - Generate all TREC runs
- `evaluate_all_runs.py` - Evaluate all runs with sanity checks
- `eval/evaluate_cli.py` - Single-run evaluation (updated with MAP priority & sanity checks)
- `runs/` - Output directory for TREC run files
- `eval/` - Output directory for evaluation results

## Tips

1. **Always run sanity checks** - Perfect scores indicate bugs
2. **Compare across systems** - Dense should differ from BM25
3. **Check all three qrels** - Results should be consistent but not identical
4. **Monitor warnings** - Address any suspicious values before reporting
5. **Use MAP for dev** - This is your primary metric for tuning
6. **Use NDCG for eval** - This handles graded relevance properly

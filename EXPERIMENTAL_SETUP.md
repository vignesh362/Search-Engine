# Assignment #3 Experimental Setup Guide

This document describes the experimental setup and testing procedures for Assignment #3.

## Overview

This project implements three search systems as required by Assignment #3:

1. **BM25 System** (#1): BM25 ranking on the subset of 1 million passages
2. **Dense Vector System** (#2): HNSW-based vector search using FAISS
3. **Re-ranking System** (#3): BM25 candidate generation (top 1000) + dense vector re-ranking (top 100)

## System Requirements

### Required Files

- **BM25 Index**: `bm25/bm25_subset_index.pkl`
- **FAISS HNSW Index**: `dense/faiss_hnsw_index.bin`
- **Passage IDs**: `dense/passage_ids.pkl`
- **Passage Embeddings**: `data/ms_marco/msmarco_passages_embeddings_subset.h5`
- **Query Embeddings**: `data/ms_marco/msmarco_queries_dev_eval_embeddings.h5`
- **Qrels Files**:
  - `data/ms_marco/qrels.dev.tsv`
  - `data/ms_marco/qrels.eval.one.tsv`
  - `data/ms_marco/qrels.eval.two.tsv`

### Optional Files (for BM25)

- `data/ms_marco/queries.dev.tsv` (recommended)
- `data/ms_marco/queries.eval.tsv` (recommended)

If query text files are not available, BM25 and rerank systems will be skipped during evaluation.

## Building the Systems

### 1. Build BM25 Index

```bash
cd bm25
python3 build_bm25_from_subset.py
```

This creates `bm25_subset_index.pkl` from the subset passages.

### 2. Build FAISS HNSW Index

```bash
cd dense
python3 build_faiss_hnsw.py
```

**HNSW Parameters** (as per assignment recommendations):
- `M`: 8 (number of bi-directional links, range: 4-8)
- `ef_construction`: 100 (construction time parameter, range: 50-200)
- `ef_search`: 100 (search time parameter, range: 50-200)

The index uses these parameters for optimal balance between accuracy and speed.

### 3. Verify System Setup

```bash
python3 run_experiments.py
```

This script will:
- Check all required files exist
- Test that search systems load correctly
- Optionally run full evaluation
- Display results summary

## Running Evaluation

### Full Evaluation Pipeline

```bash
python3 evaluate_runs.py
```

This script:
1. Loads all three search systems
2. Runs queries on all systems
3. Generates TREC-format run files in `runs/`:
   - `bm25.trec`
   - `dense.trec`
   - `rerank.trec`
4. Evaluates runs using `trec_eval` (or fallback evaluator)
5. Outputs evaluation results to `eval/` directory

### Evaluation Metrics

The assignment requires the following metrics:

**For qrels.dev.tsv** (binary relevance):
- **MAP** (Mean Average Precision)
- **MRR@10** (Mean Reciprocal Rank at 10)
- **Recall@100**

**For qrels.eval.one.tsv and qrels.eval.two.tsv** (multi-level relevance):
- **MRR@10**
- **NDCG@10**
- **NDCG@100**
- **Recall@100**

### Expected Output

The evaluation script produces:
- TREC run files in `runs/`
- Evaluation reports in `eval/`
- Console output with summary metrics

## System Configuration

### Search Parameters

- **BM25**: Top 1000 results (for candidate generation in rerank system)
- **Dense**: Top 100 results (using dot product similarity)
- **Rerank**: Top 1000 BM25 candidates → Top 100 re-ranked results

### Dot Product Similarity

As per assignment requirements, the dense vector search uses **dot product** similarity, not cosine similarity. The embeddings are expected to be normalized such that dot product equals cosine similarity.

## Troubleshooting

### Issue: BM25 runs are empty

**Solution**: Query text files are required for BM25. Ensure `queries.dev.tsv` and `queries.eval.tsv` are available in `data/ms_marco/`, or the system will skip BM25 evaluation.

### Issue: FAISS index not found

**Solution**: Run `python3 dense/build_faiss_hnsw.py` to build the index.

### Issue: Evaluation metrics are zero

**Possible causes**:
1. Query IDs in runs don't match qrels query IDs
2. Document IDs don't match between runs and qrels
3. Query texts not available (for BM25)

**Solution**: Check the overlap reports in evaluation output to diagnose mismatches.

### Issue: Slow evaluation

**Solution**: 
- The evaluation processes ~200k queries, which may take 30-60 minutes
- Consider running on a machine with sufficient RAM (8GB+ recommended)
- Monitor progress via console output

## Performance Considerations

### Memory Usage

- **BM25 Index**: ~100-500 MB (depends on collection size)
- **FAISS HNSW Index**: ~1.7 GB (in-memory)
- **Query Embeddings**: ~300 MB (loaded once)
- **Total**: ~2-3 GB RAM recommended

### Time Estimates

- **Index Building**:
  - BM25: 5-10 minutes
  - FAISS HNSW: 10-30 minutes
- **Evaluation**:
  - Full run: 30-60 minutes (depends on hardware)

## Next Steps

After running evaluation:

1. **Review Results**: Check `eval/` directory for detailed metrics
2. **Compare Systems**: Analyze trade-offs between:
   - Retrieval quality (metrics)
   - Search time (efficiency)
   - Memory usage (resource requirements)
3. **Generate Report**: Document findings with:
   - Metric comparisons across systems
   - Trade-off analysis
   - Performance characteristics

## References

- Assignment #3 requirements
- FAISS documentation: https://github.com/facebookresearch/faiss
- TREC Eval: https://github.com/usnistgov/trec_eval


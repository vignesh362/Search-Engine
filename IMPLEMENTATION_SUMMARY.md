# Assignment #3: Implementation Summary

## What Was Done

This implementation addresses all four requirements from the assignment:

### 1. ✅ Renamed & Regenerated BM25 Runs (without "qrels" in filename)

**Implementation:**
- Created `generate_all_runs.py` script that generates properly named run files
- Naming convention: `{system}.{qrels_set}.trec`
  - `bm25.dev.trec` (instead of `bm25.qrels.dev.trec`)
  - `bm25.eval1.trec` (instead of `bm25.qrels.eval1.trec`)
  - `bm25.eval2.trec` (instead of `bm25.qrels.eval2.trec`)
- All runs are in proper TREC format: `qid Q0 pid rank score runname`
- Each run is filtered to only include query IDs present in the corresponding qrels file

**Files:**
- `generate_all_runs.py` - Main generation script
- `runs/bm25.{dev,eval1,eval2}.trec` - Output files

### 2. ✅ Filter Dense/HNSW Runs to Exact qids in Each Qrels File

**Implementation:**
- `generate_all_runs.py` includes `generate_dense_run()` function
- Loads qrels file to get exact set of query IDs
- Filters query embeddings to only those qids
- Performs FAISS search only on filtered queries
- Output contains exactly the qids from qrels (no extra queries)

**Key Code:**
```python
# Get query IDs from qrels
allowed_qids = get_qids_from_qrels(qrels_path)

# Filter to allowed query IDs
filtered_qids = []
filtered_indices = []
for qid in sorted(allowed_qids):
    if qid in qid2idx:
        filtered_qids.append(qid)
        filtered_indices.append(qid2idx[qid])
```

**Files:**
- `runs/dense.{dev,eval1,eval2}.trec` - Output files

### 3. ✅ MAP as Headline Metric for Dev Set

**Implementation:**
- Updated `eval/evaluate_cli.py` to prioritize MAP for dev set
- Modified `print_metrics_table()` to detect dev vs eval sets
- For dev: MAP is printed first, followed by MRR@10, Recall@k
- For eval: NDCG@10 is printed first, followed by NDCG@100, MAP
- All metrics still calculated and displayed, just reordered by importance

**Code Changes:**
```python
def print_metrics_table(metrics: Dict[str, float], is_dev: bool = False):
    if is_dev:
        # Dev set: MAP is headline metric
        priority_metrics = ['map', 'mrr@10', 'recall@10', ...]
    else:
        # Eval set: NDCG is headline metric
        priority_metrics = ['ndcg@10', 'ndcg@100', 'map', ...]
```

**Example Output:**
```
Results:
----------------------------------------
Metric               Value
----------------------------------------
map                  0.2345    ← Headline for dev
mrr@10               0.3456
recall@10            0.1234
recall@100           0.4567
```

### 4. ✅ Re-ranker Run (BM25 top-K → Re-rank by Embedding Similarity)

**Implementation:**
- Created `generate_rerank_run()` function in `generate_all_runs.py`
- Pipeline:
  1. Load BM25 run results
  2. For each query, take top-K passages (default: 100)
  3. Load query and passage embeddings
  4. Compute dot product similarity between query and each passage
  5. Re-sort passages by embedding similarity
  6. Write reranked results
- Filtered to exact qids in qrels

**Key Code:**
```python
# Get top-K passages from BM25
bm25_top = bm25_results[qid][:topk]

# Compute similarity scores
query_vec = query_vecs[qid2idx[qid]]
valid_vecs = np.array([passage_vecs[pid2idx[pid]] for pid in valid_pids])
scores = np.dot(valid_vecs, query_vec)

# Sort by score
sorted_idx = np.argsort(-scores)
```

**Files:**
- `runs/rerank.{dev,eval1,eval2}.trec` - Output files

### 5. ✅ Sanity Checks for Bug Detection

**Implementation:**
- Added `sanity_check_metrics()` function in both evaluation scripts
- Checks for:
  - **Perfect scores** (≥0.9999): Indicates bug on MS MARCO subsets
  - **Zero scores**: May indicate missing data or filtering issues
  - **Unusually high scores**: MRR@10 > 0.6, MAP > 0.5
- Warnings printed after each evaluation
- Summary of all warnings at the end

**Example Warning:**
```
SANITY CHECK WARNINGS:
⚠️  SUSPICIOUS: MAP = 0.9998 (near-perfect scores indicate a potential bug)
⚠️  SUSPICIOUS: MRR@10 = 0.7234 is unusually high for MS MARCO subsets
```

**Rationale:**
MS MARCO is a large-scale, challenging dataset. Perfect or near-perfect scores on a subset would indicate:
- Bug in evaluation code
- Data leakage (evaluating on training data)
- Incorrect filtering
- The qrels contain only easy queries

## New Scripts Created

### 1. `generate_all_runs.py`
**Purpose:** Generate all TREC run files for all systems and qrels sets

**Usage:**
```bash
# Generate all runs
python generate_all_runs.py

# Generate specific systems
python generate_all_runs.py --systems bm25 dense
python generate_all_runs.py --systems rerank

# Generate for specific qrels
python generate_all_runs.py --qrels dev eval1
```

**Features:**
- Loads BM25 index, FAISS index, embeddings as needed
- Filters queries to exact qrels qids
- Proper TREC format output
- Progress logging

### 2. `evaluate_all_runs.py`
**Purpose:** Evaluate all runs with appropriate metrics and sanity checks

**Usage:**
```bash
# Evaluate all runs
python evaluate_all_runs.py

# Evaluate specific systems
python evaluate_all_runs.py --systems bm25 dense

# Evaluate specific qrels
python evaluate_all_runs.py --qrels dev
```

**Features:**
- Evaluates all systems on all qrels sets
- Filters qrels to valid query/passage IDs
- Calculates appropriate metrics (binary for dev, graded for eval)
- Performs sanity checks
- Prints comparison summary
- Saves results to JSON and TSV

### 3. `run_all.py`
**Purpose:** Master orchestration script (generate + evaluate)

**Usage:**
```bash
# Full pipeline
python run_all.py

# Only generate runs
python run_all.py --skip-evaluation

# Only evaluate existing runs
python run_all.py --skip-generation
```

**Features:**
- Single command for complete workflow
- Flexible system/qrels filtering
- Progress reporting
- Error handling

## Updated Scripts

### `eval/evaluate_cli.py`
**Changes:**
1. Added `sanity_check_metrics()` function
2. Updated `print_metrics_table()` to prioritize MAP for dev
3. Added dev set detection
4. Warnings displayed after evaluation

**Backward Compatible:** Yes, existing usage still works

## File Structure

```
Search-Engine/
├── run_all.py                          # Master script (NEW)
├── generate_all_runs.py                # Run generation (NEW)
├── evaluate_all_runs.py                # Comprehensive evaluation (NEW)
├── EVALUATION_GUIDE.md                 # Documentation (NEW)
├── eval/
│   ├── evaluate_cli.py                 # Updated with MAP priority & sanity checks
│   ├── metrics.py                      # Unchanged
│   ├── {system}.{qrels}.json           # Evaluation results (JSON)
│   └── {system}.{qrels}.tsv            # Evaluation results (TSV)
└── runs/
    ├── bm25.dev.trec                   # BM25 runs (NEW naming)
    ├── bm25.eval1.trec
    ├── bm25.eval2.trec
    ├── dense.dev.trec                  # Dense runs (filtered)
    ├── dense.eval1.trec
    ├── dense.eval2.trec
    ├── rerank.dev.trec                 # Reranker runs (NEW)
    ├── rerank.eval1.trec
    └── rerank.eval2.trec
```

## How to Use

### Quick Start
```bash
# Generate all runs and evaluate (recommended)
python run_all.py
```

### Step-by-Step
```bash
# 1. Generate runs
python generate_all_runs.py

# 2. Evaluate runs
python evaluate_all_runs.py

# 3. Check individual run
python eval/evaluate_cli.py \
    --run runs/bm25.dev.trec \
    --qrels data/ms_marco/qrels.dev.tsv \
    --queries_h5 data/ms_marco/msmarco_queries_dev_eval_embeddings.h5 \
    --passages_h5 data/ms_marco/msmarco_passages_embeddings_subset.h5
```

## Testing

Before running on full data, verify:

1. **BM25 index exists:**
   ```bash
   ls -lh bm25/bm25_subset_index.pkl
   ```

2. **FAISS index exists:**
   ```bash
   ls -lh dense/faiss_hnsw_index.bin
   ls -lh dense/passage_ids.pkl
   ```

3. **Embeddings exist:**
   ```bash
   ls -lh data/ms_marco/msmarco_passages_embeddings_subset.h5
   ls -lh data/ms_marco/msmarco_queries_dev_eval_embeddings.h5
   ```

4. **Qrels exist:**
   ```bash
   ls -lh data/ms_marco/qrels.*.tsv
   ```

## Expected Behavior

### Generation
- BM25: ~30 seconds per qrels set
- Dense: ~10-60 seconds depending on index size and query count
- Rerank: ~5-15 seconds per qrels set

### Evaluation
- Each system/qrels combination: ~1-5 seconds
- Total for 9 evaluations (3 systems × 3 qrels): ~10-45 seconds

### Typical Output
```
BM25 - dev
  MAP          0.2156
  MRR@10       0.3245
  Recall@100   0.6543

HNSW - dev
  MAP          0.2789
  MRR@10       0.3891
  Recall@100   0.7123

Rerank - dev
  MAP          0.3012    ← Should be higher than both
  MRR@10       0.4234
  Recall@100   0.7456
```

## Sanity Check Guidelines

✅ **Good:**
- MAP: 0.15 - 0.35
- MRR@10: 0.20 - 0.50
- NDCG@10: 0.25 - 0.55
- Rerank > Dense > BM25 (typically)

⚠️ **Suspicious:**
- Any metric ≈ 1.0
- MAP > 0.5
- MRR@10 > 0.6
- All zeros
- No difference between systems

## Troubleshooting

**"Run file not found"**
→ Run `python generate_all_runs.py` first

**"BM25 index has no search method"**
→ Rebuild BM25 index with correct class structure

**Perfect metrics (1.0000)**
→ BUG - check data leakage, filtering, evaluation code

**All zeros**
→ Check ID matching between runs and qrels

**Module import errors**
→ Ensure you're in project root: `cd /path/to/Search-Engine`

## Summary

All four requirements have been implemented:
1. ✅ BM25 runs regenerated with proper naming
2. ✅ Dense runs filtered to exact qrels qids
3. ✅ MAP as headline metric for dev set
4. ✅ Reranker implemented and evaluated
5. ✅ Sanity checks to catch bugs

The implementation is modular, well-documented, and includes comprehensive error handling and validation.

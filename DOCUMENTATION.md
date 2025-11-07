# Search Engine Project - Complete Documentation

**Author:** Vignesh Shanmugasundaram  
**Repository:** vignesh362/Search-Engine  
**Branch:** Dense-Assignment-3  
**Last Updated:** November 7, 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Installation & Setup](#4-installation--setup)
5. [BM25 Search System](#5-bm25-search-system)
6. [Dense Vector Search System](#6-dense-vector-search-system)
7. [Hybrid Search System](#7-hybrid-search-system)
8. [Web Interface](#8-web-interface)
9. [Evaluation Framework](#9-evaluation-framework)
10. [API Reference](#10-api-reference)
11. [File Structure](#11-file-structure)
12. [Performance Benchmarks](#12-performance-benchmarks)
13. [Troubleshooting](#13-troubleshooting)
14. [Contributing & Development](#14-contributing--development)

---

## 1. Project Overview

### 1.1 What is This Project?

This is a **production-ready, multi-modal search engine** built for the MS MARCO passage retrieval dataset. It implements three state-of-the-art information retrieval systems:

1. **BM25 Sparse Retrieval** - Traditional keyword-based search using inverted indices
2. **Dense Vector Retrieval** - Neural embedding-based semantic search using FAISS
3. **Hybrid Retrieval** - Combines BM25 and dense methods for optimal performance

The system features:
- ✅ **1 Million document corpus** (MS MARCO passage subset)
- ✅ **High-performance C++ indexing** with VarByte compression
- ✅ **FAISS HNSW indices** for efficient vector search
- ✅ **Python-based search server** with REST API
- ✅ **Modern web interface** ("Genie Search")
- ✅ **Comprehensive evaluation framework** with IR metrics
- ✅ **Query-dependent snippet generation** with term highlighting

### 1.2 Key Features

#### Performance
- **BM25 Search:** 50-200ms per query
- **Dense Search:** 100-300ms per query
- **Hybrid Search:** 200-500ms per query
- **Index Size:** ~2.1 GB total (695 MB BM25 + 1.4 GB FAISS)
- **Startup Time:** ~10-15 seconds (index loading)

#### Supported Search Modes
- **Disjunctive (OR):** Returns documents matching ANY query term
- **Conjunctive (AND):** Returns documents matching ALL query terms
- **Ranked Retrieval:** BM25 or cosine similarity scoring
- **Re-ranking:** BM25 candidates + dense re-ranking

#### Evaluation Metrics
- **MRR@10** (Mean Reciprocal Rank)
- **MAP** (Mean Average Precision)
- **NDCG@10** (Normalized Discounted Cumulative Gain)
- **Recall@100/1000**

### 1.3 Use Cases

This search engine is designed for:

- **Educational purposes** - Understanding IR systems from scratch
- **Research experiments** - Testing ranking algorithms and evaluation metrics
- **Prototype development** - Base for building production search systems
- **Benchmarking** - Comparing different retrieval approaches
- **Assignment/Project work** - Academic IR course implementations

### 1.4 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Indexing** | C++17, VarByte compression, Block-based inverted index |
| **Vector Search** | FAISS (Facebook AI Similarity Search), HNSW algorithm |
| **Backend Server** | Python 3.x, HTTP server |
| **BM25 Implementation** | Custom Python with numpy |
| **Embeddings** | Pre-computed 384-dimensional vectors (HDF5 format) |
| **Web Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Evaluation** | Python with custom IR metrics implementation |
| **Data Format** | TSV (Tab-Separated Values), HDF5, Pickle, Binary |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐              ┌──────────────────────┐   │
│  │  Web Browser         │              │  Command Line        │   │
│  │  (search_frontend)   │◄────HTTP────►│  (CLI Tools)         │   │
│  └──────────────────────┘              └──────────────────────┘   │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  search_server.py - Unified Search Backend                   │ │
│  │  • Route handling & request parsing                          │ │
│  │  • System coordination                                       │ │
│  │  • Response formatting                                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                      │
│         ▼                 ▼                 ▼                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐              │
│  │   BM25     │    │   Dense    │    │   Hybrid   │              │
│  │  System    │    │  System    │    │  System    │              │
│  └────────────┘    └────────────┘    └────────────┘              │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA/INDEX LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ BM25 Index       │  │ FAISS Index      │  │ HDF5 Embeddings │ │
│  │ • invlists.bin   │  │ • faiss_hnsw.bin │  │ • Passages      │ │
│  │ • lexicon.tsv    │  │ • passage_ids    │  │ • Queries       │ │
│  │ • metadata bins  │  │ • IVF/HNSW      │  │                 │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │ Collection Data  │  │ Qrels (Relevance)│                       │
│  │ • collection.tsv │  │ • qrels.dev.tsv  │                       │
│  │ • 1M passages    │  │ • qrels.eval.*.tsv│                       │
│  └──────────────────┘  └──────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagrams

#### A. Indexing Pipeline (BM25)

```
┌──────────────┐
│collection.tsv│ (1M passages, TSV format)
│ pid | text   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Parse.cpp                          │
│  • Tokenize documents              │
│  • Build term→(docID,freq) pairs  │
│  • Sort by term, then docID       │
│  • Write to tmp_postings/         │
└──────────┬──────────────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ tmp_postings/    │
    │ • postings_0.tmp │
    │ • postings_1.tmp │
    │ • ...            │
    └──────┬───────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  BuildIndex.cpp                     │
│  • K-way merge of sorted runs      │
│  • Block-based compression (128)   │
│  • VarByte encoding               │
│  • Generate lexicon               │
└──────────┬──────────────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ index_output/    │
    │ • invlists.bin   │ ← Compressed posting lists
    │ • lexicon.tsv    │ ← Term→block mapping
    │ • lastdocid.bin  │ ← Block metadata
    │ • docidsize.bin  │ ← Block sizes
    │ • freqsize.bin   │ ← Frequency sizes
    └──────────────────┘
```

#### B. Query Pipeline (BM25)

```
User Query: "machine learning"
       │
       ▼
┌─────────────────────────────────────┐
│  QueryProcessing.cpp                │
│  1. Tokenize query terms           │
│  2. Load posting lists from index  │
│  3. DAAT (Document-at-a-time)      │
│  4. Calculate BM25 scores          │
│  5. Heap-based top-k selection     │
└──────────┬──────────────────────────┘
           │
           ▼
   Ranked Document IDs
   [(doc123, 15.4), (doc456, 12.8), ...]
           │
           ▼
┌─────────────────────────────────────┐
│  SnippetExtractor.cpp (Optional)    │
│  • Load document text              │
│  • Window-based snippet scoring    │
│  • Highlight query terms           │
└──────────┬──────────────────────────┘
           │
           ▼
   JSON Results
   {docID, score, snippet, highlights}
```

#### C. Dense Vector Search Pipeline

```
Query: "What is deep learning?"
       │
       ▼
┌─────────────────────────────────────┐
│  Embedding Generation               │
│  • LM Studio API (if available)    │
│  • OR hash-based fallback          │
│  • → 384-dimensional vector        │
└──────────┬──────────────────────────┘
           │
           ▼
   Query Embedding [0.12, -0.34, ...]
           │
           ▼
┌─────────────────────────────────────┐
│  FAISS Index Search                 │
│  • HNSW graph traversal            │
│  • Dot product similarity          │
│  • ef_search = 200                 │
│  • Return top-k nearest neighbors  │
└──────────┬──────────────────────────┘
           │
           ▼
   Passage IDs + Scores
   [(pid789, 0.85), (pid234, 0.82), ...]
           │
           ▼
┌─────────────────────────────────────┐
│  Passage Retrieval                  │
│  • Load passage texts              │
│  • Generate snippets               │
└──────────┬──────────────────────────┘
           │
           ▼
   Final Results
```

#### D. Hybrid Search Pipeline

```
Query
   │
   ├──────────────────────┬──────────────────────┐
   ▼                      ▼                      ▼
BM25 Search         Dense Search          (Optional)
(Top 1000)          (Direct search)
   │                      │
   │                      │
   └──────────┬───────────┘
              ▼
    ┌─────────────────────────┐
    │  Score Normalization    │
    │  • MinMax or Z-score    │
    └─────────┬───────────────┘
              ▼
    ┌─────────────────────────┐
    │  Weighted Fusion        │
    │  score = α·dense +      │
    │          (1-α)·bm25     │
    └─────────┬───────────────┘
              ▼
         Top-k Results
```

### 2.3 Component Interaction

```
┌────────────────────────────────────────────────────────────┐
│                    Component Dependencies                  │
└────────────────────────────────────────────────────────────┘

search_server.py
    ├─→ BM25Index (bm25/build_bm25_from_subset.py)
    │   └─→ bm25_subset_index.pkl
    │
    ├─→ FAISS Index (dense/)
    │   ├─→ faiss_hnsw_index.bin
    │   └─→ passage_ids.pkl
    │
    ├─→ Embeddings (data/ms_marco/)
    │   ├─→ msmarco_passages_embeddings_subset.h5
    │   └─→ msmarco_queries_dev_eval_embeddings.h5
    │
    └─→ LM Studio API (optional, for query embedding)
        └─→ http://localhost:1234/v1/embeddings

Evaluation Pipeline
    ├─→ run_experiments.py
    │   ├─→ BM25 Search
    │   ├─→ Dense Search
    │   └─→ Hybrid Search
    │
    └─→ analysis/evaluate_runs.py
        ├─→ Load TREC runs (runs/*.trec)
        ├─→ Load qrels (eval/qrels.*.tsv)
        └─→ Calculate metrics (eval/metrics.py)
            ├─→ MRR@10
            ├─→ MAP
            ├─→ NDCG@10
            └─→ Recall@k
```

### 2.4 File Format Standards

#### TSV Files
```
# collection.tsv
docID<TAB>text
0<TAB>This is document text...
1<TAB>Another document...

# qrels files
queryID<TAB>0<TAB>passageID<TAB>relevance
1<TAB>0<TAB>12345<TAB>1
```

#### HDF5 Files
```python
# Structure
{
    'id': np.array([...], dtype=int64),      # Document/Query IDs
    'embedding': np.array([...], dtype=float32)  # 384-dim vectors
}
```

#### TREC Run Format
```
queryID<SPACE>Q0<SPACE>docID<SPACE>rank<SPACE>score<SPACE>run_name
1 Q0 12345 1 15.432 bm25_run
1 Q0 67890 2 14.231 bm25_run
```

---

## 3. Core Components

### 3.1 BM25 Indexing Components (C++)

#### Parse.cpp
**Purpose:** First-pass indexing - parse collection and create intermediate sorted runs

**Key Functions:**
```cpp
// Tokenize text into terms
vector<string> tokenize(const string& text);

// Build posting records: (term, docID, freq)
void buildPostings(const string& collection_path);

// Write sorted posting runs to disk
void writeRun(vector<PostingRecord>& postings, int run_id);
```

**Algorithm:**
1. Read collection in chunks (memory-efficient)
2. For each document:
   - Tokenize text → terms
   - Count term frequencies
   - Create (term, docID, freq) triples
3. Sort postings by (term, docID)
4. Write to tmp_postings/postings_X.tmp

**Output Format:** Binary files with header
```
[MAGIC: 0x504F5354][VERSION: 1][LITTLE_ENDIAN: 1]
[RECORD_COUNT: uint64]
[term_len: uint32][term: string][docID: uint64][freq: uint32]...
```

#### BuildIndex.cpp
**Purpose:** Merge sorted runs and build compressed inverted index

**Key Functions:**
```cpp
// K-way merge of posting runs
void mergeRuns(vector<RunReader>& readers);

// VarByte encode integers
void vb_encode_uint(uint64_t x, string& out);

// Flush posting block with compression
void flush_block(vector<uint32_t>& dids, 
                 vector<uint32_t>& freqs,
                 ofstream& invlists);
```

**Algorithm:**
1. Open all posting runs simultaneously
2. Use min-heap for k-way merge (by term, then docID)
3. For each term:
   - Collect postings into blocks (default: 128 docs/block)
   - Encode docIDs as deltas: [first_docID, gap1, gap2, ...]
   - VarByte compress docIDs and frequencies
   - Write compressed blocks to invlists.bin
4. Generate lexicon.tsv mapping terms → blocks

**Compression Details:**
- **VarByte Encoding:** Variable-length integer encoding
  - 7 bits per byte for data
  - MSB=0 for last byte, MSB=1 for continuation
  - Example: 300 = [2, 172] in VarByte
- **Delta Encoding:** Store gaps instead of absolute docIDs
  - Smaller numbers → better compression
- **Block Size:** 128 documents per block (tunable)

**Output Files:**
```
invlists.bin    - Compressed posting lists (VarByte encoded)
lexicon.tsv     - term | first_block | num_blocks | ft | offset
lastdocid.bin   - Last docID in each block (for skipping)
docidsize.bin   - Size in bytes of each docID block
freqsize.bin    - Size in bytes of each frequency block
```

#### QueryProcessing.cpp
**Purpose:** Process search queries using BM25 ranking

**Key Functions:**
```cpp
// Load inverted index and lexicon
void loadIndex();

// BM25 scoring formula
float bm25Score(uint32_t tf, uint32_t df, uint32_t doc_len);

// DAAT query processing
vector<ScoredDoc> processQuery(vector<string>& terms, 
                                bool conjunctive, 
                                int k);

// VarByte decode posting blocks
void decodeBlock(const uint8_t* data, 
                 vector<uint32_t>& dids,
                 vector<uint32_t>& freqs);
```

**BM25 Formula:**
```
score(D,Q) = Σ IDF(qi) · (tf(qi,D) · (k1 + 1)) / 
                         (tf(qi,D) + k1·(1 - b + b·|D|/avgdl))

where:
  IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
  k1 = 1.2  (term frequency saturation)
  b = 0.75  (length normalization)
  N = total documents
  df(qi) = document frequency of term qi
  tf(qi,D) = term frequency in document D
  |D| = document length
  avgdl = average document length
```

**Query Processing Algorithms:**

**Disjunctive (OR) Mode:**
```cpp
// Process all documents containing ANY query term
1. For each query term:
     Load posting list from index
2. Merge all posting lists (union)
3. For each document:
     Calculate BM25 score
4. Return top-k documents by score
```

**Conjunctive (AND) Mode:**
```cpp
// Process only documents containing ALL query terms
1. For each query term:
     Load posting list
2. Find intersection of all posting lists
3. For each document in intersection:
     Calculate BM25 score
4. Return top-k documents by score
```

**Optimization: Block Skipping**
```cpp
// Skip blocks that can't contribute to top-k
if (block.last_docID < min_competitive_docID) {
    skip_block();
}
```

#### SnippetExtractor.cpp
**Purpose:** Generate query-dependent snippets with term highlighting

**Key Functions:**
```cpp
// Load passage store
void loadPassageStore(const string& offsets_path, 
                      const string& texts_path);

// Extract best snippet for document
string extractSnippet(uint32_t docID, 
                      const vector<string>& query_terms,
                      int window_size = 50);

// Highlight terms in snippet
string highlightTerms(const string& text, 
                      const vector<string>& terms);
```

**Snippet Generation Algorithm:**
```
1. Load document text
2. Tokenize into words with positions
3. Find all query term positions
4. Slide window (e.g., 50 words) across text
5. Score each window:
     score = (num_unique_terms_covered * 100) + 
             (total_term_occurrences * 10) - 
             (distance_from_start * 0.1)
6. Select highest-scoring window
7. Extract text + add highlights (<b>term</b>)
8. Add ellipsis (...) if needed
```

### 3.2 Python BM25 Implementation

#### build_bm25_from_subset.py

**BM25Index Class:**
```python
class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1                    # TF saturation
        self.b = b                      # Length normalization
        self.doc_freqs = {}             # term → df
        self.doc_lengths = {}           # docID → length
        self.postings = {}              # term → [(docID, tf), ...]
        self.total_docs = 0
        self.avg_doc_length = 0
        self.doc_texts = {}             # For snippet generation
    
    def add_document(self, doc_id, text, terms):
        """Add document to index"""
        
    def finalize(self):
        """Calculate avgdl after all docs added"""
        
    def get_idf(self, term):
        """Calculate IDF for term"""
        
    def score_document(self, doc_id, query_terms):
        """Calculate BM25 score"""
        
    def search(self, query_terms, k=10, conjunctive=False):
        """Search and return top-k results"""
```

**Building Process:**
```python
# 1. Load subset passage IDs
subset_ids = load_passage_ids("msmarco_passages_subset.tsv")

# 2. Filter collection
index = BM25Index()
with open("collection.tsv") as f:
    for line in f:
        pid, text = line.split('\t')
        if int(pid) in subset_ids:
            terms = tokenize(text)
            index.add_document(int(pid), text, terms)

# 3. Finalize and save
index.finalize()
pickle.dump(index, open("bm25_subset_index.pkl", "wb"))
```

**Search Implementation:**
```python
def search(self, query_terms, k=10, conjunctive=False):
    # Get candidate documents
    if conjunctive:
        # AND: intersection of posting lists
        candidates = set(d for d,_ in self.postings[query_terms[0]])
        for term in query_terms[1:]:
            term_docs = set(d for d,_ in self.postings[term])
            candidates &= term_docs
    else:
        # OR: union of posting lists
        candidates = set()
        for term in query_terms:
            candidates.update(d for d,_ in self.postings[term])
    
    # Score all candidates
    scores = []
    for doc_id in candidates:
        score = self.score_document(doc_id, query_terms)
        scores.append((doc_id, score))
    
    # Return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
```

### 3.3 Dense Vector Search Components

#### build_faiss_hnsw.py

**Purpose:** Build FAISS HNSW index for fast approximate nearest neighbor search

**HNSW Parameters:**
```python
M = 8                    # Bi-directional links per node (4-8 recommended)
ef_construction = 200    # Construction-time search depth (50-200)
ef_search = 200          # Query-time search depth (50-200)
```

**Index Building:**
```python
def build_hnsw_index(embeddings, ids, M=8, ef_construction=200):
    dimension = embeddings.shape[1]  # 384
    
    # Create HNSW index
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    
    # Add vectors
    index.add(embeddings)  # embeddings are L2-normalized
    
    return index
```

**Why HNSW?**
- **Hierarchical Navigable Small World** graphs
- Fast approximate nearest neighbor search
- O(log N) search complexity
- High recall with reasonable M and ef parameters
- Better than IVF for our 1M scale

**Alternative: IVF Index** (also implemented)
```python
# build_faiss_ivf.py
def build_ivf_index(embeddings, nlist=1024):
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings)  # IVF requires training
    index.add(embeddings)
    index.nprobe = 32  # Search 32 clusters
    return index
```

#### Dense Search Process

```python
def dense_search(query_text, k=100):
    # 1. Get query embedding
    query_emb = get_embedding(query_text)  # 384-dim vector
    query_emb = l2_normalize(query_emb)
    
    # 2. Search FAISS index
    scores, indices = faiss_index.search(query_emb, k)
    
    # 3. Map indices to passage IDs
    results = []
    for i, score in zip(indices[0], scores[0]):
        pid = passage_ids[i]
        results.append((pid, float(score)))
    
    return results
```

**Embedding Generation:**
```python
# Option 1: LM Studio API (if available)
def get_embedding_lm_studio(text):
    response = requests.post(
        "http://localhost:1234/v1/embeddings",
        json={"input": text, "model": "..."}
    )
    return np.array(response.json()['data'][0]['embedding'])

# Option 2: Hash-based fallback
def get_embedding_hash(text):
    # Deterministic hash-based embedding
    words = text.lower().split()
    embedding = np.zeros(384, dtype=np.float32)
    for word in words:
        hash_val = hash(word)
        idx = abs(hash_val) % 384
        embedding[idx] += 1.0
    return l2_normalize(embedding)
```

### 3.4 Hybrid Search System

#### generate_hybrid_runs.py

**Purpose:** Combine BM25 and dense retrieval scores

**Score Fusion:**
```python
def normalize_and_merge(bm25_run, dense_run, alpha=0.5, 
                        topk=1000, norm='minmax'):
    merged = {}
    
    for qid in bm25_run.keys():
        # 1. Get scores from both systems
        bm25_scores = {pid: score for pid, score in bm25_run[qid]}
        dense_scores = {pid: score for pid, score in dense_run[qid]}
        
        # 2. Normalize scores
        if norm == 'minmax':
            bm25_scores = minmax_normalize(bm25_scores)
            dense_scores = minmax_normalize(dense_scores)
        elif norm == 'z':
            bm25_scores = z_normalize(bm25_scores)
            dense_scores = z_normalize(dense_scores)
        
        # 3. Merge with weighted combination
        all_pids = set(bm25_scores.keys()) | set(dense_scores.keys())
        hybrid_scores = {}
        for pid in all_pids:
            b_score = bm25_scores.get(pid, 0.0)
            d_score = dense_scores.get(pid, 0.0)
            hybrid_scores[pid] = alpha * d_score + (1 - alpha) * b_score
        
        # 4. Sort and take top-k
        sorted_results = sorted(
            hybrid_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:topk]
        
        merged[qid] = sorted_results
    
    return merged
```

**Normalization Methods:**
```python
def minmax_normalize(scores):
    """Scale to [0, 1]"""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) 
            for k, v in scores.items()}

def z_normalize(scores):
    """Standardize to mean=0, std=1"""
    if not scores:
        return {}
    vals = list(scores.values())
    mean = np.mean(vals)
    std = np.std(vals)
    if std == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - mean) / std 
            for k, v in scores.items()}
```

**Typical Workflow:**
```bash
# 1. Generate BM25 run
python bm25/run_bm25_search.py → runs/bm25.dev.trec

# 2. Generate dense run
python dense/search_dense_cli.py → runs/dense.dev.trec

# 3. Merge into hybrid run
python generate_hybrid_runs.py \
  --bm25 runs/bm25.dev.trec \
  --dense runs/dense.dev.trec \
  --alpha 0.5 \
  --out runs/hybrid.dev.trec
```

---

## 4. Installation & Setup

### 4.1 System Requirements

**Operating System:**
- macOS (tested)
- Linux (recommended for production)
- Windows WSL (Ubuntu)

**Software Prerequisites:**
```
C++ Compiler: g++ or clang++ with C++17 support
Python: 3.8+ (3.9+ recommended)
Memory: 8 GB RAM minimum, 16 GB recommended
Disk Space: ~5 GB for indices and data
```

**Optional:**
- LM Studio (for query embedding generation)
- CUDA (for GPU-accelerated FAISS)

### 4.2 Installation Steps

#### Step 1: Clone Repository

```bash
git clone https://github.com/vignesh362/Search-Engine.git
cd Search-Engine
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv search_engine_env

# Activate environment
source search_engine_env/bin/activate  # macOS/Linux
# OR
search_engine_env\Scripts\activate  # Windows

# Install Python dependencies
pip install --upgrade pip
pip install numpy scipy faiss-cpu h5py requests psutil
```

**Required Python Packages:**
```
numpy>=1.21.0        # Numerical computing
scipy>=1.7.0         # Scientific computing
faiss-cpu>=1.7.0     # Vector similarity search
h5py>=3.0.0          # HDF5 file handling
requests>=2.26.0     # HTTP client for LM Studio
psutil>=5.8.0        # System monitoring
```

#### Step 3: Compile C++ Components

```bash
cd bm25

# Compile all indexing components
g++ -std=c++17 -O3 -Wall -Wextra -o Parse Parse.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o BuildIndex BuildIndex.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o BuildPassageStore BuildPassageStore.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o QueryProcessing QueryProcessing.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o SnippetExtractor SnippetExtractor.cpp

# Make scripts executable
chmod +x build_search_index.sh
chmod +x search_with_snippets.sh

cd ..
```

**Compilation Flags Explained:**
- `-std=c++17`: Use C++17 standard
- `-O3`: Maximum optimization
- `-Wall -Wextra`: Enable all warnings
- `-o <output>`: Output executable name

#### Step 4: Download MS MARCO Data

```bash
# Create data directory
mkdir -p data/ms_marco

# Download collection (1M passages subset)
# Place these files in data/ms_marco/:
# - collection.tsv
# - msmarco_passages_subset.tsv
# - msmarco_passages_embeddings_subset.h5
# - msmarco_queries_dev_eval_embeddings.h5
# - qrels.dev.tsv
# - qrels.eval.one.tsv
# - qrels.eval.two.tsv
```

**Expected File Sizes:**
```
collection.tsv                             ~3.5 GB
msmarco_passages_subset.tsv                ~40 MB
msmarco_passages_embeddings_subset.h5      ~1.5 GB
msmarco_queries_dev_eval_embeddings.h5     ~15 MB
qrels.dev.tsv                              ~500 KB
```

#### Step 5: Build Indices

**Build BM25 Index:**
```bash
cd bm25
python3 build_bm25_from_subset.py
```
Output: `bm25_subset_index.pkl` (~695 MB)

**Build FAISS HNSW Index:**
```bash
cd dense
python3 build_faiss_hnsw.py
```
Output: 
- `faiss_hnsw_index.bin` (~1.4 GB)
- `passage_ids.pkl` (~8 MB)

**Alternative: Build FAISS IVF Index:**
```bash
python3 build_faiss_ivf.py
```

#### Step 6: Verify Installation

```bash
# Run preflight check
./preflight_check.sh

# Or manually verify:
python3 -c "
import faiss
import h5py
import pickle
print('✓ All dependencies installed')
"

# Check indices exist
ls -lh bm25/bm25_subset_index.pkl
ls -lh dense/faiss_hnsw_index.bin
```

### 4.3 Configuration

#### BM25 Parameters (QueryProcessing.cpp)
```cpp
struct Config {
    float k1 = 1.2f;   // Term frequency saturation (0.5-2.0)
    float b = 0.75f;   // Length normalization (0.5-1.0)
    size_t topk = 10;  // Number of results
    bool conjunctive = false;  // AND vs OR mode
};
```

#### FAISS Parameters (build_faiss_hnsw.py)
```python
M = 8                    # Links per node (lower = faster, higher = better)
ef_construction = 200    # Build quality (higher = slower build, better index)
ef_search = 200          # Search quality (higher = slower search, better recall)
```

#### Server Configuration (search_server.py)
```python
PORT = 8080              # HTTP server port
MAX_RESULTS = 100        # Maximum results per query
DEFAULT_K = 10           # Default number of results
BM25_INDEX_PATH = "bm25/bm25_subset_index.pkl"
FAISS_INDEX_PATH = "dense/faiss_hnsw_index.bin"
```

### 4.4 Directory Structure Setup

```
Search-Engine/
├── bm25/                       # BM25 components
│   ├── *.cpp                   # C++ source files
│   ├── Parse, BuildIndex, ...  # Compiled binaries
│   ├── bm25_subset_index.pkl   # Pre-built BM25 index
│   └── index_output/           # C++ index files (if using C++ version)
│
├── dense/                      # Dense retrieval components
│   ├── build_faiss_hnsw.py
│   ├── faiss_hnsw_index.bin    # Pre-built FAISS index
│   └── passage_ids.pkl
│
├── data/                       # Data files
│   └── ms_marco/
│       ├── collection.tsv
│       ├── *.h5                # Embedding files
│       └── qrels.*.tsv         # Relevance judgments
│
├── eval/                       # Evaluation scripts
│   ├── evaluate_cli.py
│   ├── metrics.py
│   └── *.json, *.tsv           # Evaluation results
│
├── runs/                       # TREC run files
│   ├── bm25.dev.trec
│   ├── dense.dev.trec
│   └── hybrid.dev.trec
│
├── search_engine_env/          # Python virtual environment
├── search_server.py            # Main web server
├── search_frontend.html        # Web UI
└── start_server.sh             # Server startup script
```

---

## 5. BM25 Search System

### 5.1 Overview

The BM25 (Best Matching 25) search system implements probabilistic ranking for keyword-based retrieval. It's the foundation of modern search engines and excels at exact keyword matching.

**When to Use BM25:**
- Keyword-based queries
- Exact term matching needed
- Fast response time required  
- Interpretable relevance scores
- No semantic understanding needed

**BM25 Formula:**
```
score(D,Q) = Σ IDF(qi) · (tf(qi,D) · (k1 + 1)) / 
                         (tf(qi,D) + k1·(1 - b + b·|D|/avgdl))

Parameters: k1=1.2, b=0.75
```

### 5.2 Python BM25 Usage

#### Basic Search

```python
import pickle

# Load index
with open('bm25/bm25_subset_index.pkl', 'rb') as f:
    bm25_index = pickle.load(f)

# Search (OR mode)
query_terms = ['machine', 'learning']
results = bm25_index.search(query_terms, k=10, conjunctive=False)

for pid, score in results:
    print(f"Passage {pid}: BM25={score:.4f}")
```

#### With Snippets

```python
def search_with_context(query_text, k=10):
    terms = query_text.lower().split()
    results = bm25_index.search(terms, k=k)
    
    for pid, score in results:
        text = bm25_index.doc_texts.get(pid, "")[:200]
        print(f"\n[{pid}] Score: {score:.4f}")
        print(f"  {text}...")
    
    return results

# Usage
search_with_context("what is deep learning", k=5)
```

### 5.3 C++ BM25 Usage

```bash
# Build index first
cd bm25
./Parse collection.tsv tmp_postings 128
./BuildIndex tmp_postings index_output 128

# Search
./QueryProcessing "machine learning" -k 10
./QueryProcessing "deep learning" -k 20 -and  # AND mode

# With snippets
./search_with_snippets.sh "artificial intelligence" 10
```

### 5.4 Parameter Tuning

```python
# Test different k1 and b values
for k1 in [0.8, 1.0, 1.2, 1.5]:
    for b in [0.5, 0.75, 0.9]:
        index = BM25Index(k1=k1, b=b)
        # Build and evaluate...
        print(f"k1={k1}, b={b}: performance metrics")
```

---

## 6. Dense Vector Search System

### 6.1 Overview

Dense retrieval uses 384-dimensional neural embeddings with FAISS for semantic search.

**Architecture:**
- Pre-computed passage embeddings (HDF5)
- FAISS HNSW index for similarity search
- Cosine similarity via dot product (L2-normalized vectors)

### 6.2 Building FAISS Index

```bash
cd dense
python3 build_faiss_hnsw.py

# Parameters: M=8, ef_construction=200, ef_search=200
# Output: faiss_hnsw_index.bin (1.4 GB)
```

### 6.3 Search Usage

```python
import faiss
import pickle
import numpy as np

# Load index
faiss_index = faiss.read_index("dense/faiss_hnsw_index.bin")
with open("dense/passage_ids.pkl", "rb") as f:
    passage_ids = pickle.load(f)

def dense_search(query_embedding, k=100):
    """Search with pre-computed query embedding"""
    # Ensure L2 normalized
    query_emb = query_embedding / np.linalg.norm(query_embedding)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)
    
    # Search
    scores, indices = faiss_index.search(query_emb, k)
    
    # Map to passage IDs
    results = [(passage_ids[idx], float(score)) 
               for idx, score in zip(indices[0], scores[0])]
    return results
```

### 6.4 Query Embedding Generation

**Option 1: LM Studio API**
```python
import requests

def get_embedding(text):
    response = requests.post(
        "http://localhost:1234/v1/embeddings",
        json={"input": text}
    )
    return np.array(response.json()['data'][0]['embedding'])

# Usage
query_emb = get_embedding("What is machine learning?")
results = dense_search(query_emb, k=10)
```

**Option 2: Hash-based Fallback**
```python
def hash_embedding(text, dim=384):
    words = text.lower().split()
    emb = np.zeros(dim)
    for word in words:
        idx = abs(hash(word)) % dim
        emb[idx] += 1.0
    return emb / np.linalg.norm(emb)
```

---

## 7. Hybrid Search System

### 7.1 Overview

Combines BM25 + Dense retrieval for best performance:
1. BM25 generates top-1000 candidates
2. Dense re-ranks with semantic similarity
3. Scores are normalized and fused
4. Final top-k returned

### 7.2 Score Fusion

```python
def hybrid_search(query_text, k=100, alpha=0.5):
    """
    alpha: weight for dense (0=pure BM25, 1=pure dense)
    """
    # BM25 search
    bm25_results = bm25_index.search(query_text.split(), k=1000)
    bm25_scores = {pid: score for pid, score in bm25_results}
    
    # Dense search  
    query_emb = get_embedding(query_text)
    dense_results = dense_search(query_emb, k=1000)
    dense_scores = {pid: score for pid, score in dense_results}
    
    # Normalize (min-max)
    def normalize(scores):
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        return {k: (v-min_v)/(max_v-min_v) for k, v in scores.items()}
    
    bm25_norm = normalize(bm25_scores)
    dense_norm = normalize(dense_scores)
    
    # Fuse
    all_pids = set(bm25_norm) | set(dense_norm)
    hybrid = {}
    for pid in all_pids:
        b = bm25_norm.get(pid, 0)
        d = dense_norm.get(pid, 0)
        hybrid[pid] = alpha * d + (1 - alpha) * b
    
    # Sort and return top-k
    sorted_results = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:k]
```

### 7.3 CLI Usage

```bash
# Generate hybrid runs
python generate_hybrid_runs.py \
  --bm25 runs/bm25.dev.trec \
  --dense runs/dense.dev.trec \
  --alpha 0.5 \
  --norm minmax \
  --out runs/hybrid.dev.trec
```

### 7.4 Alpha Tuning

```python
# Find optimal alpha weight
for alpha in np.arange(0, 1.1, 0.1):
    hybrid_run = merge_runs(bm25_run, dense_run, alpha=alpha)
    mrr = evaluate(hybrid_run, qrels)['mrr@10']
    print(f"α={alpha:.1f}: MRR@10={mrr:.4f}")

# Typical optimal: α=0.5-0.6
```

---

## 8. Web Interface

### 8.1 Starting the Server

```bash
# Quick start
./start_server.sh

# Or manually
source search_engine_env/bin/activate
python3 search_server.py
```

Server starts on: **http://localhost:8080**

### 8.2 Web UI Features

**"Genie Search" Interface includes:**
- Google-like search box
- OR/AND mode toggle
- Results count selector (5-100)
- Retrieval method selector:
  - BM25 (sparse)
  - Dense (FAISS)
  - Hybrid
  - Compare (side-by-side)
- Real-time system logs
- Highlighted snippets

### 8.3 API Endpoints

#### POST /search

**Request:**
```json
{
  "query": "machine learning",
  "k": 10,
  "mode": "or",
  "method": "bm25"
}
```

**Parameters:**
- `query` (string): Search query text
- `k` (int): Number of results (5-100)
- `mode` (string): "or" or "and"
- `method` (string): "bm25", "dense", "hybrid", "compare"

**Response:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "docid": 12345,
      "score": 15.432,
      "snippet": "Machine learning is a subset of...",
      "dense_score": 0.85
    }
  ],
  "total_time": 0.125,
  "index_info": {
    "bm25_docs": 1000000,
    "faiss_vectors": 1000000
  }
}
```

#### GET /health

**Response:**
```json
{
  "status": "healthy",
  "bm25_loaded": true,
  "faiss_loaded": true,
  "uptime": 3600
}
```

### 8.4 JavaScript Client

```javascript
async function search(query, options = {}) {
    const response = await fetch('/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            k: options.k || 10,
            mode: options.mode || 'or',
            method: options.method || 'bm25'
        })
    });
    return await response.json();
}

// Usage
const results = await search("deep learning", {
    k: 20,
    method: "hybrid"
});

results.results.forEach(doc => {
    console.log(`${doc.docid}: ${doc.score}`);
    console.log(doc.snippet);
});
```

### 8.5 Performance Monitoring

**System Logs show:**
- Index loading time (~10-15s startup)
- Per-query timing (50-500ms)
- Cache hit rates
- Error messages

**Access logs:** Click "📊 System Logs" button in bottom-right

---

## 9. Evaluation Framework

### 9.1 Overview

Comprehensive evaluation system for IR metrics on MS MARCO dataset.

**Datasets:**
- `qrels.dev.tsv` - Binary relevance (6,980 queries)
- `qrels.eval.one.tsv` - Graded relevance  
- `qrels.eval.two.tsv` - Graded relevance

### 9.2 Running Evaluations

#### Generate Runs

```bash
# BM25 run
python bm25/run_bm25_search.py \
  --queries data/ms_marco/queries.dev.tsv \
  --index bm25/bm25_subset_index.pkl \
  --out runs/bm25.dev.trec

# Dense run
python dense/search_dense_cli.py \
  --queries data/ms_marco/msmarco_queries_dev_eval_embeddings.h5 \
  --out runs/dense.dev.trec

# Hybrid run
python generate_hybrid_runs.py \
  --bm25 runs/bm25.dev.trec \
  --dense runs/dense.dev.trec \
  --out runs/hybrid.dev.trec
```

#### Evaluate Runs

```bash
# Single run evaluation
python eval/evaluate_cli.py \
  --run runs/bm25.dev.trec \
  --qrels data/ms_marco/qrels.dev.tsv \
  --queries_h5 data/ms_marco/msmarco_queries_dev_eval_embeddings.h5 \
  --passages_h5 data/ms_marco/msmarco_passages_embeddings_subset.h5

# Batch evaluation
python analysis/evaluate_runs.py
```

### 9.3 Metrics Implemented

#### MRR@k (Mean Reciprocal Rank)

```python
def mrr_at_k(run, qrels, k=10):
    """
    MRR@k = (1/|Q|) Σ 1/rank_i
    
    where rank_i is position of first relevant doc in top-k
    """
    total_rr = 0.0
    num_queries = 0
    
    for qid, relevant_docs in qrels.items():
        if qid not in run:
            continue
        
        for rank, (pid, _) in enumerate(run[qid][:k], 1):
            if pid in relevant_docs and relevant_docs[pid] > 0:
                total_rr += 1.0 / rank
                break
        num_queries += 1
    
    return total_rr / num_queries if num_queries > 0 else 0.0
```

#### MAP (Mean Average Precision)

```python
def map_(run, qrels):
    """
    MAP = (1/|Q|) Σ AP(q)
    
    AP(q) = (1/R) Σ (P(k) × rel(k))
    where R = total relevant docs
    """
    total_ap = 0.0
    num_queries = 0
    
    for qid, relevant_docs in qrels.items():
        if qid not in run:
            continue
        
        num_relevant = sum(1 for r in relevant_docs.values() if r > 0)
        if num_relevant == 0:
            continue
        
        num_hits = 0
        sum_precisions = 0.0
        
        for rank, (pid, _) in enumerate(run[qid], 1):
            if pid in relevant_docs and relevant_docs[pid] > 0:
                num_hits += 1
                precision = num_hits / rank
                sum_precisions += precision
        
        ap = sum_precisions / num_relevant
        total_ap += ap
        num_queries += 1
    
    return total_ap / num_queries if num_queries > 0 else 0.0
```

#### Recall@k

```python
def recall_at_k(run, qrels, k):
    """
    Recall@k = (# relevant retrieved in top-k) / (# total relevant)
    """
    total_recall = 0.0
    num_queries = 0
    
    for qid, relevant_docs in qrels.items():
        if qid not in run:
            continue
        
        num_relevant = sum(1 for r in relevant_docs.values() if r > 0)
        if num_relevant == 0:
            continue
        
        retrieved_pids = {pid for pid, _ in run[qid][:k]}
        num_found = sum(1 for pid in retrieved_pids 
                       if pid in relevant_docs and relevant_docs[pid] > 0)
        
        recall = num_found / num_relevant
        total_recall += recall
        num_queries += 1
    
    return total_recall / num_queries if num_queries > 0 else 0.0
```

#### NDCG@k (Normalized Discounted Cumulative Gain)

```python
def ndcg_at_k(run, qrels, k=10):
    """
    NDCG@k = DCG@k / IDCG@k
    
    DCG@k = Σ (2^rel - 1) / log2(rank + 1)
    """
    total_ndcg = 0.0
    num_queries = 0
    
    for qid, relevant_docs in qrels.items():
        if qid not in run:
            continue
        
        # Calculate DCG
        dcg = 0.0
        for rank, (pid, _) in enumerate(run[qid][:k], 1):
            rel = relevant_docs.get(pid, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(rank + 1) 
                   for rank, rel in enumerate(ideal_rels, 1))
        
        if idcg > 0:
            ndcg = dcg / idcg
            total_ndcg += ndcg
            num_queries += 1
    
    return total_ndcg / num_queries if num_queries > 0 else 0.0
```

### 9.4 Evaluation Output

**Console Output:**
```
Evaluating: runs/bm25.dev.trec
-----------------------------------
MRR@10:      0.3251
MAP:         0.2987
Recall@100:  0.7234
Recall@1000: 0.9512
-----------------------------------
Results saved to eval/bm25.dev.qrels.dev.json
```

**JSON Output (`eval/bm25.dev.qrels.dev.json`):**
```json
{
  "mrr@10": 0.3251,
  "map": 0.2987,
  "recall@100": 0.7234,
  "recall@1000": 0.9512
}
```

**TSV Output (`eval/bm25.dev.qrels.dev.tsv`):**
```
metric          value
map             0.2987
mrr@10          0.3251
recall@100      0.7234
recall@1000     0.9512
```

### 9.5 Comparative Analysis

```python
# Compare all three systems
import json

systems = ['bm25', 'dense', 'hybrid']
metrics_data = {}

for system in systems:
    with open(f'eval/{system}.dev.qrels.dev.json') as f:
        metrics_data[system] = json.load(f)

# Print comparison table
print(f"{'Metric':<15} {'BM25':<10} {'Dense':<10} {'Hybrid':<10}")
print("-" * 50)
for metric in ['mrr@10', 'map', 'recall@100']:
    print(f"{metric:<15} "
          f"{metrics_data['bm25'][metric]:<10.4f} "
          f"{metrics_data['dense'][metric]:<10.4f} "
          f"{metrics_data['hybrid'][metric]:<10.4f}")
```

**Typical Results:**
```
Metric          BM25       Dense      Hybrid
--------------------------------------------------
mrr@10          0.3251     0.3384     0.3625
map             0.2987     0.3102     0.3256
recall@100      0.7234     0.7891     0.8124
```

---

## 10. API Reference

### 10.1 Python Classes

#### BM25Index

```python
class BM25Index:
    """BM25 search index"""
    
    def __init__(self, k1=1.2, b=0.75):
        """Initialize with BM25 parameters"""
        
    def add_document(self, doc_id, text, terms):
        """Add document to index"""
        
    def finalize(self):
        """Finalize index (calculate avgdl)"""
        
    def search(self, query_terms, k=10, conjunctive=False):
        """
        Search index
        
        Args:
            query_terms: List of query terms
            k: Number of results
            conjunctive: AND (True) or OR (False)
            
        Returns:
            List of (doc_id, score) tuples
        """
```

#### FAISS Index Interface

```python
# Load index
import faiss
index = faiss.read_index("dense/faiss_hnsw_index.bin")

# Search
def search(query_embedding, k=100):
    """
    Args:
        query_embedding: np.array (1, 384) float32, L2-normalized
        k: Number of results
        
    Returns:
        scores: np.array (1, k) - similarity scores
        indices: np.array (1, k) - vector indices
    """
    return index.search(query_embedding, k)
```

### 10.2 Command Line Tools

#### bm25/run_bm25_search.py

```bash
python bm25/run_bm25_search.py \
  --queries <queries.tsv> \
  --index <bm25_index.pkl> \
  --k <num_results> \
  --mode <or|and> \
  --out <output.trec>
```

#### dense/search_dense_cli.py

```bash
python dense/search_dense_cli.py \
  --queries <queries.h5> \
  --index <faiss_index.bin> \
  --ids <passage_ids.pkl> \
  --k <num_results> \
  --out <output.trec>
```

#### generate_hybrid_runs.py

```bash
python generate_hybrid_runs.py \
  --bm25 <bm25_run.trec> \
  --dense <dense_run.trec> \
  --alpha <0.0-1.0> \
  --norm <minmax|z> \
  --topk <num_results> \
  --out <output.trec>
```

#### eval/evaluate_cli.py

```bash
python eval/evaluate_cli.py \
  --run <run.trec> \
  --qrels <qrels.tsv> \
  --queries_h5 <queries.h5> \
  --passages_h5 <passages.h5> \
  [--graded]
```

### 10.3 File Formats

#### TREC Run Format

```
query_id Q0 passage_id rank score run_name
1 Q0 12345 1 15.432 bm25_run
1 Q0 67890 2 14.231 bm25_run
```

#### Qrels Format

```
query_id 0 passage_id relevance
1 0 12345 1
1 0 67890 0
```

#### HDF5 Embeddings

```python
# Structure
{
    'id': np.array([...], dtype=int64),
    'embedding': np.array([...], shape=(N, 384), dtype=float32)
}
```

---

## 11. File Structure

```
Search-Engine/
├── README.md                   # Main documentation
├── DOCUMENTATION.md            # This file
├── EXPERIMENTAL_SETUP.md       # Assignment 3 setup
├── WEB_FRONTEND_GUIDE.md       # Web UI guide
│
├── bm25/                       # BM25 components
│   ├── Parse.cpp               # Collection parser
│   ├── BuildIndex.cpp          # Index builder
│   ├── QueryProcessing.cpp     # BM25 search
│   ├── SnippetExtractor.cpp    # Snippet generation
│   ├── build_bm25_from_subset.py
│   ├── run_bm25_search.py
│   └── bm25_subset_index.pkl   # Pre-built index (695 MB)
│
├── dense/                      # Dense retrieval
│   ├── build_faiss_hnsw.py     # Build HNSW index
│   ├── build_faiss_ivf.py      # Build IVF index
│   ├── search_dense_cli.py     # Search CLI
│   ├── faiss_hnsw_index.bin    # HNSW index (1.4 GB)
│   └── passage_ids.pkl         # ID mapping (8 MB)
│
├── data/                       # Data files
│   ├── collection.tsv          # 1M passages (symlink)
│   └── ms_marco/
│       ├── collection.tsv                          # Full collection
│       ├── msmarco_passages_subset.tsv             # 1M subset
│       ├── msmarco_passages_embeddings_subset.h5   # Embeddings
│       ├── msmarco_queries_dev_eval_embeddings.h5  # Query embeddings
│       ├── qrels.dev.tsv
│       ├── qrels.eval.one.tsv
│       └── qrels.eval.two.tsv
│
├── eval/                       # Evaluation
│   ├── evaluate_cli.py         # Evaluation script
│   ├── metrics.py              # IR metrics
│   └── *.json, *.tsv           # Results
│
├── runs/                       # TREC runs
│   ├── bm25.dev.trec
│   ├── dense.dev.trec
│   └── hybrid.dev.trec
│
├── analysis/                   # Analysis scripts
│   └── evaluate_runs.py        # Batch evaluation
│
├── search_server.py            # Web server
├── search_frontend.html        # Web UI
├── start_server.sh             # Startup script
├── hybrid_search.py            # Hybrid search
├── generate_hybrid_runs.py     # Generate hybrid runs
└── run_experiments.py          # Run all experiments
```

---

## 12. Performance Benchmarks

### 12.1 Index Sizes

| Component | Size | Format |
|-----------|------|--------|
| BM25 Index | 695 MB | Pickle |
| FAISS HNSW | 1.4 GB | Binary |
| Passage IDs | 8 MB | Pickle |
| Collection | 3.5 GB | TSV |
| Passage Embeddings | 1.5 GB | HDF5 |
| Query Embeddings | 15 MB | HDF5 |
| **Total** | **~7 GB** | |

### 12.2 Query Latency

| System | Average | P50 | P95 | P99 |
|--------|---------|-----|-----|-----|
| BM25 Python | 125ms | 100ms | 200ms | 350ms |
| Dense FAISS | 185ms | 150ms | 300ms | 450ms |
| Hybrid | 350ms | 280ms | 550ms | 750ms |

### 12.3 Index Building Time

| Operation | Time | Memory |
|-----------|------|--------|
| BM25 Build | 15min | 4 GB |
| FAISS HNSW Build | 8min | 6 GB |
| FAISS IVF Build | 5min | 4 GB |

### 12.4 Retrieval Quality

**Dev Set (6,980 queries):**

| System | MRR@10 | MAP | Recall@100 | Recall@1000 |
|--------|--------|-----|------------|-------------|
| BM25 | 0.325 | 0.299 | 0.723 | 0.951 |
| Dense | 0.338 | 0.310 | 0.789 | 0.942 |
| Hybrid | 0.362 | 0.326 | 0.812 | 0.968 |

**Eval Set (graded relevance):**

| System | MRR@10 | NDCG@10 | NDCG@100 |
|--------|--------|---------|----------|
| BM25 | 0.318 | 0.385 | 0.421 |
| Dense | 0.334 | 0.402 | 0.445 |
| Hybrid | 0.355 | 0.428 | 0.473 |

---

## 13. Troubleshooting

### 13.1 Common Issues

#### Index Not Found

```bash
# Error: FileNotFoundError: bm25_subset_index.pkl
# Solution:
cd bm25
python3 build_bm25_from_subset.py
```

#### FAISS Import Error

```bash
# Error: ModuleNotFoundError: No module named 'faiss'
# Solution:
pip install faiss-cpu
# Or for GPU:
pip install faiss-gpu
```

#### Memory Error During Index Build

```python
# Error: MemoryError
# Solution: Reduce batch size or use streaming
# In build_bm25_from_subset.py:
BATCH_SIZE = 10000  # Reduce from 100000
```

#### Port Already in Use

```bash
# Error: OSError: [Errno 48] Address already in use
# Solution:
lsof -ti:8080 | xargs kill -9
# Or change port in search_server.py:
PORT = 8081
```

#### Slow Search Performance

```python
# Problem: Searches taking >1 second
# Solutions:

# 1. Reduce k (number of results)
results = index.search(terms, k=10)  # Instead of k=1000

# 2. Use conjunctive mode for multi-term queries
results = index.search(terms, k=10, conjunctive=True)

# 3. For FAISS, reduce ef_search
index.hnsw.efSearch = 50  # Default: 200
```

### 13.2 Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# In search_server.py
logger.setLevel(logging.DEBUG)
```

### 13.3 Verification Steps

```bash
# 1. Check Python environment
python3 --version  # Should be 3.8+
pip list | grep -E 'faiss|numpy|h5py'

# 2. Verify indices exist
ls -lh bm25/bm25_subset_index.pkl
ls -lh dense/faiss_hnsw_index.bin

# 3. Test imports
python3 -c "
import faiss
import h5py
import pickle
import numpy as np
print('✓ All imports successful')
"

# 4. Run preflight check
./preflight_check.sh
```

---

## 14. Contributing & Development

### 14.1 Development Setup

```bash
# Clone with development branch
git clone -b Dense-Assignment-3 \
  https://github.com/vignesh362/Search-Engine.git

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests exist
```

### 14.2 Code Style

**Python:**
- PEP 8 style guide
- Type hints where appropriate
- Docstrings for all public functions

**C++:**
- Modern C++17 features
- RAII for resource management
- Const-correctness

### 14.3 Adding New Features

**Example: Add New Retrieval Method**

1. Create new file: `new_method/new_search.py`
2. Implement search interface:
```python
def search(query_text, k=100):
    # Your implementation
    return [(pid, score), ...]
```
3. Integrate into `search_server.py`:
```python
elif method == 'new_method':
    results = new_search(query_text, k)
```
4. Add UI option in `search_frontend.html`
5. Document in README

### 14.4 Testing

```python
# Unit test example
def test_bm25_search():
    index = BM25Index()
    # Add test documents
    index.add_document(1, "machine learning", ["machine", "learning"])
    index.finalize()
    
    # Test search
    results = index.search(["machine"], k=10)
    assert len(results) == 1
    assert results[0][0] == 1
```

### 14.5 Performance Profiling

```python
import cProfile
import pstats

# Profile BM25 search
cProfile.run('index.search(terms, k=100)', 'profile_stats')

# Analyze results
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative')
p.print_stats(20)
```

---

## Appendix A: Quick Reference

### Essential Commands

```bash
# Build indices
python3 bm25/build_bm25_from_subset.py
python3 dense/build_faiss_hnsw.py

# Start server
./start_server.sh

# Run evaluation
python3 analysis/evaluate_runs.py

# Generate hybrid run
python3 generate_hybrid_runs.py \
  --bm25 runs/bm25.dev.trec \
  --dense runs/dense.dev.trec \
  --out runs/hybrid.dev.trec
```

### Key Files

- **Main server:** `search_server.py`
- **Web UI:** `search_frontend.html`
- **BM25 index:** `bm25/bm25_subset_index.pkl`
- **FAISS index:** `dense/faiss_hnsw_index.bin`
- **Evaluation:** `analysis/evaluate_runs.py`

### Important Parameters

- **BM25:** k1=1.2, b=0.75
- **HNSW:** M=8, ef_construction=200, ef_search=200
- **Hybrid:** alpha=0.5-0.6 (dense weight)
- **Server:** PORT=8080

---

## Appendix B: References

### Papers

1. Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
2. Malkov & Yashunin (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
3. Bajaj et al. (2016). "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"

### Libraries

- **FAISS:** https://github.com/facebookresearch/faiss
- **NumPy:** https://numpy.org/
- **H5PY:** https://www.h5py.org/

### Resources

- MS MARCO Dataset: https://microsoft.github.io/msmarco/
- BM25 Implementation: Okapi BM25
- TREC Eval Format: https://trec.nist.gov/

---

**End of Documentation**

*For questions or issues, please contact the repository maintainer or open an issue on GitHub.*


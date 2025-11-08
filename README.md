# 🔍 MS MARCO Search Engine - Multi-Modal Retrieval System# 🔍 High-Performance Search Engine



A comprehensive search engine implementation featuring **BM25 (lexical)**, **Dense Neural Retrieval (semantic)**, and **Cascading Hybrid Search** on the MS MARCO passage ranking dataset. Built with C++ for performance-critical components and Python for machine learning integration.A complete, production-ready search engine implementation featuring BM25 ranking, compressed inverted index with VarByte encoding, query-dependent snippet generation, and a modern web interface. Built in C++ for performance and Python for serving.



![Python](https://img.shields.io/badge/Python-3.8+-blue)![Search Engine](https://img.shields.io/badge/Language-C%2B%2B17-blue)

![C++](https://img.shields.io/badge/C%2B%2B-17-green)![Python](https://img.shields.io/badge/Python-3.x-green)

![License](https://img.shields.io/badge/License-MIT-yellow)![License](https://img.shields.io/badge/License-MIT-yellow)



---## 🌟 Features



## 📋 Table of Contents### Core Search Engine

- **BM25 Ranking Algorithm**: State-of-the-art probabilistic ranking function with configurable k1 and b parameters

- [Overview](#-overview)- **Compressed Inverted Index**: VarByte encoding with block-based compression for efficient storage and retrieval

- [Architecture](#-architecture)- **Dual Query Modes**: 

- [Three Search Systems](#-three-search-systems)  - **Disjunctive (OR)**: Returns documents matching any query term

- [Installation](#-installation)  - **Conjunctive (AND)**: Returns only documents matching all query terms

- [Quick Start](#-quick-start)- **Scalable Architecture**: Handles millions of documents efficiently

- [Usage Guide](#-usage-guide)- **Query-Dependent Snippets**: Intelligent snippet generation with term highlighting

- [Evaluation Results](#-evaluation-results)

- [Analysis Tools](#-analysis-tools)### Web Interface

- [Project Structure](#-project-structure)- **Google-like UI**: Modern, responsive search interface with beautiful gradient design

- [Technical Details](#-technical-details)- **Real-time Search**: Instant results with highlighted query terms

- **Live System Logs**: Built-in console for monitoring search operations

---- **Configurable Results**: Adjustable number of results (5-100)

- **Mode Toggle**: Easy switching between OR/AND search modes

## 🌟 Overview

### Performance

This project implements and compares three different information retrieval approaches:- **Fast Indexing**: Multi-pass indexing with external merge sort

- **Optimized Queries**: Block-skipping and efficient decompression

1. **BM25 (Lexical)** - Classic term-based ranking with compressed inverted index- **Low Memory Footprint**: Streaming I/O for large-scale indexing

2. **Dense Neural Retrieval** - Semantic search using pre-trained embeddings and FAISS- **Production-Ready**: Compiled with -O3 optimizations

3. **Cascading Hybrid** - Two-stage retrieval combining BM25 candidate generation with dense reranking

## 📋 Table of Contents

### Key Features

- [Architecture](#-architecture)

✅ **High Performance** - C++ implementation for BM25 with VarByte compression  - [Installation](#-installation)

✅ **Semantic Search** - Dense embeddings with FAISS vector search (HNSW & IVF)  - [Quick Start](#-quick-start)

✅ **Hybrid Approach** - Cascading architecture: BM25 → Dense reranking  - [Usage Guide](#-usage-guide)

✅ **Comprehensive Evaluation** - MRR@10, MAP, NDCG@10, Recall@100 metrics  - [Index Building Pipeline](#-index-building-pipeline)

✅ **Analysis Tools** - Query-by-query comparison across all three systems  - [Query Processing](#-query-processing)

✅ **Web Interface** - Real-time search with all three methods  - [Web Interface](#-web-interface)

- [File Structure](#-file-structure)

---- [Technical Details](#-technical-details)

- [Configuration](#-configuration)

## 🏗️ Architecture- [Troubleshooting](#-troubleshooting)



```## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────────────┐

│                         SEARCH SYSTEMS                              │```

├─────────────────────────────────────────────────────────────────────┤┌─────────────────────────────────────────────────────────────┐

│                                                                     ││                    INDEXING PIPELINE                        │

│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐    │├─────────────────────────────────────────────────────────────┤

│  │     BM25     │      │    DENSE     │      │   HYBRID     │    ││                                                             │

│  │  (Lexical)   │      │  (Semantic)  │      │ (Cascading)  │    ││  collection.tsv                                             │

│  └──────────────┘      └──────────────┘      └──────────────┘    ││       │                                                      │

│         │                      │                      │            ││       ▼                                                      │

│         │                      │                      │            ││  ┌─────────┐     tmp_postings/                             │

│  ┌──────▼──────┐      ┌───────▼────────┐    ┌───────▼────────┐  ││  │  Parse  │ ──▶ postings_*.tmp                            │

│  │  Inverted   │      │ FAISS Index    │    │  BM25 + Dense  │  ││  └─────────┘     (sorted runs)                             │

│  │   Index     │      │  (HNSW/IVF)    │    │   Reranking    │  ││       │                                                      │

│  │ (VarByte)   │      │  384-D vectors │    │                │  ││       ▼                                                      │

│  └─────────────┘      └────────────────┘    └────────────────┘  ││  ┌─────────────┐   index_output/                           │

│                                                                     ││  │ BuildIndex  │──▶ invlists.bin    (VarByte compressed)   │

└─────────────────────────────────────────────────────────────────────┘│  └─────────────┘   lexicon.tsv     (term → block mapping)  │

│                    lastdocid.bin   (block metadata)         │

┌─────────────────────────────────────────────────────────────────────┐│                    docidsize.bin   (block sizes)            │

│                     DATA FLOW: HYBRID SEARCH                        ││                    freqsize.bin    (frequency sizes)        │

├─────────────────────────────────────────────────────────────────────┤│       │                                                      │

│                                                                     ││       ▼                                                      │

│  Query: "machine learning algorithms"                              ││  ┌──────────────────┐  index_output/                       │

│    │                                                                ││  │BuildPassageStore │─▶ offsets.bin  (doc offsets)         │

│    ├─► BM25 Search                                                 ││  └──────────────────┘   texts.bin    (original text)       │

│    │   └─► Top-1000 Candidates (term matching)                    │└─────────────────────────────────────────────────────────────┘

│    │         │                                                      │

│    │         ▼                                                      │┌─────────────────────────────────────────────────────────────┐

│    └─► Dense Reranking                                             ││                    QUERY PIPELINE                           │

│        └─► Top-1000 Results (semantic similarity)                  │├─────────────────────────────────────────────────────────────┤

│                                                                     ││                                                             │

│  Benefits:                                                          ││  User Query                                                 │

│  • Fast candidate generation (BM25)                                ││       │                                                      │

│  • Semantic reranking improves relevance (Dense)                   ││       ▼                                                      │

│  • Best of both worlds!                                            ││  ┌──────────────────┐                                       │

│                                                                     ││  │ QueryProcessing  │──▶ Ranked Doc IDs + BM25 Scores      │

└─────────────────────────────────────────────────────────────────────┘│  └──────────────────┘    (DAAT with skipping)              │

```│       │                                                      │

│       ▼                                                      │

---│  ┌──────────────────┐                                       │

│  │SnippetExtractor  │──▶ Highlighted Snippets              │

## 🔍 Three Search Systems│  └──────────────────┘    (window-based scoring)            │

│       │                                                      │

### 1. BM25 (Baseline - Lexical Matching)│       ▼                                                      │

│  JSON Results                                               │

**How it works:**└─────────────────────────────────────────────────────────────┘

- Classic probabilistic ranking function

- Term-based matching with TF-IDF weighting┌─────────────────────────────────────────────────────────────┐

- Fast retrieval using compressed inverted index│                    WEB INTERFACE                            │

├─────────────────────────────────────────────────────────────┤

**Strengths:**│                                                             │

- Very fast query processing│  Browser ◄──▶ search_server.py ◄──▶ QueryProcessing       │

- Handles exact term matching well│              (HTTP Server)      │                          │

- No training required│                                  └──▶ SnippetExtractor      │

│                                                             │

**Weaknesses:**│  • search_frontend.html (Google-like UI)                   │

- Vocabulary mismatch problem│  • Real-time logs and monitoring                           │

- No semantic understanding│  • OR/AND mode toggle                                      │

- Can't handle synonyms or paraphrasing└─────────────────────────────────────────────────────────────┘

```

**Parameters:**

- `k1=0.9` - Term saturation parameter## 🚀 Installation

- `b=0.4` - Length normalization

### Prerequisites

### 2. Dense Neural Retrieval (Semantic Matching)

- **C++ Compiler**: g++ or clang++ with C++17 support

**How it works:**- **Python**: 3.x (for web server)

- Pre-trained neural embeddings (384-D vectors)- **Operating System**: Linux, macOS, or WSL on Windows

- Semantic similarity via dot product

- FAISS indexing for fast approximate search### Building the Project



**Strengths:**```bash

- Understands semantic meaning# Clone the repository

- Handles synonyms and paraphrasinggit clone https://github.com/yourusername/Search-Engine.git

- Better for conceptual queriescd Search-Engine



**Weaknesses:**# Compile all C++ components

- Slower than BM25g++ -std=c++17 -O3 -Wall -Wextra -o Parse Parse.cpp

- May miss exact term matchesg++ -std=c++17 -O3 -Wall -Wextra -o BuildIndex BuildIndex.cpp

- Requires pre-computed embeddingsg++ -std=c++17 -O3 -Wall -Wextra -o BuildPassageStore BuildPassageStore.cpp

g++ -std=c++17 -O3 -Wall -Wextra -o QueryProcessing QueryProcessing.cpp

**Index Types:**g++ -std=c++17 -O3 -Wall -Wextra -o SnippetExtractor SnippetExtractor.cpp

- **HNSW** (Hierarchical Navigable Small World) - High recall, more memoryg++ -std=c++17 -O3 -Wall -Wextra -o InspectPostings InspectPostings.cpp

- **IVF** (Inverted File) - Faster, lower memory, slightly lower recall

# Make scripts executable

### 3. Cascading Hybrid (Best of Both Worlds)chmod +x build_search_index.sh

chmod +x search_with_snippets.sh

**How it works:**chmod +x search_server.py

1. **Stage 1:** BM25 retrieves top-K candidates (fast, broad coverage)```

2. **Stage 2:** Dense reranking of candidates (semantic refinement)

## ⚡ Quick Start

**Strengths:**

- Combines speed of BM25 with accuracy of dense retrieval### 1. Prepare Your Data

- Better than either system alone

- Industry-standard approachYour collection should be in TSV format (`collection.tsv`):

```

**Configuration:**docID    text

- `candidate_k=1000` - Number of BM25 candidates0        This is the first document about machine learning.

- `topk=1000` - Final results after dense reranking1        Search engines use inverted indices for fast retrieval.

2        BM25 is a popular ranking function in information retrieval.

---```



## 🚀 Installation### 2. Build the Index



### PrerequisitesUse the automated build script (recommended):



```bash```bash

# Python 3.8+./build_search_index.sh

python3 --version```



# C++ compiler with C++17 supportOr build manually:

g++ --version

``````bash

# Step 1: Parse collection and create intermediate postings

### Setup./Parse collection.tsv tmp_postings 128



```bash# Step 2: Merge and compress postings into final index

# 1. Clone the repository./BuildIndex tmp_postings index_output 128

git clone https://github.com/vignesh362/Search-Engine.git

cd Search-Engine# Step 3: Build passage store for snippets

./BuildPassageStore collection.tsv index_output

# 2. Create virtual environment```

python3 -m venv search_engine_env

source search_engine_env/bin/activate### 3. Start Searching



# 3. Install Python dependencies**Option A: Web Interface (Recommended)**

pip install numpy scipy h5py faiss-cpu tqdm

```bash

# 4. Compile C++ components# Start the web server

cd bm25python3 search_server.py

./build_search_index.sh

cd ..# Open browser to http://localhost:8080

``````



### Data Setup**Option B: Command Line**



The project expects MS MARCO data in `data/ms_marco/`:```bash

- `collection.tsv` - Passage collection# Simple search

- `queries.dev.tsv`, `queries.eval*.tsv` - Query sets./QueryProcessing university research

- `qrels.dev.tsv`, `qrels.eval*.tsv` - Relevance judgments

- `msmarco_passages_embeddings_subset.h5` - Passage embeddings# Search with custom top-k

- `msmarco_queries_dev_eval_embeddings.h5` - Query embeddings./QueryProcessing machine learning -k 20



---# Conjunctive (AND) search

./QueryProcessing data science -and

## ⚡ Quick Start

# Complete search with snippets

### Step 1: Build BM25 Index./search_with_snippets.sh "artificial intelligence" 10

```

```bash

# Build inverted index from collection## 📖 Usage Guide

cd bm25

./BuildIndex --collection ../data/collection.tsv --output ./index_output### Index Building Pipeline



# Build passage store for snippet generation#### Parse

./BuildPassageStore --collection ../data/collection.tsv --output ./index_outputParses the document collection and creates sorted intermediate posting files.

cd ..

``````bash

./Parse <collection.tsv> <output_dir> [memory_limit_MB]

### Step 2: Generate BM25 Runs```



```bash**Arguments:**

# Generate runs for all three datasets- `collection.tsv`: Input document collection (docID \t text format)

python3 bm25/generate_all_bm25_runs.py- `output_dir`: Directory for intermediate posting files (e.g., `tmp_postings`)

```- `memory_limit_MB`: Optional memory limit in MB (default: 128)



Output: `runs/bm25.dev.trec`, `runs/bm25.eval1.trec`, `runs/bm25.eval2.trec`**Output:** Multiple `postings_*.tmp` files in sorted order



### Step 3: Generate Dense Runs**Features:**

- Term normalization (lowercase, alphanumeric only)

```bash- External sorting for scalability

# Generate dense retrieval runs using FAISS- Memory-efficient streaming

python3 generate_dense_runs.py

```#### BuildIndex

Merges sorted runs and builds the compressed inverted index.

Output: `runs/dense.dev.trec`, `runs/dense.eval1.trec`, `runs/dense.eval2.trec`

```bash

### Step 4: Generate Hybrid Runs (Cascading)./BuildIndex <tmp_runs_dir> <index_output_dir> [block_size]

```

```bash

# Generate hybrid runs: BM25 candidates + Dense reranking**Arguments:**

python3 generate_hybrid_runs.py- `tmp_runs_dir`: Directory containing `postings_*.tmp` files

```- `index_output_dir`: Output directory for final index files

- `block_size`: Postings per block (default: 128)

Output: `runs/hybrid.dev.trec`, `runs/hybrid.eval1.trec`, `runs/hybrid.eval2.trec`

**Output:**

### Step 5: Evaluate All Systems- `invlists.bin`: Compressed posting lists (VarByte encoded)

- `lexicon.tsv`: Term dictionary with metadata

```bash- `lastdocid.bin`: Last docID per block

# Compare metrics across all three systems- `docidsize.bin`: Compressed docID block sizes

python3 analysis/evaluate_runs.py- `freqsize.bin`: Compressed frequency block sizes

```

**Features:**

---- K-way merge using min-heap

- VarByte compression with delta encoding

## 📊 Evaluation Results- Block-based organization for efficient query processing



### Metrics Explained#### BuildPassageStore

Creates the passage store for snippet extraction.

- **MRR@10** (Mean Reciprocal Rank) - Average of 1/rank of first relevant result (higher is better)

- **MAP** (Mean Average Precision) - Average precision across all recall levels (higher is better)```bash

- **Recall@100** - Percentage of relevant docs found in top-100 (higher is better)./BuildPassageStore <collection.tsv> <index_output_dir>

- **NDCG@K** (Normalized Discounted Cumulative Gain) - Ranking quality metric (higher is better)```



### Performance Summary**Output:**

- `offsets.bin`: Document byte offsets (uint64_t array)

**Key Findings:**- `texts.bin`: Original document texts (UTF-8)

- ✅ Dense retrieval outperforms BM25 by ~35% on MRR@10

- ✅ Hybrid (cascading) is best overall, 5-10% better than dense alone### Query Processing

- ✅ Dense has significantly better recall (~70-85% vs 55-75% for BM25)

- ✅ Hybrid combines speed of BM25 with accuracy of dense retrieval#### QueryProcessing

The core search engine that returns ranked documents.

---

```bash

## 🔬 Analysis Tools./QueryProcessing <term1> <term2> ... [options]

```

### 1. Evaluate All Runs

**Options:**

Compare metrics across all systems:- `-k <N>`: Return top-N results (default: 10)

- `-and`: Use conjunctive (AND) mode (default: disjunctive/OR)

```bash- `--k1 <val>`: BM25 k1 parameter (default: 1.2)

python3 analysis/evaluate_runs.py- `--b <val>`: BM25 b parameter (default: 0.75)

```

**Examples:**

Shows MRR@10, MAP, Recall@100, NDCG@10, NDCG@100 for each system.```bash

# Find top 10 documents with "machine" OR "learning"

### 2. Query-by-Query Comparison./QueryProcessing machine learning



Analyze differences between systems for 20 custom test queries:# Find top 20 documents with "data" AND "science"

./QueryProcessing data science -k 20 -and

```bash

python3 compare_queries_compact.py# Custom BM25 parameters

```./QueryProcessing search engine --k1 1.5 --b 0.8

```

**Output:**

- One line per query showing overlaps and correlations**Output Format:**

- Summary statistics by query length```

- Notable queries (highest/lowest agreement)Query: machine learning

Mode: disjunctive (OR)

**Example Output:**---

```Top-10 results:

#   Len  Query                           BM25↔Dense    BM25↔Hybrid   Dense↔Hybrid 1. doc=12345 score=8.4521

1   2    machine learning                0/10 (0%)     0/10 (0%)     1/10 (10%) 2. doc=67890 score=7.9832

2   4    best laptop for programming     1/10 (10%)    2/10 (20%)    5/10 (50%) ...

...```

```

#### SnippetExtractor

### 3. Detailed System ComparisonGenerates query-dependent snippets with term highlighting.



In-depth comparison with document-level analysis:```bash

./SnippetExtractor --store-dir <dir> --docs <docIDs> --query <query> [options]

```bash```

python3 compare_search_systems.py --num-queries 20

```**Options:**

- `--store-dir`: Directory containing `offsets.bin` and `texts.bin` (default: `index_output`)

**Features:**- `--docs`: Comma-separated document IDs (e.g., "12,45,67")

- Side-by-side document comparison- `--docs-file`: File containing document IDs (one per line or CSV)

- Overlap analysis- `--query`: Query string for highlighting

- Rank correlation metrics- `--windowsz`: Tokens per window (default: 40)

- Unique document identification- `--per`: Snippets per document (default: 2)

- Score distribution statistics

**Example:**

### 4. Custom Query Testing```bash

./SnippetExtractor --store-dir index_output \

Test your own queries by modifying `CUSTOM_QUERIES` in `compare_queries_compact.py`    --docs "1234,5678,9012" \

    --query "machine learning" \

---    --windowsz 50 \

    --per 2

## 📁 Project Structure```



```**Output Format (JSON):**

Search-Engine/```json

├── README.md                          # This file{"docID":1234,"snippets":["...text with <b>machine</b> <b>learning</b>...","...another snippet..."]}

├── bm25/                              # BM25 implementation{"docID":5678,"snippets":["...more results..."]}

│   ├── BuildIndex.cpp                 # Build inverted index```

│   ├── QueryProcessing.cpp            # Query search engine

│   ├── SnippetExtractor.cpp           # Snippet generation### Complete Search Pipeline

│   ├── build_bm25_index.py           # Python wrapper

│   ├── generate_all_bm25_runs.py     # Generate all BM25 runsUse the convenience script for end-to-end search:

│   └── index_output/                  # Built indices

│       ├── invlists.bin              # Compressed inverted lists```bash

│       ├── lexicon.tsv               # Term dictionary./search_with_snippets.sh "query terms" [top-k]

│       └── offsets.bin               # Document offsets```

├── dense/                             # Dense retrieval

│   ├── build_faiss_hnsw.py           # Build HNSW index**Example:**

│   ├── build_faiss_ivf.py            # Build IVF index```bash

│   └── search_dense_cli.py           # Dense search tool./search_with_snippets.sh "information retrieval" 10

├── runs/                              # Generated run files```

│   ├── bm25.*.trec                   # BM25 runs

│   ├── dense.*.trec                  # Dense runsThis script:

│   └── hybrid.*.trec                 # Hybrid runs1. Runs `QueryProcessing` to get ranked documents

├── data/                              # Data directory2. Extracts document IDs from results

│   └── ms_marco/                      # MS MARCO dataset3. Runs `SnippetExtractor` to generate highlighted snippets

│       ├── collection.tsv            # Passages4. Outputs complete search results with snippets

│       ├── queries.*.tsv             # Queries

│       ├── qrels.*.tsv               # Relevance judgments## 🌐 Web Interface

│       └── *.h5                      # Embeddings

├── eval/                              # Evaluation### Starting the Server

│   ├── metrics.py                    # Metrics implementation

│   └── evaluate_cli.py               # Evaluation CLI```bash

├── analysis/                          # Analysis tools# Start on default port 8080

│   └── evaluate_runs.py              # Compare all systemspython3 search_server.py

├── runs/                              # Run utilities

│   └── run_io.py                     # TREC format I/O, hybrid logic# Or make it executable

├── generate_dense_runs.py            # Generate dense runschmod +x search_server.py

├── generate_hybrid_runs.py           # Generate hybrid runs./search_server.py

├── hybrid_search.py                  # Hybrid search CLI```

├── compare_queries_compact.py        # Query comparison (compact)

├── compare_search_systems.py         # Detailed comparisonThen open your browser to: **http://localhost:8080**

├── search_server.py                  # Web server

└── search_frontend.html              # Web interface### Features

```

- **Modern UI**: Google-inspired design with smooth animations

---- **Smart Search**: 

  - OR mode: Find documents with any query term

## 🛠️ Usage Guide  - AND mode: Find documents with all query terms

- **Configurable Results**: Choose 5, 10, 20, 30, 50, or 100 results

### BM25 Search- **Live Logs**: Real-time system monitoring and debugging

- **Instant Results**: AJAX-based search with loading indicators

```bash- **Highlighted Snippets**: Query terms are highlighted in bold

# Search with BM25

cd bm25### API Endpoint

./QueryProcessing --index-dir ./index_output \

                  --query "machine learning" \The server exposes a REST API:

                  --k 10 \

                  --mode disjunctive**POST /search**

```

Request:

**Modes:**```json

- `disjunctive` (OR) - Match any term (default){

- `conjunctive` (AND) - Match all terms  "query": "machine learning",

  "topk": 10,

### Dense Search  "conjunctive": false

}

```bash```

# Search with dense embeddings

python3 dense/search_dense_cli.py \Response:

    --index dense/msmarco.faiss.hnsw \```json

    --embeddings data/ms_marco/msmarco_passages_embeddings_subset.h5 \{

    --query "machine learning" \  "query": "machine learning",

    --k 10  "count": 10,

```  "results": [

    {

### Hybrid Search      "docID": 12345,

      "score": 8.4521,

```bash      "snippets": [

# Hybrid search: BM25 candidates + dense reranking        "...text about <b>machine</b> <b>learning</b>...",

python3 hybrid_search.py \        "...another relevant snippet..."

    --bm25 runs/bm25.dev.trec \      ]

    --queries_h5 data/ms_marco/msmarco_queries_dev_eval_embeddings.h5 \    },

    --passages_h5 data/ms_marco/msmarco_passages_embeddings_subset.h5 \    ...

    --candidate_k 1000 \  ]

    --topk 1000 \}

    --out runs/hybrid.output.trec```

```

### Customization

### Generate All Runs

**Change Port:**

```bashEdit `search_server.py`:

# Activate virtual environment```python

source search_engine_env/bin/activaterun_server(port=8080)  # Change to your preferred port

```

# Generate runs for all three systems on all datasets

python3 bm25/generate_all_bm25_runs.py**Adjust Snippet Parameters:**

python3 generate_dense_runs.pyEdit snippet extraction parameters in `search_server.py`:

python3 generate_hybrid_runs.py```python

```snippet_cmd = [

    './SnippetExtractor',

### Web Interface    '--store-dir', 'index_output',

    '--docs', doc_ids_str,

```bash    '--query', query,

# Start web server    '--windowsz', '50',  # Tokens per snippet

python3 search_server.py    '--per', '2'         # Snippets per document

]

# Open browser to:```

# http://localhost:8000

```## 📁 File Structure



**Features:**```

- Search with all three systemsSearch-Engine/

- Real-time results├── 📄 README.md                          # This file

- Snippet highlighting├── 📄 GENIE_SEARCH_README.md            # Web interface documentation

- System comparison├── 📄 collection.tsv                     # Document collection (TSV format)

│

---├── 🔧 C++ Source Files

│   ├── Parse.cpp                         # Stage 1: Parse collection → posting runs

## 🔧 Technical Details│   ├── BuildIndex.cpp                    # Stage 2: Merge runs → compressed index

│   ├── BuildPassageStore.cpp            # Stage 3: Build passage store

### BM25 Implementation│   ├── QueryProcessing.cpp              # Query engine (BM25 ranking)

│   ├── SnippetExtractor.cpp             # Snippet generation

**Inverted Index:**│   └── InspectPostings.cpp              # Index inspection tool

- VarByte compression for posting lists│

- Block-based compression├── 🐍 Python Files

- Skip pointers for fast traversal│   └── search_server.py                  # HTTP server for web interface

│

**Scoring Formula:**├── 🌐 Web Interface

```│   └── search_frontend.html             # Google-like search UI

BM25(d,q) = Σ IDF(qi) · (f(qi,d) · (k1+1)) / (f(qi,d) + k1·(1-b+b·|d|/avgdl))│

```├── 🛠️ Scripts

│   ├── build_search_index.sh            # Automated index building

Where:│   └── search_with_snippets.sh          # CLI search with snippets

- `f(qi,d)` = term frequency of qi in document d│

- `|d|` = document length├── 📦 Compiled Binaries

- `avgdl` = average document length│   ├── Parse                            # Compiled parser

- `k1=0.9` = term saturation│   ├── BuildIndex                       # Compiled index builder

- `b=0.4` = length normalization│   ├── BuildPassageStore               # Compiled passage store builder

│   ├── QueryProcessing                 # Compiled query processor

### Dense Retrieval│   ├── SnippetExtractor                # Compiled snippet extractor

│   └── InspectPostings                 # Compiled inspection tool

**Embeddings:**│

- Pre-trained sentence transformers├── 📁 tmp_postings/                     # Intermediate posting files (temporary)

- 384-dimensional vectors│   └── postings_*.tmp

- Normalized for cosine similarity│

└── 📁 index_output/                     # Final index files

**FAISS Indices:**    ├── invlists.bin                     # Compressed inverted lists

    ├── lexicon.tsv                      # Term dictionary

**HNSW (Hierarchical Navigable Small World):**    ├── lastdocid.bin                    # Block metadata (last docID)

```python    ├── docidsize.bin                    # Block metadata (docID size)

index = faiss.IndexHNSWFlat(384, 32)    ├── freqsize.bin                     # Block metadata (frequency size)

index.hnsw.efConstruction = 200    ├── offsets.bin                      # Document offsets

index.hnsw.efSearch = 128    └── texts.bin                        # Document texts

``````

- High recall (>95%)

- Fast approximate search## 🔬 Technical Details

- More memory usage

### Index Format

**IVF (Inverted File):**

```python#### Lexicon (lexicon.tsv)

quantizer = faiss.IndexFlatIP(384)TSV file mapping terms to their posting list metadata:

index = faiss.IndexIVFFlat(quantizer, 384, 100)```

index.nprobe = 10term    start_slot    end_slot    first_block    num_blocks    ft

```machine    0            1250        0              10            1250

- Faster searchlearning   1250         2100        10             7             850

- Lower memory```

- Slightly lower recall (~90-95%)

**Fields:**

### Hybrid Cascading- `term`: The indexed term (normalized)

- `start_slot`: Global posting slot start (inclusive)

**Algorithm:**- `end_slot`: Global posting slot end (exclusive)

```python- `first_block`: Index of first block in metadata arrays

def cascading_hybrid(query, candidate_k=1000, topk=1000):- `num_blocks`: Number of blocks for this term

    # Stage 1: BM25 candidate generation- `ft`: Document frequency (number of postings)

    candidates = bm25_search(query, k=candidate_k)

    #### Inverted Lists (invlists.bin)

    # Stage 2: Dense rerankingBinary file containing compressed posting lists:

    query_vec = encode_query(query)- **Block Structure**: Each block contains up to B postings (default: 128)

    scores = []- **Compression**: VarByte encoding with delta compression

    for doc_id in candidates:- **Layout**: [DocID Block][Frequency Block] for each block

        doc_vec = get_passage_embedding(doc_id)

        score = dot_product(query_vec, doc_vec)#### Metadata Arrays

        scores.append((doc_id, score))Three parallel arrays for block information:

    - `lastdocid.bin`: Last docID in each block (for skipping)

    # Return top-k after reranking- `docidsize.bin`: Size in bytes of each docID block

    scores.sort(reverse=True)- `freqsize.bin`: Size in bytes of each frequency block

    return scores[:topk]

```#### Passage Store

- `offsets.bin`: uint64_t array where `offsets[i]` = byte start of doc i in texts.bin

**Benefits:**- `texts.bin`: Concatenated document texts (UTF-8)

- Fast: BM25 quickly narrows down candidates

- Accurate: Dense reranking improves relevance### BM25 Ranking

- Scalable: Only reranks top candidates, not entire corpus

The search engine uses the BM25 formula:

**Implementation:**

See `runs/run_io.py` → `rerank_with_dense()` function```

score(Q,d) = Σ IDF(qᵢ) · (f(qᵢ,d) · (k₁ + 1)) / (f(qᵢ,d) + k₁ · (1 - b + b · |d|/avgdl))

---```



## 📈 Performance ComparisonWhere:

- `Q`: Query terms

### Overlap Analysis (20 Test Queries)- `d`: Document

- `f(qᵢ,d)`: Frequency of term qᵢ in document d

Average document overlap in top-10 results:- `|d|`: Document length

- `avgdl`: Average document length

| Comparison      | Overlap | Interpretation                           |- `k₁`: Term frequency saturation parameter (default: 1.2)

|-----------------|---------|------------------------------------------|- `b`: Length normalization factor (default: 0.75)

| BM25 ↔ Dense    | 9.5%    | Very different retrieval strategies      |- `IDF(qᵢ)`: Inverse document frequency of term qᵢ

| BM25 ↔ Hybrid   | 15.5%   | Hybrid changes BM25 ranking significantly|

| Dense ↔ Hybrid  | 41.5%   | Hybrid preserves dense scoring           |**IDF Formula:**

```

### Rank CorrelationIDF(qᵢ) = ln((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5))

```

Average rank correlation (1.0 = perfect agreement):

Where:

| Comparison      | Correlation | Interpretation                    |- `N`: Total number of documents

|-----------------|-------------|-----------------------------------|- `n(qᵢ)`: Number of documents containing term qᵢ

| BM25 ↔ Dense    | 0.150       | Minimal ranking agreement         |

| BM25 ↔ Hybrid   | 0.306       | Some structure preserved          |### Query Processing Algorithms

| Dense ↔ Hybrid  | 0.699       | Strong agreement (expected)       |

**Document-at-a-Time (DAAT):**

### Query Length Impact- Process one document at a time across all query terms

- Use min-heap to efficiently advance through posting lists

BM25↔Hybrid overlap by query length:- Apply block-skipping optimization for faster traversal



- **Short queries (1-2 words):** 12.5% overlap**Block Skipping:**

- **Medium queries (3-7 words):** 21.2% overlap ⭐ Best!- Check `lastdocid` of current block

- **Long queries (8+ words):** 11.2% overlap- Skip entire block if lastdocid < target

- Reduces decompression overhead

**Insight:** Medium-length queries show best agreement, suggesting they provide enough context for both term-based and semantic matching.

**Conjunctive Mode:**

---- Only score documents that contain ALL query terms

- Early termination if any term's posting list exhausted

## 🎯 Key Takeaways

**Disjunctive Mode:**

### When to Use Each System- Score documents that contain ANY query term

- Aggregate scores across all matching terms

**Use BM25 when:**

- Need very fast search (<10ms)### Snippet Generation

- Queries have specific terminology

- Exact term matching is important**Window Scoring:**

- Resource-constrained environmentSnippets are scored based on:

1. **Coverage**: Unique query terms matched

**Use Dense when:**2. **Density**: Total query term occurrences

- Semantic understanding is crucial3. **Proximity**: Inverse of span (first match to last match)

- Handling synonyms and paraphrasing4. **Position**: Slight bonus for earlier windows

- Conceptual queries

- Have pre-computed embeddings**Highlighting:**

- Case-insensitive matching

**Use Hybrid when:**- Exact word boundaries

- Want best overall performance- HTML `<b>` tags for highlighting

- Can afford two-stage retrieval

- Need balance of speed and accuracy## ⚙️ Configuration

- Production deployment (industry standard)

### BM25 Parameters

### System Characteristics

Default values work well for most collections, but can be tuned:

| Aspect              | BM25    | Dense   | Hybrid  |

|---------------------|---------|---------|---------|```bash

| Speed               | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | ⭐⭐⭐⭐  |# Lower k1 (0.8-1.0): Penalize term frequency saturation more

| Accuracy            | ⭐⭐⭐   | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ |# Higher k1 (1.5-2.0): Reward high-frequency terms more

| Semantic Understand | ⭐      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |./QueryProcessing machine learning --k1 1.5

| Resource Usage      | ⭐      | ⭐⭐⭐⭐  | ⭐⭐⭐   |

| Setup Complexity    | ⭐      | ⭐⭐⭐   | ⭐⭐⭐   |# Lower b (0.5-0.6): Reduce length normalization

# Higher b (0.8-0.9): Increase length normalization

---./QueryProcessing machine learning --b 0.8

```

## 📚 References

### Block Size

- **MS MARCO:** [microsoft.github.io/msmarco](https://microsoft.github.io/msmarco/)

- **BM25 Paper:** Robertson & Zaragoza (2009)Larger blocks = Better compression, slower query processing

- **FAISS:** [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)Smaller blocks = Worse compression, faster queries with skipping

- **Dense Retrieval:** Karpukhin et al. (2020) - Dense Passage Retrieval

- **Hybrid Search:** Lin et al. (2021) - Pyserini```bash

# Small blocks (64-128): Good for interactive search

---./BuildIndex tmp_postings index_output 64



## 👤 Author# Large blocks (256-512): Good for batch processing

./BuildIndex tmp_postings index_output 256

**Vignesh Shanmugasundaram**```

- GitHub: [@vignesh362](https://github.com/vignesh362)

- Repository: [Search-Engine](https://github.com/vignesh362/Search-Engine)### Memory Limits



---Control memory usage during parsing:



## 📄 License```bash

# Small memory (64 MB): More intermediate files

MIT License - See LICENSE file for details./Parse collection.tsv tmp_postings 64



---# Large memory (256 MB): Fewer intermediate files, faster

./Parse collection.tsv tmp_postings 256

## 🙏 Acknowledgments```



- MS MARCO team for the dataset## 🐛 Troubleshooting

- FAISS team for the vector search library

- sentence-transformers for pre-trained embeddings### Compilation Issues



---**Error: "C++17 required"**

```bash

## 🐛 Troubleshooting# Ensure you use -std=c++17 flag

g++ -std=c++17 -O3 -o QueryProcessing QueryProcessing.cpp

### Common Issues```



**"Module not found" errors:****Error: "filesystem not found"**

```bash```bash

# Make sure virtual environment is activated# On older systems, you may need to link filesystem library

source search_engine_env/bin/activateg++ -std=c++17 -O3 -o Parse Parse.cpp -lstdc++fs

```

# Reinstall dependencies

pip install numpy scipy h5py faiss-cpu tqdm### Index Building Issues

```

**Error: "No postings_*.tmp found"**

**C++ compilation errors:**- Ensure Parse completed successfully

```bash- Check that `tmp_postings` directory exists and has files

# Ensure C++17 support- Run `ls -la tmp_postings/` to verify

g++ --version  # Should be 7.0+

**Error: "Cannot write lexicon.tsv"**

# Recompile with verbose output- Ensure `index_output` directory exists: `mkdir -p index_output`

cd bm25- Check disk space: `df -h .`

g++ -std=c++17 -O3 -o BuildIndex BuildIndex.cpp- Verify write permissions: `ls -ld index_output`

```

### Query Issues

**FAISS import errors:**

```bash**No results found**

# Install CPU version of FAISS- Terms are case-sensitive during indexing but normalized to lowercase

pip install faiss-cpu- Try single-word queries first

- Check lexicon: `grep "yourterm" index_output/lexicon.tsv`

# Or GPU version (if CUDA available)- Use InspectPostings to debug: `./InspectPostings yourterm`

pip install faiss-gpu

```**Slow queries**

- Reduce block size for faster query processing

**Embedding files not found:**- Ensure you're using `-O3` optimization flag

```bash- Check if index files are on SSD vs HDD

# Check data directory structure

ls -lh data/ms_marco/*.h5### Web Server Issues



# Files should be:**Error: "QueryProcessing executable not found"**

# - msmarco_passages_embeddings_subset.h5```bash

# - msmarco_queries_dev_eval_embeddings.h5# Ensure executables are compiled and in current directory

```ls -l QueryProcessing SnippetExtractor

# Should show executable files (rwxr-xr-x)

---```



**Happy Searching! 🚀****Error: "Port already in use"**

```bash
# Find and kill process using port 8080
lsof -ti:8080 | xargs kill -9
# Or change port in search_server.py
```

**Error: "search_frontend.html not found"**
- Ensure you're running `search_server.py` from the project root directory
- Check: `ls -l search_frontend.html`

### Performance Issues

**Slow indexing**
- Increase memory limit: `./Parse collection.tsv tmp_postings 512`
- Use SSD for temp files
- Ensure sufficient disk space (3-4x collection size)

**High memory usage during queries**
- Reduce top-k value
- Use conjunctive mode for multi-term queries
- Rebuild index with smaller blocks

## 📊 Performance Benchmarks

Typical performance on modern hardware (SSD, 16GB RAM):

| Collection Size | Index Time | Index Size | Query Time (avg) |
|----------------|-----------|------------|------------------|
| 100K docs      | ~30s      | ~50 MB     | <10 ms           |
| 1M docs        | ~5 min    | ~500 MB    | ~20 ms           |
| 10M docs       | ~45 min   | ~5 GB      | ~50 ms           |

*Query times for top-10 with 2-3 term queries*

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Positional index for phrase queries
- [ ] Query expansion and relevance feedback
- [ ] Parallel index building
- [ ] Distributed query processing
- [ ] More sophisticated snippet ranking
- [ ] Support for additional languages
- [ ] Document metadata indexing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- BM25 algorithm from Robertson & Zaragoza (2009)
- VarByte compression technique from Williams & Zobel
- Modern search engine architecture inspired by industry best practices

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Happy Searching! 🔍✨**


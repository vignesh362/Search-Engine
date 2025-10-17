# 🔍 High-Performance Search Engine

A complete, production-ready search engine implementation featuring BM25 ranking, compressed inverted index with VarByte encoding, query-dependent snippet generation, and a modern web interface. Built in C++ for performance and Python for serving.

![Search Engine](https://img.shields.io/badge/Language-C%2B%2B17-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### Core Search Engine
- **BM25 Ranking Algorithm**: State-of-the-art probabilistic ranking function with configurable k1 and b parameters
- **Compressed Inverted Index**: VarByte encoding with block-based compression for efficient storage and retrieval
- **Dual Query Modes**: 
  - **Disjunctive (OR)**: Returns documents matching any query term
  - **Conjunctive (AND)**: Returns only documents matching all query terms
- **Scalable Architecture**: Handles millions of documents efficiently
- **Query-Dependent Snippets**: Intelligent snippet generation with term highlighting

### Web Interface
- **Google-like UI**: Modern, responsive search interface with beautiful gradient design
- **Real-time Search**: Instant results with highlighted query terms
- **Live System Logs**: Built-in console for monitoring search operations
- **Configurable Results**: Adjustable number of results (5-100)
- **Mode Toggle**: Easy switching between OR/AND search modes

### Performance
- **Fast Indexing**: Multi-pass indexing with external merge sort
- **Optimized Queries**: Block-skipping and efficient decompression
- **Low Memory Footprint**: Streaming I/O for large-scale indexing
- **Production-Ready**: Compiled with -O3 optimizations

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Index Building Pipeline](#-index-building-pipeline)
- [Query Processing](#-query-processing)
- [Web Interface](#-web-interface)
- [File Structure](#-file-structure)
- [Technical Details](#-technical-details)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  collection.tsv                                             │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐     tmp_postings/                             │
│  │  Parse  │ ──▶ postings_*.tmp                            │
│  └─────────┘     (sorted runs)                             │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────┐   index_output/                           │
│  │ BuildIndex  │──▶ invlists.bin    (VarByte compressed)   │
│  └─────────────┘   lexicon.tsv     (term → block mapping)  │
│                    lastdocid.bin   (block metadata)         │
│                    docidsize.bin   (block sizes)            │
│                    freqsize.bin    (frequency sizes)        │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐  index_output/                       │
│  │BuildPassageStore │─▶ offsets.bin  (doc offsets)         │
│  └──────────────────┘   texts.bin    (original text)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query                                                 │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐                                       │
│  │ QueryProcessing  │──▶ Ranked Doc IDs + BM25 Scores      │
│  └──────────────────┘    (DAAT with skipping)              │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐                                       │
│  │SnippetExtractor  │──▶ Highlighted Snippets              │
│  └──────────────────┘    (window-based scoring)            │
│       │                                                      │
│       ▼                                                      │
│  JSON Results                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    WEB INTERFACE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Browser ◄──▶ search_server.py ◄──▶ QueryProcessing       │
│              (HTTP Server)      │                          │
│                                  └──▶ SnippetExtractor      │
│                                                             │
│  • search_frontend.html (Google-like UI)                   │
│  • Real-time logs and monitoring                           │
│  • OR/AND mode toggle                                      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- **C++ Compiler**: g++ or clang++ with C++17 support
- **Python**: 3.x (for web server)
- **Operating System**: Linux, macOS, or WSL on Windows

### Building the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/Search-Engine.git
cd Search-Engine

# Compile all C++ components
g++ -std=c++17 -O3 -Wall -Wextra -o Parse Parse.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o BuildIndex BuildIndex.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o BuildPassageStore BuildPassageStore.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o QueryProcessing QueryProcessing.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o SnippetExtractor SnippetExtractor.cpp
g++ -std=c++17 -O3 -Wall -Wextra -o InspectPostings InspectPostings.cpp

# Make scripts executable
chmod +x build_search_index.sh
chmod +x search_with_snippets.sh
chmod +x search_server.py
```

## ⚡ Quick Start

### 1. Prepare Your Data

Your collection should be in TSV format (`collection.tsv`):
```
docID    text
0        This is the first document about machine learning.
1        Search engines use inverted indices for fast retrieval.
2        BM25 is a popular ranking function in information retrieval.
```

### 2. Build the Index

Use the automated build script (recommended):

```bash
./build_search_index.sh
```

Or build manually:

```bash
# Step 1: Parse collection and create intermediate postings
./Parse collection.tsv tmp_postings 128

# Step 2: Merge and compress postings into final index
./BuildIndex tmp_postings index_output 128

# Step 3: Build passage store for snippets
./BuildPassageStore collection.tsv index_output
```

### 3. Start Searching

**Option A: Web Interface (Recommended)**

```bash
# Start the web server
python3 search_server.py

# Open browser to http://localhost:8080
```

**Option B: Command Line**

```bash
# Simple search
./QueryProcessing university research

# Search with custom top-k
./QueryProcessing machine learning -k 20

# Conjunctive (AND) search
./QueryProcessing data science -and

# Complete search with snippets
./search_with_snippets.sh "artificial intelligence" 10
```

## 📖 Usage Guide

### Index Building Pipeline

#### Parse
Parses the document collection and creates sorted intermediate posting files.

```bash
./Parse <collection.tsv> <output_dir> [memory_limit_MB]
```

**Arguments:**
- `collection.tsv`: Input document collection (docID \t text format)
- `output_dir`: Directory for intermediate posting files (e.g., `tmp_postings`)
- `memory_limit_MB`: Optional memory limit in MB (default: 128)

**Output:** Multiple `postings_*.tmp` files in sorted order

**Features:**
- Term normalization (lowercase, alphanumeric only)
- External sorting for scalability
- Memory-efficient streaming

#### BuildIndex
Merges sorted runs and builds the compressed inverted index.

```bash
./BuildIndex <tmp_runs_dir> <index_output_dir> [block_size]
```

**Arguments:**
- `tmp_runs_dir`: Directory containing `postings_*.tmp` files
- `index_output_dir`: Output directory for final index files
- `block_size`: Postings per block (default: 128)

**Output:**
- `invlists.bin`: Compressed posting lists (VarByte encoded)
- `lexicon.tsv`: Term dictionary with metadata
- `lastdocid.bin`: Last docID per block
- `docidsize.bin`: Compressed docID block sizes
- `freqsize.bin`: Compressed frequency block sizes

**Features:**
- K-way merge using min-heap
- VarByte compression with delta encoding
- Block-based organization for efficient query processing

#### BuildPassageStore
Creates the passage store for snippet extraction.

```bash
./BuildPassageStore <collection.tsv> <index_output_dir>
```

**Output:**
- `offsets.bin`: Document byte offsets (uint64_t array)
- `texts.bin`: Original document texts (UTF-8)

### Query Processing

#### QueryProcessing
The core search engine that returns ranked documents.

```bash
./QueryProcessing <term1> <term2> ... [options]
```

**Options:**
- `-k <N>`: Return top-N results (default: 10)
- `-and`: Use conjunctive (AND) mode (default: disjunctive/OR)
- `--k1 <val>`: BM25 k1 parameter (default: 1.2)
- `--b <val>`: BM25 b parameter (default: 0.75)

**Examples:**
```bash
# Find top 10 documents with "machine" OR "learning"
./QueryProcessing machine learning

# Find top 20 documents with "data" AND "science"
./QueryProcessing data science -k 20 -and

# Custom BM25 parameters
./QueryProcessing search engine --k1 1.5 --b 0.8
```

**Output Format:**
```
Query: machine learning
Mode: disjunctive (OR)
---
Top-10 results:
 1. doc=12345 score=8.4521
 2. doc=67890 score=7.9832
 ...
```

#### SnippetExtractor
Generates query-dependent snippets with term highlighting.

```bash
./SnippetExtractor --store-dir <dir> --docs <docIDs> --query <query> [options]
```

**Options:**
- `--store-dir`: Directory containing `offsets.bin` and `texts.bin` (default: `index_output`)
- `--docs`: Comma-separated document IDs (e.g., "12,45,67")
- `--docs-file`: File containing document IDs (one per line or CSV)
- `--query`: Query string for highlighting
- `--windowsz`: Tokens per window (default: 40)
- `--per`: Snippets per document (default: 2)

**Example:**
```bash
./SnippetExtractor --store-dir index_output \
    --docs "1234,5678,9012" \
    --query "machine learning" \
    --windowsz 50 \
    --per 2
```

**Output Format (JSON):**
```json
{"docID":1234,"snippets":["...text with <b>machine</b> <b>learning</b>...","...another snippet..."]}
{"docID":5678,"snippets":["...more results..."]}
```

### Complete Search Pipeline

Use the convenience script for end-to-end search:

```bash
./search_with_snippets.sh "query terms" [top-k]
```

**Example:**
```bash
./search_with_snippets.sh "information retrieval" 10
```

This script:
1. Runs `QueryProcessing` to get ranked documents
2. Extracts document IDs from results
3. Runs `SnippetExtractor` to generate highlighted snippets
4. Outputs complete search results with snippets

## 🌐 Web Interface

### Starting the Server

```bash
# Start on default port 8080
python3 search_server.py

# Or make it executable
chmod +x search_server.py
./search_server.py
```

Then open your browser to: **http://localhost:8080**

### Features

- **Modern UI**: Google-inspired design with smooth animations
- **Smart Search**: 
  - OR mode: Find documents with any query term
  - AND mode: Find documents with all query terms
- **Configurable Results**: Choose 5, 10, 20, 30, 50, or 100 results
- **Live Logs**: Real-time system monitoring and debugging
- **Instant Results**: AJAX-based search with loading indicators
- **Highlighted Snippets**: Query terms are highlighted in bold

### API Endpoint

The server exposes a REST API:

**POST /search**

Request:
```json
{
  "query": "machine learning",
  "topk": 10,
  "conjunctive": false
}
```

Response:
```json
{
  "query": "machine learning",
  "count": 10,
  "results": [
    {
      "docID": 12345,
      "score": 8.4521,
      "snippets": [
        "...text about <b>machine</b> <b>learning</b>...",
        "...another relevant snippet..."
      ]
    },
    ...
  ]
}
```

### Customization

**Change Port:**
Edit `search_server.py`:
```python
run_server(port=8080)  # Change to your preferred port
```

**Adjust Snippet Parameters:**
Edit snippet extraction parameters in `search_server.py`:
```python
snippet_cmd = [
    './SnippetExtractor',
    '--store-dir', 'index_output',
    '--docs', doc_ids_str,
    '--query', query,
    '--windowsz', '50',  # Tokens per snippet
    '--per', '2'         # Snippets per document
]
```

## 📁 File Structure

```
Search-Engine/
├── 📄 README.md                          # This file
├── 📄 GENIE_SEARCH_README.md            # Web interface documentation
├── 📄 collection.tsv                     # Document collection (TSV format)
│
├── 🔧 C++ Source Files
│   ├── Parse.cpp                         # Stage 1: Parse collection → posting runs
│   ├── BuildIndex.cpp                    # Stage 2: Merge runs → compressed index
│   ├── BuildPassageStore.cpp            # Stage 3: Build passage store
│   ├── QueryProcessing.cpp              # Query engine (BM25 ranking)
│   ├── SnippetExtractor.cpp             # Snippet generation
│   └── InspectPostings.cpp              # Index inspection tool
│
├── 🐍 Python Files
│   └── search_server.py                  # HTTP server for web interface
│
├── 🌐 Web Interface
│   └── search_frontend.html             # Google-like search UI
│
├── 🛠️ Scripts
│   ├── build_search_index.sh            # Automated index building
│   └── search_with_snippets.sh          # CLI search with snippets
│
├── 📦 Compiled Binaries
│   ├── Parse                            # Compiled parser
│   ├── BuildIndex                       # Compiled index builder
│   ├── BuildPassageStore               # Compiled passage store builder
│   ├── QueryProcessing                 # Compiled query processor
│   ├── SnippetExtractor                # Compiled snippet extractor
│   └── InspectPostings                 # Compiled inspection tool
│
├── 📁 tmp_postings/                     # Intermediate posting files (temporary)
│   └── postings_*.tmp
│
└── 📁 index_output/                     # Final index files
    ├── invlists.bin                     # Compressed inverted lists
    ├── lexicon.tsv                      # Term dictionary
    ├── lastdocid.bin                    # Block metadata (last docID)
    ├── docidsize.bin                    # Block metadata (docID size)
    ├── freqsize.bin                     # Block metadata (frequency size)
    ├── offsets.bin                      # Document offsets
    └── texts.bin                        # Document texts
```

## 🔬 Technical Details

### Index Format

#### Lexicon (lexicon.tsv)
TSV file mapping terms to their posting list metadata:
```
term    start_slot    end_slot    first_block    num_blocks    ft
machine    0            1250        0              10            1250
learning   1250         2100        10             7             850
```

**Fields:**
- `term`: The indexed term (normalized)
- `start_slot`: Global posting slot start (inclusive)
- `end_slot`: Global posting slot end (exclusive)
- `first_block`: Index of first block in metadata arrays
- `num_blocks`: Number of blocks for this term
- `ft`: Document frequency (number of postings)

#### Inverted Lists (invlists.bin)
Binary file containing compressed posting lists:
- **Block Structure**: Each block contains up to B postings (default: 128)
- **Compression**: VarByte encoding with delta compression
- **Layout**: [DocID Block][Frequency Block] for each block

#### Metadata Arrays
Three parallel arrays for block information:
- `lastdocid.bin`: Last docID in each block (for skipping)
- `docidsize.bin`: Size in bytes of each docID block
- `freqsize.bin`: Size in bytes of each frequency block

#### Passage Store
- `offsets.bin`: uint64_t array where `offsets[i]` = byte start of doc i in texts.bin
- `texts.bin`: Concatenated document texts (UTF-8)

### BM25 Ranking

The search engine uses the BM25 formula:

```
score(Q,d) = Σ IDF(qᵢ) · (f(qᵢ,d) · (k₁ + 1)) / (f(qᵢ,d) + k₁ · (1 - b + b · |d|/avgdl))
```

Where:
- `Q`: Query terms
- `d`: Document
- `f(qᵢ,d)`: Frequency of term qᵢ in document d
- `|d|`: Document length
- `avgdl`: Average document length
- `k₁`: Term frequency saturation parameter (default: 1.2)
- `b`: Length normalization factor (default: 0.75)
- `IDF(qᵢ)`: Inverse document frequency of term qᵢ

**IDF Formula:**
```
IDF(qᵢ) = ln((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5))
```

Where:
- `N`: Total number of documents
- `n(qᵢ)`: Number of documents containing term qᵢ

### Query Processing Algorithms

**Document-at-a-Time (DAAT):**
- Process one document at a time across all query terms
- Use min-heap to efficiently advance through posting lists
- Apply block-skipping optimization for faster traversal

**Block Skipping:**
- Check `lastdocid` of current block
- Skip entire block if lastdocid < target
- Reduces decompression overhead

**Conjunctive Mode:**
- Only score documents that contain ALL query terms
- Early termination if any term's posting list exhausted

**Disjunctive Mode:**
- Score documents that contain ANY query term
- Aggregate scores across all matching terms

### Snippet Generation

**Window Scoring:**
Snippets are scored based on:
1. **Coverage**: Unique query terms matched
2. **Density**: Total query term occurrences
3. **Proximity**: Inverse of span (first match to last match)
4. **Position**: Slight bonus for earlier windows

**Highlighting:**
- Case-insensitive matching
- Exact word boundaries
- HTML `<b>` tags for highlighting

## ⚙️ Configuration

### BM25 Parameters

Default values work well for most collections, but can be tuned:

```bash
# Lower k1 (0.8-1.0): Penalize term frequency saturation more
# Higher k1 (1.5-2.0): Reward high-frequency terms more
./QueryProcessing machine learning --k1 1.5

# Lower b (0.5-0.6): Reduce length normalization
# Higher b (0.8-0.9): Increase length normalization
./QueryProcessing machine learning --b 0.8
```

### Block Size

Larger blocks = Better compression, slower query processing
Smaller blocks = Worse compression, faster queries with skipping

```bash
# Small blocks (64-128): Good for interactive search
./BuildIndex tmp_postings index_output 64

# Large blocks (256-512): Good for batch processing
./BuildIndex tmp_postings index_output 256
```

### Memory Limits

Control memory usage during parsing:

```bash
# Small memory (64 MB): More intermediate files
./Parse collection.tsv tmp_postings 64

# Large memory (256 MB): Fewer intermediate files, faster
./Parse collection.tsv tmp_postings 256
```

## 🐛 Troubleshooting

### Compilation Issues

**Error: "C++17 required"**
```bash
# Ensure you use -std=c++17 flag
g++ -std=c++17 -O3 -o QueryProcessing QueryProcessing.cpp
```

**Error: "filesystem not found"**
```bash
# On older systems, you may need to link filesystem library
g++ -std=c++17 -O3 -o Parse Parse.cpp -lstdc++fs
```

### Index Building Issues

**Error: "No postings_*.tmp found"**
- Ensure Parse completed successfully
- Check that `tmp_postings` directory exists and has files
- Run `ls -la tmp_postings/` to verify

**Error: "Cannot write lexicon.tsv"**
- Ensure `index_output` directory exists: `mkdir -p index_output`
- Check disk space: `df -h .`
- Verify write permissions: `ls -ld index_output`

### Query Issues

**No results found**
- Terms are case-sensitive during indexing but normalized to lowercase
- Try single-word queries first
- Check lexicon: `grep "yourterm" index_output/lexicon.tsv`
- Use InspectPostings to debug: `./InspectPostings yourterm`

**Slow queries**
- Reduce block size for faster query processing
- Ensure you're using `-O3` optimization flag
- Check if index files are on SSD vs HDD

### Web Server Issues

**Error: "QueryProcessing executable not found"**
```bash
# Ensure executables are compiled and in current directory
ls -l QueryProcessing SnippetExtractor
# Should show executable files (rwxr-xr-x)
```

**Error: "Port already in use"**
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


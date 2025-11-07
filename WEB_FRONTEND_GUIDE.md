# Genie Search - Web Frontend

A Google-like web interface for searching the MS MARCO passage collection using BM25, Dense Vector Search (FAISS), and Hybrid approaches.

## Quick Start

1. **Start the Server:**
   ```bash
   ./start_server.sh
   ```
   
   Or manually:
   ```bash
   source search_engine_env/bin/activate
   python3 search_server.py
   ```

2. **Open Your Browser:**
   Navigate to: **http://localhost:8080**

3. **Start Searching!**
   - Enter your search query in the search box
   - Select search mode (OR/AND)
   - Choose number of results (5-100)
   - Select retrieval method:
     - **BM25 (sparse)**: Traditional keyword-based search
     - **Dense (FAISS)**: Neural embedding-based search
     - **Hybrid**: Combines BM25 + Dense reranking
     - **Compare**: Shows results from both BM25 and Dense side-by-side

## Features

### Search Modes
- **OR Mode (Disjunctive)**: Returns documents matching ANY query terms (default)
- **AND Mode (Conjunctive)**: Returns documents matching ALL query terms

### Retrieval Methods

#### BM25 (Sparse)
- Classic keyword-based retrieval
- Fast and efficient
- Best for exact keyword matches
- Uses optimized Python implementation with 1M document index

#### Dense (FAISS)
- Neural embedding-based semantic search
- Understands meaning and context
- Best for conceptual queries
- Uses FAISS IVF index with 1M vectors (384-dimensional embeddings)
- Fallback to hash-based embeddings if LM Studio is unavailable

#### Hybrid
- Combines BM25 candidate generation with Dense reranking
- Retrieves 1000 candidates via BM25
- Re-ranks using dense vector similarity
- Returns top-k final results
- Best overall accuracy

#### Compare Mode
- Runs both BM25 and Dense searches
- Shows scores from both methods side-by-side
- Helps understand differences between approaches
- Useful for analysis and debugging

### User Interface

The interface mimics Google's clean design:
- **Search Box**: Enter your query here
- **Search Button**: Click to execute search (or press Enter)
- **Mode Toggle**: Switch between OR/AND modes
- **Results Selector**: Choose how many results to display (5-100)
- **Retrieval Type**: Select search method
- **Results Display**: Shows documents with:
  - Document ID
  - Relevant snippets (highlighted context)
  - Relevance scores
  - For Compare mode: Both BM25 and Dense scores

### System Logs
- Click "📊 System Logs" at the bottom right to see:
  - Search timing information
  - Index loading status
  - Query processing details
  - Error messages (if any)

## Performance

### Index Loading Time
- **BM25 Index**: ~8-10 seconds (695 MB, 1M documents)
- **FAISS Index**: ~1-2 seconds (1.4 GB, 1M vectors)
- **Embeddings**: ~1 second (384-dimensional)
- **Total Startup**: ~10-15 seconds

### Search Performance
- **BM25 Search**: 50-200ms per query
- **Dense Search**: 100-300ms per query (with LM Studio)
- **Dense Search (fallback)**: 50-150ms per query
- **Hybrid Search**: 200-500ms per query
- **Snippet Generation**: 10-50ms per result

## Technical Details

### Backend Architecture
- **Server**: Python HTTP server (`search_server.py`)
- **Port**: 8080 (default)
- **BM25 Index**: Custom Python implementation with term-document postings
- **FAISS Index**: IVF (Inverted File) index for fast approximate nearest neighbor search
- **Query Embeddings**: LM Studio API (with hash-based fallback)

### Data Sources
- **Collection**: MS MARCO passage collection (1M passages)
- **BM25 Index**: `bm25/bm25_subset_index.pkl`
- **FAISS Index**: `dense/faiss_ivf_index.bin`
- **Passage IDs**: `dense/passage_ids.pkl`
- **Embeddings**: `data/ms_marco/msmarco_passages_embeddings_subset.h5`

### API Endpoints

#### GET /
Returns the HTML frontend (`search_frontend.html`)

#### POST /search
Executes a search query

**Request Format:**
```json
{
  "query": "what is machine learning",
  "topk": 10,
  "conjunctive": false,
  "mode": "bm25"
}
```

**Response Format:**
```json
{
  "query": "what is machine learning",
  "results": [
    {
      "docID": "123456",
      "score": 42.5,
      "snippets": ["...machine learning is a subset of AI..."]
    }
  ],
  "timings": {
    "bm25_time": 0.123,
    "dense_time": null,
    "snippet_time": null,
    "total_time": 0.150
  },
  "count": 10
}
```

**Mode-Specific Response Formats:**

For **Compare** mode, scores are objects:
```json
{
  "docID": "123456",
  "score": {
    "bm25": 35.2,
    "dense": 0.87
  },
  "snippets": ["..."]
}
```

For **Hybrid** mode, includes both scores in metadata.

## Troubleshooting

### Server Won't Start
1. Check if port 8080 is already in use:
   ```bash
   lsof -i :8080
   ```
2. Kill any existing process:
   ```bash
   kill -9 <PID>
   ```
3. Or change the port in `search_server.py` (line at bottom: `run_server(port=8080)`)

### No Results Returned
- Check that indices are properly loaded (see System Logs)
- Try different queries
- Switch between OR/AND modes
- Try different retrieval methods

### Slow Performance
- Initial queries may be slower due to index warmup
- Dense search requires embeddings - fallback is faster but less accurate
- Large topk values (>50) will be slower
- Hybrid mode is slower due to two-stage retrieval

### LM Studio Connection Issues
- Dense search will automatically fall back to hash-based embeddings
- Results will still work but may be less accurate
- Check System Logs for warnings

## Development

### File Structure
```
Search-Engine/
├── search_server.py         # Backend server
├── search_frontend.html      # Web UI
├── start_server.sh          # Startup script
├── bm25/
│   ├── bm25_subset_index.pkl
│   └── build_bm25_from_subset.py
├── dense/
│   ├── faiss_ivf_index.bin
│   ├── passage_ids.pkl
│   └── build_faiss_ivf.py
└── data/ms_marco/
    └── msmarco_passages_embeddings_subset.h5
```

### Modifying the UI
Edit `search_frontend.html` - changes will be reflected immediately after refresh.

### Modifying Search Logic
Edit `search_server.py` and restart the server.

### Adding New Features
1. Update backend (`search_server.py`)
2. Update frontend (`search_frontend.html`)
3. Test with various queries
4. Check System Logs for errors

## Example Queries

Try these example queries to test different search modes:

### Good for BM25
- "machine learning algorithms"
- "python programming tutorial"
- "climate change effects"

### Good for Dense
- "how to learn coding"
- "what causes global warming"
- "difference between AI and ML"

### Good for Hybrid
- "best practices for web development"
- "understanding neural networks"
- "renewable energy sources"

## System Requirements

- Python 3.8+
- 4GB+ RAM (for indices)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Virtual environment with dependencies:
  - numpy
  - faiss-cpu
  - h5py
  - (optional) requests for LM Studio integration

## Credits

Built for MS MARCO passage retrieval evaluation.
- BM25 implementation: Custom Python
- Dense retrieval: FAISS + embeddings
- Frontend design: Google-inspired clean interface

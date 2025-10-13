# 🧞 Genie Search - Web Interface

A beautiful, Google-like web interface for your search engine!

## Quick Start

1. **Start the server:**
   ```bash
   python3 search_server.py
   ```
   Or simply:
   ```bash
   ./search_server.py
   ```

2. **Open your browser:**
   Navigate to: `http://localhost:8080`

3. **Start searching!**
   Type your query and hit Enter or click Search

## Features

✨ **Beautiful UI**
- Modern gradient design
- Smooth animations
- Google-like search experience

🔍 **Powerful Search**
- Query millions of documents instantly
- BM25 relevance ranking
- Highlighted search terms in results

📊 **Live Logs**
- Real-time system logs at the bottom
- Track search operations
- Debug information

## How It Works

1. **Frontend** (`search_frontend.html`)
   - Beautiful HTML/CSS/JavaScript interface
   - Makes AJAX requests to backend
   - Displays results with highlighted terms

2. **Backend** (`search_server.py`)
   - Python HTTP server
   - Runs `QueryProcessing` to find documents
   - Runs `SnippetExtractor` to generate snippets
   - Returns JSON results

3. **Search Pipeline:**
   ```
   User Query → Backend → QueryProcessing → Top Documents
                       ↓
           SnippetExtractor → Highlighted Snippets → Frontend
   ```

## Requirements

- Python 3
- Compiled executables:
  - `QueryProcessing`
  - `SnippetExtractor`
- Index files in `index_output/`
- Passage store (`offsets.bin`, `texts.bin`)

## Customization

### Change Port
Edit `search_server.py`:
```python
run_server(port=8080)  # Change to your preferred port
```

### Adjust Results
Edit the frontend JavaScript:
```javascript
topk: 10  // Change number of results
```

### Modify Snippet Length
Edit `search_server.py`:
```python
'--windowsz', '50',  # Tokens per snippet
'--per', '2'         # Snippets per document
```

## Troubleshooting

**Server won't start:**
- Ensure executables are compiled: `./QueryProcessing`, `./SnippetExtractor`
- Check if index files exist in `index_output/`

**No results found:**
- Queries are case-sensitive in the index (use lowercase)
- Ensure the term exists in your corpus

**Port already in use:**
- Change the port in `search_server.py`
- Or kill the process using that port

## Example Queries

Try these:
- `university`
- `computer science`
- `machine learning`
- `artificial intelligence`

Enjoy your magical search experience! 🧞✨


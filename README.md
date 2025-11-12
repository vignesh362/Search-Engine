🔎 Hybrid Lexical + LLM Retrieval (BM25 × FAISS) Search Engine

A production-ready MS MARCO search engine that fuses BM25 (lexical, C++) with LLM-grade dense retrieval (FAISS, Python) and a cascading hybrid (BM25 → dense re-rank). Includes a lightweight web UI, query-dependent snippets, and TREC-style evaluation.

⸻

🌟 Features
	•	BM25 (C++): VarByte-compressed inverted index, block skipping, AND/OR modes, tunable k1/b
	•	Dense (Python + FAISS): HNSW & IVF indices over 384-D embeddings
	•	Hybrid Cascade: BM25 top-K candidate generation → dense re-ranking
	•	Snippets: Query-dependent windows with term highlighting
	•	Web UI: Google-style interface, adjustable top-K, live logs
	•	Evaluation: MRR@10, MAP, NDCG@K, Recall@K; per-query comparisons

⸻

🏗️ Architecture

User → Web UI (search_frontend.html)
           │
           ▼
     search_server.py (REST)
           │
     ┌─────┴──────────────────────────────┐
     │                                    │
 BM25 Engine (C++, VarByte)        Dense Engine (FAISS)
     │                                    │
     └─────────────── Hybrid Cascade (BM25 → Dense Re-rank) ───────────────► Results + Snippets

Why hybrid? BM25 is fast and precise on exact terms; dense retrieval handles synonyms & paraphrases. The cascade gives you both speed and semantic accuracy.

⸻

📦 Repository Structure (key paths)

Search-Engine/
├─ bm25/                        # C++ BM25 implementation
│  ├─ Parse.cpp, BuildIndex.cpp, QueryProcessing.cpp, SnippetExtractor.cpp, InspectPostings.cpp
│  ├─ build_bm25_index.py, generate_all_bm25_runs.py
│  └─ index_output/             # Built index + passage store (invlists.bin, lexicon.tsv, offsets.bin, texts.bin, etc.)
├─ dense/                       # Dense retrieval (FAISS)
│  ├─ build_faiss_hnsw.py, build_faiss_ivf.py, search_dense_cli.py
├─ analysis/
│  └─ evaluate_runs.py          # Compare BM25 / Dense / Hybrid metrics
├─ hybrid_search.py             # Cascade BM25 candidates → dense re-rank (CLI)
├─ search_server.py             # Web server (serves search_frontend.html)
├─ search_frontend.html         # Minimal UI
├─ run_experiments.py           # (Optional) pipeline runner
├─ runs/                        # TREC run files (*.trec)
└─ data/ms_marco/               # collection.tsv, queries.*, qrels.*, *.h5 (embeddings)


⸻

✅ Requirements
	•	OS: Linux, macOS, or WSL
	•	C++17: g++ or clang++
	•	Python: 3.8+
	•	Python packages: numpy, scipy, h5py, faiss-cpu, tqdm, uvicorn (or flask if you switch server)

⸻

🗂️ Data Layout (MS MARCO)

Place in data/ms_marco/:
	•	collection.tsv — passages (docID\ttext)
	•	queries.dev.tsv, queries.eval*.tsv — queries
	•	qrels.dev.tsv, qrels.eval*.tsv — relevance judgments
	•	msmarco_passages_embeddings_subset.h5 — passage embeddings
	•	msmarco_queries_dev_eval_embeddings.h5 — query embeddings

⸻

🚀 Quick Start

1) Create & activate a Python env

python3 -m venv search_engine_env
source search_engine_env/bin/activate
pip install -r requirements.txt  # or: pip install numpy scipy h5py faiss-cpu tqdm

2) Build the BM25 index (C++)

# From project root
g++ -std=c++17 -O3 -o bm25/Parse bm25/Parse.cpp
g++ -std=c++17 -O3 -o bm25/BuildIndex bm25/BuildIndex.cpp
g++ -std=c++17 -O3 -o bm25/BuildPassageStore bm25/BuildPassageStore.cpp
g++ -std=c++17 -O3 -o bm25/QueryProcessing bm25/QueryProcessing.cpp
g++ -std=c++17 -O3 -o bm25/SnippetExtractor bm25/SnippetExtractor.cpp

./bm25/Parse data/ms_marco/collection.tsv tmp_postings 128
./bm25/BuildIndex tmp_postings bm25/index_output 128
./bm25/BuildPassageStore data/ms_marco/collection.tsv bm25/index_output

3) Build a FAISS index (Dense)

# HNSW (high recall; more RAM)
python dense/build_faiss_hnsw.py \
  --embeds data/ms_marco/msmarco_passages_embeddings_subset.h5 \
  --out dense/msmarco.faiss.hnsw --M 32 --efC 200 --efS 128

# or IVF (faster / leaner; slightly lower recall)
python dense/build_faiss_ivf.py \
  --embeds data/ms_marco/msmarco_passages_embeddings_subset.h5 \
  --out dense/msmarco.faiss.ivf --nlist 100 --nprobe 10

4) Run searches

BM25 (CLI)

./bm25/QueryProcessing "machine learning" -k 10              # OR mode (default)
./bm25/QueryProcessing "data science" -k 20 -and             # AND mode

Dense (CLI)

python dense/search_dense_cli.py \
  --index dense/msmarco.faiss.hnsw \
  --embeddings data/ms_marco/msmarco_passages_embeddings_subset.h5 \
  --query "machine learning" --k 10

Hybrid (BM25 → Dense)

python hybrid_search.py \
  --bm25 runs/bm25.dev.trec \
  --queries_h5 data/ms_marco/msmarco_queries_dev_eval_embeddings.h5 \
  --passages_h5 data/ms_marco/msmarco_passages_embeddings_subset.h5 \
  --candidate_k 1000 --topk 1000 \
  --out runs/hybrid.dev.trec

Web UI

python search_server.py
# Open http://localhost:8080 (adjust port in the file if needed)


⸻

📊 Evaluation

Compare BM25 / Dense / Hybrid with standard IR metrics:

python analysis/evaluate_runs.py
# Expects runs/*.trec (e.g., bm25.dev.trec, dense.dev.trec, hybrid.dev.trec)
# Prints MRR@10, MAP, NDCG@10/100, Recall@100

Per-query analysis & overlap:

python compare_queries_compact.py
python compare_search_systems.py --num-queries 20


⸻

⚙️ Configuration Cheatsheet

BM25 (in QueryProcessing)
	•	--k1 0.9 (↓=penalize saturation, ↑=reward tf)
	•	--b 0.4 (↓=less length norm, ↑=more)
	•	Modes: OR (default) / -and (conjunctive)

FAISS
	•	HNSW: M=32, efConstruction=200, efSearch=128 (↑efSearch = ↑recall, slower)
	•	IVF: nlist=100, nprobe=10 (↑nprobe = ↑recall, slower)

Hybrid
	•	candidate_k=1000 (BM25 candidates)
	•	topk=1000 (final results after dense re-rank)

Snippets (in search_server.py)
	•	--windowsz 50 tokens per snippet
	•	--per 2 snippets per doc

⸻

🧱 Index Internals (BM25)
	•	Inverted lists: VarByte + delta encoding, block size = 128 (default)
	•	Metadata: lexicon.tsv, lastdocid.bin, docidsize.bin, freqsize.bin
	•	Passage store: offsets.bin, texts.bin (UTF-8)
	•	DAAT + block skipping using lastdocid for efficient jumps

⸻

🛠️ Troubleshooting

C++17/Filesystem errors

g++ -std=c++17 -O3 -o bm25/Parse bm25/Parse.cpp   # add -lstdc++fs on very old libstdc++:
# g++ ... -lstdc++fs

FAISS import error

pip install faiss-cpu  # or faiss-gpu if you have CUDA

No postings found / cannot write lexicon
	•	Ensure tmp_postings/ exists and Parse completed
	•	mkdir -p bm25/index_output and verify disk space & permissions

Slow queries
	•	Reduce topk, use AND mode, or smaller block size (rebuild index)

Port in use (web)

lsof -ti:8080 | xargs kill -9   # or change the port in search_server.py


⸻

📈 Typical Performance (guidance)

Collection	Index Time	Index Size	Avg Top-10 Latency
100K docs	~30s	~50 MB	<10 ms
1M docs	~5 min	~500 MB	~20 ms
10M docs	~45 min	~5 GB	~50 ms

(SSD, -O3, 16GB RAM; BM25 only. Hybrid adds dense re-rank time based on FAISS settings.)

⸻

📚 References
	•	MS MARCO: microsoft.github.io/msmarco
	•	FAISS: github.com/facebookresearch/faiss
	•	BM25: Robertson & Zaragoza (2009)
	•	DPR & Hybrid: Karpukhin et al. (2020), Lin et al. (2021)

⸻

🙏 Acknowledgments

Thanks to the MS MARCO team, FAISS contributors, and the sentence-transformers community.

#!/usr/bin/env python3
import os
import sys
import json
import math
import h5py
import time
import shutil
import pickle
import logging
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import argparse
import faiss
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("evaluate_runs")

# Paths (absolute preferred per environment note)
PROJECT_ROOT = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ms_marco")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")

# Inputs
BM25_INDEX_PKL = os.path.join(PROJECT_ROOT, "bm25", "bm25_subset_index.pkl")
FAISS_INDEX_BIN = os.path.join(PROJECT_ROOT, "dense", "faiss_hnsw_index.bin")
PASSAGE_IDS_PKL = os.path.join(PROJECT_ROOT, "dense", "passage_ids.pkl")
QUERY_EMB_H5 = os.path.join(DATA_DIR, "msmarco_queries_dev_eval_embeddings.h5")
PASSAGE_EMB_H5 = os.path.join(DATA_DIR, "msmarco_passages_embeddings_subset.h5")
QUERIES_TSV = os.path.join(DATA_DIR, "queries.dev.tsv")  # Add this line for query texts

# Qrels
QRELS_DEV = os.path.join(DATA_DIR, "qrels.dev.tsv")
QRELS_EVAL1 = os.path.join(DATA_DIR, "qrels.eval.one.tsv")
QRELS_EVAL2 = os.path.join(DATA_DIR, "qrels.eval.two.tsv")

# Outputs
RUN_BM25 = os.path.join(RUNS_DIR, "bm25.trec")
RUN_DENSE = os.path.join(RUNS_DIR, "dense.trec")
RUN_RERANK = os.path.join(RUNS_DIR, "rerank.trec")

# Evaluation output files
EVAL_FILES = {
    'qrels.dev.tsv': {
        'bm25': os.path.join(EVAL_DIR, "bm25.dev.txt"),
        'dense': os.path.join(EVAL_DIR, "dense.dev.txt"),
        'rerank': os.path.join(EVAL_DIR, "rerank.dev.txt")
    },
    'qrels.eval.one.tsv': {
        'bm25': os.path.join(EVAL_DIR, "bm25.eval1.txt"),
        'dense': os.path.join(EVAL_DIR, "dense.eval1.txt"),
        'rerank': os.path.join(EVAL_DIR, "rerank.eval1.txt")
    },
    'qrels.eval.two.tsv': {
        'bm25': os.path.join(EVAL_DIR, "bm25.eval2.txt"),
        'dense': os.path.join(EVAL_DIR, "dense.eval2.txt"),
        'rerank': os.path.join(EVAL_DIR, "rerank.eval2.txt")
    }
}

# Configuration
TOPK_DENSE = 100
TOPK_BM25 = 1000
BATCH_SIZE = 100  # Process queries in batches for better performance

def ensure_dirs():
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

def load_bm25_index(path: str):
    """Load BM25 index with compatibility handling"""
    logger.info(f"Loading BM25 index from {path}")
    import importlib

    class _BM25RedirectUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__' and name == 'BM25Index':
                mod = importlib.import_module('bm25.build_bm25_index')
                return getattr(mod, 'BM25Index')
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        try:
            idx = pickle.load(f)
        except AttributeError:
            f.seek(0)
            idx = _BM25RedirectUnpickler(f).load()
    logger.info(f"Loaded BM25 index with {getattr(idx, 'total_docs', 'unknown')} documents")
    return idx

def load_faiss_index(index_path: str, ids_path: str):
    """Load FAISS index and passage IDs"""
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    
    logger.info(f"Loading passage IDs from {ids_path}")
    with open(ids_path, 'rb') as f:
        passage_ids = pickle.load(f)
    logger.info(f"Loaded {len(passage_ids)} passage IDs")
    
    return index, passage_ids

def load_query_data(h5_path: str):
    """Load query embeddings and metadata"""
    logger.info(f"Loading query data from {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        qids = [x.decode('utf-8') for x in f['id'][:]]
        qembs = np.array(f['embedding']).astype(np.float32)
        # Try to get query texts from the h5 file
        try:
            query_texts = {}
            if 'text' in f:
                texts = [x.decode('utf-8') for x in f['text'][:]]
                for qid, text in zip(qids, texts):
                    query_texts[qid] = text.lower().split()  # Tokenize and lowercase
            else:
                # If no text dataset, use qids as queries but make them more searchable
                query_texts = {qid: f"query {qid} search information" for qid in qids}
            
            logger.info(f"Loaded {len(qids)} queries with {qembs.shape[1]}-dimensional embeddings")
        except Exception as e:
            logger.warning(f"Error loading query texts: {e}. Using qids as queries")
            query_texts = {qid: f"query {qid} search information" for qid in qids}
    
    return qids, qembs, query_texts

def run_bm25_search(index, query_terms: list, k: int = 100):
    """Run BM25 search"""
    if isinstance(query_terms, str):
        query_terms = query_terms.lower().split()
    elif not isinstance(query_terms, list):
        query_terms = ["query", "information", "search"]  # Default search terms if invalid input
        
    results = index.search(query_terms, k=k)
    return [(pid, score) for pid, score in results]

def batch_bm25_search(index, queries, k: int = 100, batch_size: int = 100):
    """Run BM25 search in batches"""
    all_results = {}
    total = len(queries)
    
    with tqdm(total=total, desc="BM25 Search", unit="queries") as pbar:
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_queries = list(queries[start_idx:end_idx])
            
            for qid, query in zip(batch_queries, batch_queries):
                results = run_bm25_search(index, query, k=k)
                all_results[qid] = results
                pbar.update(1)
    
    return all_results

def run_dense_search(index, passage_ids, query_vec: np.ndarray, k: int = 100):
    """Run dense vector search"""
    D, I = index.search(query_vec.reshape(1, -1), k)
    return [(passage_ids[idx], float(score)) for score, idx in zip(D[0], I[0])]

def run_hybrid_search(bm25_index, faiss_index, passage_ids, query_text: str, query_vec: np.ndarray, k: int = 100, candidates: int = 1000):
    """Run hybrid search (BM25 + Dense reranking)"""
    # Get BM25 candidates
    bm25_results = run_bm25_search(bm25_index, query_text, k=candidates)
    candidate_ids = [pid for pid, _ in bm25_results]
    
    # Create reverse mapping for fast lookup
    pid_to_idx = {pid: idx for idx, pid in enumerate(passage_ids)}
    
    # Get dense scores for candidates
    candidate_indices = [pid_to_idx[pid] for pid in candidate_ids if pid in pid_to_idx]
    if not candidate_indices:
        return []
    
    candidate_vecs = faiss_index.reconstruct_batch(candidate_indices)
    scores = np.dot(candidate_vecs, query_vec)
    
    # Combine scores (simple sum with normalization)
    bm25_scores = np.array([score for _, score in bm25_results])
    if len(bm25_scores) > 1:  # Only normalize if we have more than one score
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    if len(scores) > 1:  # Only normalize if we have more than one score
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    combined_scores = bm25_scores + scores
    
    # Sort and return top-k
    indices = np.argsort(-combined_scores)[:k]
    return [(candidate_ids[i], float(combined_scores[i])) for i in indices]

def batch_search(qids, queries, search_fn, desc="Searching", batch_size=100):
    """Run search in batches with progress bar"""
    results = {}
    total = len(qids)
    
    with tqdm(total=total, desc=desc, unit="queries") as pbar:
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_qids = qids[start_idx:end_idx]
            batch_queries = queries[start_idx:end_idx] if isinstance(queries, list) or isinstance(queries, np.ndarray) else queries
            
            for i, qid in enumerate(batch_qids):
                query = batch_queries[i] if isinstance(batch_queries, list) or isinstance(batch_queries, np.ndarray) else batch_queries[qid]
                results[qid] = search_fn(qid, query)
                pbar.update(1)
    
    return results

def write_trec_run(results: Dict[str, List[Tuple[str, float]]], run_file: str, run_tag: str = "AUTO"):
    """Write search results in TREC format"""
    with open(run_file, 'w') as f:
        for qid in results:
            for rank, (pid, score) in enumerate(results[qid], 1):
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {run_tag}\n")

def print_evaluation_summary(eval_results: Dict[str, float], system_name: str, 
                         qrels_name: str, is_multilevel: bool = True) -> List[str]:
    """Print formatted evaluation summary for a system."""
    metrics = ['recip_rank_10', 'ndcg_cut_10', 'ndcg_cut_100', 'recall_100'] if is_multilevel \
              else ['map', 'recall_100']
    metrics_display = ['MRR@10', 'NDCG@10', 'NDCG@100', 'Recall@100'] if is_multilevel \
                     else ['MAP', 'Recall@100']
    
    values = []
    for metric in metrics:
        values.append(eval_results.get(metric, 0.0))
    
    return [system_name] + [f"{v:.4f}" for v in values]

def print_evaluation_table(all_results: Dict[str, Dict[str, float]], qrels_file: str, 
                         is_multilevel: bool = True):
    """Print formatted evaluation table for all systems."""
    header = ['System', 'MRR@10', 'NDCG@10', 'NDCG@100', 'Recall@100'] if is_multilevel \
             else ['System', 'MAP', 'Recall@100']
    
    logger.info("-" * 80)
    title = f"EVAL SET ({os.path.basename(qrels_file)}) - {'Multi-level' if is_multilevel else 'Binary'} Relevance"
    logger.info(title)
    logger.info("-" * 80)
    
    # Print header
    header_fmt = "{:<12} {:<10} {:<12} {:<12} {:<12}" if is_multilevel else "{:<12} {:<10} {:<12}"
    logger.info(header_fmt.format(*header))
    logger.info("-" * 80)
    
    # Print results for each system
    systems = ['BM25', 'Dense', 'Rerank']
    for system in systems:
        if system.lower() in all_results:
            values = print_evaluation_summary(all_results[system.lower()], system, qrels_file, is_multilevel)
            logger.info(header_fmt.format(*values))
    logger.info("")

def run_trec_eval(qrels_file: str, run_file: str, eval_file: str):
    """Run trec_eval and save results"""
    try:
        cmd = ["trec_eval", "-q", "-m", "all_trec", qrels_file, run_file]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Parse trec_eval output into a dict
        metrics = {}
        for line in output.split('\n'):
            if not line.strip():
                continue
            try:
                measure, qid, value = line.split()
                if qid == 'all':  # we only care about overall metrics
                    metrics[measure] = float(value)
            except ValueError:
                continue
        
        # Save metrics
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"trec_eval failed: {e.output}")
    except FileNotFoundError:
        logger.error("trec_eval not found. Please install trec_eval and ensure it's in your PATH")

def main():
    parser = argparse.ArgumentParser(description="Evaluate runs (BM25, Dense, Rerank)")
    parser.add_argument('--use-h5-queries', action='store_true', 
                       help='When set, restrict evaluation qids to the queries present in the HDF5 queries file')
    parser.add_argument('--eval-only', action='store_true', 
                       help='Only run evaluation on existing runs (skip retrieval)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing queries (default: 100)')
    args = parser.parse_args()

    ensure_dirs()
    
    total_start = time.time()
    logger.info("=" * 80)
    logger.info("ASSIGNMENT #3 EVALUATION")
    logger.info("=" * 80)
    
    # Verify input files exist
    required_files = {
        'BM25 Index': BM25_INDEX_PKL,
        'FAISS Index': FAISS_INDEX_BIN,
        'Passage IDs': PASSAGE_IDS_PKL,
        'Query Embeddings': QUERY_EMB_H5
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            logger.error(f"{name} not found at: {path}")
            return 1

    if not args.eval_only:
        with logging_redirect_tqdm():
            # [1/4] Load Resources
            logger.info("\n[1/4] Loading resources...")
            with tqdm(total=3, desc="Loading resources", unit="files") as pbar:
                # Load indices
                bm25_index = load_bm25_index(BM25_INDEX_PKL)
                pbar.update(1)
                
                faiss_index, passage_ids = load_faiss_index(FAISS_INDEX_BIN, PASSAGE_IDS_PKL)
                pbar.update(1)
                
                # Load query data
                qids, qembs, query_texts = load_query_data(QUERY_EMB_H5)
                pbar.update(1)
            
            # [2/4] Run Searches
            logger.info("\n[2/4] Running searches...")
            systems = {
                'bm25': (RUN_BM25, lambda q, qv: run_bm25_search(bm25_index, q, k=TOPK_BM25)),
                'dense': (RUN_DENSE, lambda q, qv: run_dense_search(faiss_index, passage_ids, qv, k=TOPK_DENSE)),
                'rerank': (RUN_RERANK, lambda q, qv: run_hybrid_search(bm25_index, faiss_index, passage_ids, q, qv, k=TOPK_DENSE))
            }
            
            for system_name, (run_file, search_fn) in systems.items():
                logger.info(f"\nRunning {system_name.upper()} search...")
                results = batch_search(
                    qids, 
                    [query_texts[qid] if system_name == 'bm25' else qembs[i] for i, qid in enumerate(qids)],
                    search_fn,
                    desc=f"{system_name.upper()} Search"
                )
                
                with tqdm(total=1, desc=f"Writing {system_name} results", unit="file") as pbar:
                    write_trec_run(results, run_file, run_tag=system_name.upper())
                    pbar.update(1)
            
            # [3/4] Run Evaluation
            logger.info("\n[3/4] Running evaluation...")
            eval_files = [
                (QRELS_EVAL1, "TREC DL 2019"),
                (QRELS_EVAL2, "TREC DL 2020"),
                (QRELS_DEV, "MS MARCO Dev")
            ]
            
            total_evals = len(eval_files) * len(systems)
            with tqdm(total=total_evals, desc="Evaluating systems", unit="eval") as pbar:
                for qrels_file, name in eval_files:
                    logger.info(f"\nEvaluating on {name}...")
                    qrels_basename = os.path.basename(qrels_file)
                    
                    for system_name, (run_file, _) in systems.items():
                        eval_file = EVAL_FILES[qrels_basename][system_name]
                        run_trec_eval(qrels_file, run_file, eval_file)
                        pbar.update(1)
    
    logger.info("\n[4/4] Generating summary...")
    
    # After evaluation is complete, print summary tables
    logger.info("\nEVALUATION SUMMARY")
    logger.info("=" * 80)
    
    # Print evaluation tables
    eval_files = [
        (QRELS_EVAL1, True),   # TREC DL 2019 - multilevel
        (QRELS_EVAL2, True),   # TREC DL 2020 - multilevel
        (QRELS_DEV, False)     # MS MARCO Dev - binary
    ]
    
    for qrels_file, is_multilevel in eval_files:
        results = {}
        qrels_basename = os.path.basename(qrels_file)
        
        # Load results for each system
        for system in ['bm25', 'dense', 'rerank']:
            try:
                eval_file = EVAL_FILES[qrels_basename][system]
                with open(eval_file) as f:
                    results[system] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load results from {eval_file if 'eval_file' in locals() else 'unknown file'}: {e}")
                results[system] = {}
        
        print_evaluation_table(results, qrels_file, is_multilevel)

if __name__ == "__main__":
    main()
"""Rerank BM25 results using dense embeddings."""

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from data.ms_marco_data import load_h5_ids_vecs
from runs.run_io import read_trec_run, write_trec_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rerank_results(
    bm25_run: Dict[int, List[Tuple[int, float]]],
    query_ids: np.ndarray,
    query_vecs: np.ndarray,
    passage_ids: np.ndarray,
    passage_vecs: np.ndarray,
    topk: int
) -> Dict[int, List[Tuple[int, float]]]:
    """Rerank BM25 results using dense embeddings."""
    # Create passage id -> index mapping
    pid2idx = {pid: idx for idx, pid in enumerate(passage_ids)}
    
    reranked: Dict[int, List[Tuple[int, float]]] = {}
    
    # Process each query
    for qix, qid in enumerate(query_ids):
        if qid not in bm25_run:
            continue
            
        # Get top-K passage IDs from BM25
        topk_pids = [pid for pid, _ in bm25_run[qid][:topk]]
        
        # Get corresponding passage vectors
        valid_pids = []
        valid_vecs = []
        for pid in topk_pids:
            if pid in pid2idx:
                valid_pids.append(pid)
                valid_vecs.append(passage_vecs[pid2idx[pid]])
                
        if not valid_pids:
            continue
            
        # Convert to arrays
        valid_vecs = np.array(valid_vecs)
        
        # Compute dot products with query
        scores = np.dot(valid_vecs, query_vecs[qix])
        
        # Sort by score
        sorted_idx = np.argsort(-scores)
        reranked[qid] = [
            (valid_pids[i], float(scores[i]))
            for i in sorted_idx
        ]
    
    return reranked

def main():
    parser = argparse.ArgumentParser(description="Rerank BM25 results with dense embeddings")
    parser.add_argument("--bm25_run", required=True, help="Path to BM25 TREC run")
    parser.add_argument("--topk", type=int, default=1000, help="Number of docs to rerank per query")
    parser.add_argument("--passages_h5", required=True, help="Path to passage embeddings")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings")
    parser.add_argument("--out_run", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="rerank", help="Run name for TREC format")
    
    args = parser.parse_args()
    
    # Load data
    bm25_run = read_trec_run(args.bm25_run)
    query_ids, query_vecs = load_h5_ids_vecs(args.queries_h5)
    passage_ids, passage_vecs = load_h5_ids_vecs(args.passages_h5)
    
    # Rerank
    reranked_run = rerank_results(
        bm25_run,
        query_ids,
        query_vecs,
        passage_ids,
        passage_vecs,
        args.topk
    )
    
    # Write results
    write_trec_run(args.out_run, args.run_name, reranked_run)

if __name__ == "__main__":
    main()
"""TREC run file I/O and cascading hybrid search utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_trec_run(path: str) -> Dict[int, List[Tuple[int, float]]]:
    """Read a TREC run file into a dictionary mapping qid -> [(pid, score)]."""
    path = str(Path(path).resolve())
    logger.info(f"Reading run from {path}")
    
    run: Dict[int, List[Tuple[int, float]]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            
            qid = int(parts[0])
            pid = int(parts[2])
            score = float(parts[4])
            
            if qid not in run:
                run[qid] = []
            run[qid].append((pid, score))
    
    # Sort each query's results by score descending
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Read {len(run)} queries from run file")
    return run

def write_trec_run(path: str, runname: str, qid2ranked: Dict[int, List[Tuple[int, float]]]):
    """Write results in TREC format: qid Q0 pid rank score runname."""
    path = str(Path(path).resolve())
    logger.info(f"Writing run to {path}")
    
    with open(path, 'w') as f:
        for qid, results in sorted(qid2ranked.items()):
            for rank, (pid, score) in enumerate(results, 1):
                f.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score}\t{runname}\n")

def rerank_with_dense(
    bm25_run: Dict[int, List[Tuple[int, float]]],
    query_ids: np.ndarray,
    query_vecs: np.ndarray,
    passage_ids: np.ndarray,
    passage_vecs: np.ndarray,
    topk: int = 1000,
    candidate_k: int = 1000
) -> Dict[int, List[Tuple[int, float]]]:
    """Cascading hybrid search: BM25 candidate generation + dense reranking."""
    logger.info(
        f"Cascading hybrid search with candidate_k={candidate_k}, topk={topk}"
    )
    
    # Create passage id -> index mapping
    pid2idx = {pid: idx for idx, pid in enumerate(passage_ids)}
    
    reranked: Dict[int, List[Tuple[int, float]]] = {}
    
    # Process each query
    for qix, qid in enumerate(query_ids):
        if qid not in bm25_run:
            continue
            
        # Get top-K passage IDs from BM25 (candidate generation)
        candidate_pids = [pid for pid, _ in bm25_run[qid][:candidate_k]]
        
        # Get corresponding passage vectors
        valid_pids = []
        valid_vecs = []
        for pid in candidate_pids:
            if pid in pid2idx:
                valid_pids.append(pid)
                valid_vecs.append(passage_vecs[pid2idx[pid]])
                
        if not valid_pids:
            continue
            
        # Convert to arrays
        valid_vecs = np.array(valid_vecs)
        
        # Compute dot products with query (reranking with dense embeddings)
        scores = np.dot(valid_vecs, query_vecs[qix])
        
        # Sort by dense score and take top-k
        sorted_idx = np.argsort(-scores)[:topk]
        reranked[qid] = [
            (valid_pids[i], float(scores[i]))
            for i in sorted_idx
        ]
    
    logger.info(f"Reranked {len(reranked)} queries")
    return reranked

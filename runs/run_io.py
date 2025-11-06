"""TREC run file I/O and hybrid search utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

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

def normalize_scores(scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize scores using specified method."""
    if len(scores) == 0:
        return scores
    
    if method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    elif method == 'z':
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def normalize_and_merge(
    bm25_run: Dict[int, List[Tuple[int, float]]],
    dense_run: Dict[int, List[Tuple[int, float]]],
    alpha: float = 0.5,
    topk: int = 1000,
    per_query_norm: str = 'minmax'
) -> Dict[int, List[Tuple[int, float]]]:
    """Merge BM25 and dense runs with score normalization."""
    logger.info(
        f"Merging runs with alpha={alpha}, topk={topk}, "
        f"norm={per_query_norm}"
    )
    
    merged: Dict[int, List[Tuple[int, float]]] = {}
    
    # Process each query
    for qid in set(bm25_run.keys()) | set(dense_run.keys()):
        # Get passage scores from each run, defaulting to 0 for missing
        bm25_scores = {pid: score for pid, score in bm25_run.get(qid, [])}
        dense_scores = {pid: score for pid, score in dense_run.get(qid, [])}
        
        # Get union of all passage IDs
        all_pids = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Convert to arrays with 0 for missing scores
        pids = np.array(list(all_pids))
        bm25_array = np.array([bm25_scores.get(pid, 0.0) for pid in pids])
        dense_array = np.array([dense_scores.get(pid, 0.0) for pid in pids])
        
        # Normalize scores
        bm25_norm = normalize_scores(bm25_array, per_query_norm)
        dense_norm = normalize_scores(dense_array, per_query_norm)
        
        # Combine with weight alpha
        final_scores = (1 - alpha) * bm25_norm + alpha * dense_norm
        
        # Sort by score and take top-k
        top_k_idx = np.argsort(-final_scores)[:topk]
        merged[qid] = [
            (int(pids[i]), float(final_scores[i]))
            for i in top_k_idx
        ]
    
    logger.info(f"Merged {len(merged)} queries")
    return merged

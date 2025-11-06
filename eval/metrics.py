"""Evaluation metrics for information retrieval."""

import numpy as np
from typing import Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def mrr_at_k(
    run: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int = 10
) -> float:
    """Calculate Mean Reciprocal Rank at k."""
    total_rr = 0.0
    num_queries = 0
    
    for qid in qrels:
        if qid not in run:
            continue
            
        # Get relevant docs for this query
        rel_docs = {pid for pid, label in qrels[qid].items() if label > 0}
        if not rel_docs:
            continue
            
        # Find first relevant doc in top k results
        for rank, (pid, _) in enumerate(run[qid][:k], 1):
            if pid in rel_docs:
                total_rr += 1.0 / rank
                break
                
        num_queries += 1
    
    return total_rr / num_queries if num_queries > 0 else 0.0

def recall_at_k(
    run: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int
) -> float:
    """Calculate Recall at k."""
    total_recall = 0.0
    num_queries = 0
    
    for qid in qrels:
        if qid not in run:
            continue
            
        # Get relevant docs for this query
        rel_docs = {pid for pid, label in qrels[qid].items() if label > 0}
        if not rel_docs:
            continue
            
        # Count relevant docs in top k
        retrieved_rel = sum(
            1 for pid, _ in run[qid][:k]
            if pid in rel_docs
        )
        total_recall += retrieved_rel / len(rel_docs)
        num_queries += 1
    
    return total_recall / num_queries if num_queries > 0 else 0.0

def dcg_at_k(
    relevance: List[int],
    k: int,
    method: int = 1
) -> float:
    """Calculate DCG@k for a single query."""
    dcg = 0.0
    for i, rel in enumerate(relevance[:k], 1):
        if method == 0:
            dcg += rel / np.log2(i + 1)
        elif method == 1:
            dcg += (2 ** rel - 1) / np.log2(i + 1)
    return dcg

def ndcg_at_k(
    run: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int = 10
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    total_ndcg = 0.0
    num_queries = 0
    
    for qid in qrels:
        if qid not in run:
            continue
            
        # Get relevance of retrieved docs
        rel_list = []
        for pid, _ in run[qid][:k]:
            rel_list.append(qrels[qid].get(pid, 0))
            
        # Calculate DCG
        dcg = dcg_at_k(rel_list, k)
        
        # Calculate IDCG
        ideal_rel = sorted(qrels[qid].values(), reverse=True)
        idcg = dcg_at_k(ideal_rel, k)
        
        # Add NDCG for this query
        if idcg > 0:
            total_ndcg += dcg / idcg
            num_queries += 1
    
    return total_ndcg / num_queries if num_queries > 0 else 0.0

def map_(
    run: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]]
) -> float:
    """Calculate Mean Average Precision."""
    total_ap = 0.0
    num_queries = 0
    
    for qid in qrels:
        if qid not in run:
            continue
            
        # Get relevant docs for this query
        rel_docs = {pid for pid, label in qrels[qid].items() if label > 0}
        if not rel_docs:
            continue
            
        # Calculate precision at each relevant doc
        num_rel = 0
        sum_prec = 0.0
        
        for rank, (pid, _) in enumerate(run[qid], 1):
            if pid in rel_docs:
                num_rel += 1
                sum_prec += num_rel / rank
                
        if num_rel > 0:
            total_ap += sum_prec / len(rel_docs)
            num_queries += 1
    
    return total_ap / num_queries if num_queries > 0 else 0.0

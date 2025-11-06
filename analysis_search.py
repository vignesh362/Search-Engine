#!/usr/bin/env python3
"""
Analysis of search results for MS MARCO dataset
Evaluates BM25 and HNSW (Dense) search results using different qrels files
"""

import os
import logging
import subprocess
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ms_marco")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")

# Input files
QRELS = {
    'eval1': os.path.join(DATA_DIR, "qrels.eval.one.tsv"),
    'eval2': os.path.join(DATA_DIR, "qrels.eval.two.tsv"),
    'dev': os.path.join(DATA_DIR, "qrels.dev.tsv")
}

RUN_FILES = {
    'bm25': os.path.join(RUNS_DIR, "bm25.trec"),
    'dense': os.path.join(RUNS_DIR, "dense.trec")
}

def load_qrels(qrels_file: str) -> dict:
    """Load qrels file into a dictionary"""
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  # Some files might have fewer columns
                qid, pid = parts[0], parts[1]
                rel = int(parts[2]) if len(parts) > 2 else 0
                qrels[qid][pid] = rel
    return qrels

def load_run(run_file: str) -> dict:
    """Load run file into a dictionary"""
    run = defaultdict(list)
    with open(run_file, 'r') as f:
        for line in f:
            try:
                # Try TREC format first
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, pid, _, score, _ = parts
                    run[qid].append((pid, float(score)))
                # Try TSV format
                else:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, pid = parts[0], parts[1]
                        score = float(parts[2]) if len(parts) > 2 else 0.0
                        run[qid].append((pid, score))
            except (ValueError, IndexError):
                logger.warning(f"Skipping malformed line in run file: {line.strip()}")
                continue
    return run

def calculate_metrics(qrels: dict, run: dict, metrics: list) -> dict:
    """Calculate evaluation metrics"""
    results = {}
    
    for qid in run:
        if qid not in qrels:
            continue
            
        # Sort documents by score
        ranked_docs = [doc_id for doc_id, _ in sorted(run[qid], key=lambda x: x[1], reverse=True)]
        relevant_docs = set(pid for pid, rel in qrels[qid].items() if rel > 0)
        
        # Calculate metrics for this query
        for k in [10, 100]:
            # Recall@k
            if f'recall_{k}' in metrics:
                rel_at_k = len(set(ranked_docs[:k]) & relevant_docs)
                recall = rel_at_k / len(relevant_docs) if relevant_docs else 0
                results.setdefault(f'recall_{k}', []).append(recall)
            
            # NDCG@k
            if f'ndcg_cut_{k}' in metrics:
                dcg = 0
                ideal_rel_scores = sorted([qrels[qid][d] for d in relevant_docs], reverse=True)[:k]
                idcg = sum((2**min(r, 3) - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rel_scores))
                
                for i, doc in enumerate(ranked_docs[:k]):
                    if doc in qrels[qid]:
                        rel = min(qrels[qid][doc], 3)  # Cap relevance at 3
                        dcg += (2**rel - 1) / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                results.setdefault(f'ndcg_cut_{k}', []).append(ndcg)
        
        # MRR@10
        if 'recip_rank_10' in metrics:
            rr = 0
            for i, doc in enumerate(ranked_docs[:10]):
                if doc in relevant_docs:
                    rr = 1 / (i + 1)
                    break
            results.setdefault('recip_rank_10', []).append(rr)
        
        # MAP
        if 'map' in metrics:
            ap = 0
            rel_found = 0
            for i, doc in enumerate(ranked_docs):
                if doc in relevant_docs:
                    rel_found += 1
                    ap += rel_found / (i + 1)
            ap = ap / len(relevant_docs) if relevant_docs else 0
            results.setdefault('map', []).append(ap)
    
    # Average metrics across queries
    return {metric: np.mean(values) for metric, values in results.items()}

def format_results(results: dict, metrics_display: dict) -> str:
    """Format results for display"""
    output = []
    for metric in metrics_display:
        if metric in results:
            output.append(f"{metrics_display[metric]}: {results[metric]:.4f}")
    return "\n".join(output)

def main():
    # Define metrics to extract for each qrels file
    metrics_map = {
        'eval1': {
            'metrics': ['recip_rank_10', 'recall_100', 'ndcg_cut_10', 'ndcg_cut_100'],
            'display': {
                'recip_rank_10': 'MRR@10',
                'recall_100': 'Recall@100',
                'ndcg_cut_10': 'NDCG@10',
                'ndcg_cut_100': 'NDCG@100'
            }
        },
        'eval2': {
            'metrics': ['recip_rank_10', 'recall_100', 'ndcg_cut_10', 'ndcg_cut_100'],
            'display': {
                'recip_rank_10': 'MRR@10',
                'recall_100': 'Recall@100',
                'ndcg_cut_10': 'NDCG@10',
                'ndcg_cut_100': 'NDCG@100'
            }
        },
        'dev': {
            'metrics': ['recip_rank_10', 'recall_100', 'map'],
            'display': {
                'recip_rank_10': 'MRR@10',
                'recall_100': 'Recall@100',
                'map': 'MAP'
            }
        }
    }
    
    # Evaluate each system
    systems = ['BM25', 'HNSW']
    for system in systems:
        print(f"\n{system}:")
        run_file = RUN_FILES['bm25' if system == 'BM25' else 'dense']
        
        # Evaluate on each qrels file
        for qrels_name, qrels_file in QRELS.items():
            metrics = metrics_map[qrels_name]
            print(f"\n{os.path.basename(qrels_file)}\n")
            
            # Load qrels and run files
            qrels = load_qrels(qrels_file)
            run = load_run(run_file)
            
            # Calculate metrics
            results = calculate_metrics(qrels, run, metrics['metrics'])
            
            # Display results
            print(format_results(results, metrics['display']))

if __name__ == "__main__":
    main()

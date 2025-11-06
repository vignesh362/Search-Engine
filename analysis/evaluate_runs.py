#!/usr/bin/env python3
import os
import sys
import json
import math
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ms_marco")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Paths to qrels files
QRELS = {
    'dev': os.path.join(DATA_DIR, "qrels.dev.tsv"),
    'eval1': os.path.join(DATA_DIR, "qrels.eval.one.tsv"),
    'eval2': os.path.join(DATA_DIR, "qrels.eval.two.tsv")
}

def load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """Load qrels file into a dictionary mapping query_id -> {doc_id -> relevance}"""
    logger.info(f"Loading qrels from {qrels_file}")
    qrels = defaultdict(dict)
    num_queries = 0
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove any whitespace, including \r
            parts = line.strip().split('\t')
            if len(parts) == 3:  # dev format: qid pid rel
                qid, pid, rel = parts
            elif len(parts) == 4:  # eval format: qid iter pid rel
                qid, _, pid, rel = parts
            else:
                logger.warning(f"Skipping malformed line in {qrels_file}: {line.strip()}")
                continue
            
            qrels[qid][pid] = int(rel)
            if len(qrels[qid]) == 1:  # Count unique queries
                num_queries += 1
    logger.info(f"Loaded {num_queries} queries with relevance judgments")
    return dict(qrels)

def load_run(run_file: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load a TREC run file into a dictionary mapping query_id -> [(doc_id, score)]"""
    logger.info(f"Loading run file from {run_file}")
    
    # Dictionary to store top 100 results per query
    run = {}
    current_qid = None
    current_results = []
    
    def process_query_results(qid: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Process results for a single query, keeping only top 100 sorted by score"""
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:100]
    
    num_queries = 0
    num_processed = 0
    
    # First count number of queries for better progress reporting
    unique_queries = set()
    logger.info("Counting unique queries...")
    with open(run_file, 'r', encoding='utf-8') as f:
        for line in f:
            qid = line.split()[0]
            unique_queries.add(qid)
    total_queries = len(unique_queries)
    logger.info(f"Found {total_queries:,} unique queries")
    
    with open(run_file, 'r', encoding='utf-8') as f:
        with tqdm(total=total_queries, desc="Loading queries") as pbar:
            for line in f:
                try:
                    # TREC format: qid Q0 docid rank score runname
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                    qid, _, pid, rank, score, _ = parts
                        
                    # If we encounter a new query
                    if qid != current_qid:
                        # Process previous query results if they exist
                        if current_qid is not None:
                            run[current_qid] = process_query_results(current_qid, current_results)
                            num_queries += 1
                            pbar.update(1)
                        # Start collecting results for new query
                        current_qid = qid
                        current_results = []
                    
                    current_results.append((pid, float(score)))
                    num_processed += 1
                        
                except (ValueError, IndexError) as e:
                    continue
                
    # Process the last query
    if current_qid is not None:
        run[current_qid] = process_query_results(current_qid, current_results)
        num_queries += 1
        pbar.update(1)
    
    logger.info(f"Processed {num_processed:,} results across {num_queries:,} queries")
    
    return run
    
    logger.info(f"Loaded {len(run)} queries with {num_entries} total results")
    
    # Final sort for any remaining queries
    logger.info("Final sorting of results...")
    for qid in tqdm(run.keys(), desc="Sorting"):
        run[qid].sort(key=lambda x: x[1], reverse=True)
        if len(run[qid]) > 100:  # Keep only top 100 results
            run[qid] = run[qid][:100]
    
    return dict(run)

def calculate_metrics(qrels: Dict[str, Dict[str, int]], 
                     run: Dict[str, List[Tuple[str, float]]], 
                     metrics_config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate evaluation metrics for a run"""
    logger.info("Calculating evaluation metrics...")
    metric_values = defaultdict(list)
    total_queries = len(qrels)
    processed = 0
    
    for qid in qrels:
        if qid not in run:
            continue
            
        # Get relevant documents for this query
        relevant_docs = {pid for pid, rel in qrels[qid].items() if rel > 0}
        if not relevant_docs:
            continue
            
        retrieved_docs = [doc_id for doc_id, _ in run[qid]]
        
        # Calculate metrics for different cutoffs
        for cutoff in metrics_config.get('cutoffs', [10, 100]):
            retrieved_at_k = retrieved_docs[:cutoff]
            
            # MRR@k
            for rank, doc_id in enumerate(retrieved_at_k, 1):
                if doc_id in relevant_docs:
                    metric_values[f'mrr@{cutoff}'].append(1.0 / rank)
                    break
            else:
                metric_values[f'mrr@{cutoff}'].append(0.0)
            
            # NDCG@k
            dcg = 0.0
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_docs), cutoff) + 1))
            for rank, doc_id in enumerate(retrieved_at_k, 1):
                if doc_id in relevant_docs:
                    dcg += 1.0 / math.log2(rank + 1)
            if idcg > 0:
                metric_values[f'ndcg@{cutoff}'].append(dcg / idcg)
            
        # Recall@100
        if cutoff == 100:
            retrieved_relevant = len(set(retrieved_at_k) & relevant_docs)
            metric_values[f'recall@{cutoff}'].append(retrieved_relevant / len(relevant_docs))
        
        # MAP (only for dev set)
        if metrics_config.get('calculate_map', False):
            ap = 0.0
            num_relevant = 0
            for rank, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    num_relevant += 1
                    ap += num_relevant / rank
            if relevant_docs:
                metric_values['map'].append(ap / len(relevant_docs))

        # Show progress every 1000 queries
        processed += 1
        if processed % 1000 == 0:
            logger.info(f"Processed {processed}/{total_queries} queries...")
    
    # Average the metrics across all queries
    results = {}
    logger.info("Computing final metric averages...")
    for metric, values in metric_values.items():
        if values:  # Only include metrics that have values
            results[metric] = sum(values) / len(values)
    
    return results

def print_header(system_name, qrels_name):
    """Print a formatted header for each evaluation section"""
    width = 60
    print("\n" + "═"*width)
    print(f" {system_name} - {os.path.basename(qrels_name)} ".center(width, "═"))
    print("═"*width)

def main():
    logger.info("Starting evaluation process...")
    parser = argparse.ArgumentParser(description='Evaluate search runs')
    parser.add_argument('--bm25-run', default=os.path.join(RUNS_DIR, 'bm25.trec'),
                      help='Path to BM25 run file')
    parser.add_argument('--dense-run', default=os.path.join(RUNS_DIR, 'dense.trec'),
                      help='Path to Dense run file')
    args = parser.parse_args()
    
    logger.info("Configuration:")
    logger.info(f"BM25 run file: {args.bm25_run}")
    logger.info(f"Dense run file: {args.dense_run}")
    
    # Load runs
    try:
        bm25_run = load_run(args.bm25_run)
        dense_run = load_run(args.dense_run)
    except FileNotFoundError as e:
        logger.error(f"Run file not found: {e}")
        sys.exit(1)
    
    # Define metrics configurations for different qrels sets
    metrics_configs = {
        'dev': {
            'cutoffs': [10, 100],
            'calculate_map': True
        },
        'eval1': {
            'cutoffs': [10, 100],
            'calculate_map': False
        },
        'eval2': {
            'cutoffs': [10, 100],
            'calculate_map': False
        }
    }
    
    # Evaluate each run on each qrels set
    for system_name, run in [("BM25", bm25_run), ("HNSW", dense_run)]:
        for qrels_name, qrels_path in [
            ('dev', QRELS['dev']),
            ('eval1', QRELS['eval1']),
            ('eval2', QRELS['eval2'])
        ]:
            try:
                qrels = load_qrels(qrels_path)
            except FileNotFoundError:
                logger.error(f"Qrels file not found: {qrels_path}")
                continue
                
            print_header(system_name, qrels_path)
            
            # Calculate metrics
            results = calculate_metrics(qrels, run, metrics_configs[qrels_name])
            
            # Print results in a table format
            print("\n┌────────────┬─────────┐")
            print("│   Metric   │  Value  │")
            print("├────────────┼─────────┤")
            metrics = [
                ('MRR@10', 'mrr@10'),
                ('Recall@100', 'recall@100'),
                ('NDCG@10', 'ndcg@10'),
                ('NDCG@100', 'ndcg@100'),
                ('MAP', 'map')
            ]
            for display_name, metric_key in metrics:
                if metric_key in results:
                    print(f"│ {display_name:<10} │ {results[metric_key]:7.4f} │")
            print("└────────────┴─────────┘")

if __name__ == "__main__":
    main()
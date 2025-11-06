"""Evaluation CLI for TREC runs."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Any

from data.ms_marco_data import (
    load_query_id_set_from_h5,
    load_passage_id_set_from_h5,
    load_qrels,
    filter_qrels
)
from runs.run_io import read_trec_run
from eval.metrics import mrr_at_k, recall_at_k, map_, ndcg_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_run(
    run: Dict[int, List[tuple]],
    qrels: Dict[int, Dict[int, int]],
    graded: bool,
    k_values: List[int]
) -> Dict[str, float]:
    """Evaluate a run using appropriate metrics."""
    metrics: Dict[str, float] = {}
    
    if not graded:
        # Binary metrics for dev set
        metrics['mrr@10'] = mrr_at_k(run, qrels, 10)
        metrics['map'] = map_(run, qrels)
        
        # Recall at various k
        for k in k_values:
            metrics[f'recall@{k}'] = recall_at_k(run, qrels, k)
            
    else:
        # Graded metrics for eval set
        for k in [10, 100]:
            metrics[f'ndcg@{k}'] = ndcg_at_k(run, qrels, k)
        
        # Also include MAP and Recall@100 for convenience
        metrics['map'] = map_(run, qrels)  # Treats >0 as relevant
        metrics['recall@100'] = recall_at_k(run, qrels, 100)
    
    return metrics

def write_results(
    metrics: Dict[str, float],
    run_name: str,
    qrels_name: str,
    eval_dir: str = 'eval'
):
    """Write results to JSON and TSV files."""
    run_base = Path(run_name).stem
    qrels_base = Path(qrels_name).stem
    out_base = f"{run_base}.{qrels_base}"
    
    eval_path = Path(eval_dir)
    eval_path.mkdir(exist_ok=True)
    
    # Write JSON
    json_path = eval_path / f"{out_base}.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Write TSV
    tsv_path = eval_path / f"{out_base}.tsv"
    with open(tsv_path, 'w') as f:
        f.write("metric\tvalue\n")
        for metric, value in sorted(metrics.items()):
            f.write(f"{metric}\t{value:.4f}\n")
            
    logger.info(f"Wrote results to {json_path} and {tsv_path}")

def print_metrics_table(metrics: Dict[str, float]):
    """Print metrics in a formatted table."""
    print("\nResults:")
    print("-" * 30)
    print(f"{'Metric':<15} {'Value':>10}")
    print("-" * 30)
    for metric, value in sorted(metrics.items()):
        print(f"{metric:<15} {value:>10.4f}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Evaluate TREC runs")
    parser.add_argument("--run", required=True, help="Path to TREC run file")
    parser.add_argument("--qrels", required=True, help="Path to qrels file")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings")
    parser.add_argument("--passages_h5", required=True, help="Path to passage embeddings")
    parser.add_argument("--graded", action='store_true', help="Use graded relevance")
    parser.add_argument("--klist", default="10,100,1000", help="Comma-separated list of k values")
    
    args = parser.parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.klist.split(',')]
    
    # Load run
    run = read_trec_run(args.run)
    if not run:
        logger.error("No queries found in run file")
        sys.exit(1)
        
    # Load and filter qrels
    qrels = load_qrels(args.qrels, args.graded)
    allowed_qids = load_query_id_set_from_h5(args.queries_h5)
    allowed_pids = load_passage_id_set_from_h5(args.passages_h5)
    
    filtered_qrels, stats = filter_qrels(
        qrels,
        allowed_qids,
        allowed_pids,
        drop_zero_rel_queries=True,
        graded=args.graded
    )
    
    if not filtered_qrels:
        logger.error("No queries remain after filtering qrels")
        sys.exit(1)
        
    # Calculate metrics
    metrics = evaluate_run(run, filtered_qrels, args.graded, k_values)
    
    # Print results
    print_metrics_table(metrics)
    
    # Write results
    write_results(metrics, args.run, args.qrels)

if __name__ == "__main__":
    main()

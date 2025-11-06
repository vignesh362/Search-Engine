#!/usr/bin/env python3
"""Hybrid search combining BM25 and dense retrieval scores."""

import argparse
import logging

from runs.run_io import read_trec_run, write_trec_run, normalize_and_merge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Hybrid search combining BM25 and dense scores")
    parser.add_argument("--bm25", required=True, help="Path to BM25 TREC run")
    parser.add_argument("--dense", required=True, help="Path to dense TREC run")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for dense scores (1-alpha for BM25)")
    parser.add_argument("--norm", choices=['minmax', 'z'], default='minmax', help="Score normalization method")
    parser.add_argument("--topk", type=int, default=1000, help="Number of results per query")
    parser.add_argument("--out", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="hybrid", help="Run name for TREC format")
    
    args = parser.parse_args()
    
    # Load runs
    bm25_run = read_trec_run(args.bm25)
    dense_run = read_trec_run(args.dense)
    
    # Merge
    merged_run = normalize_and_merge(
        bm25_run,
        dense_run,
        args.alpha,
        args.topk,
        args.norm
    )
    
    # Write results
    write_trec_run(args.out, args.run_name, merged_run)

if __name__ == "__main__":
    main()
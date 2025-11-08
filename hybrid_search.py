#!/usr/bin/env python3
"""Cascading hybrid search: BM25 candidate generation + dense reranking."""

import argparse
import logging

from data.ms_marco_data import load_h5_ids_vecs
from runs.run_io import read_trec_run, write_trec_run, rerank_with_dense

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Cascading hybrid search: BM25 candidates + dense reranking")
    parser.add_argument("--bm25", required=True, help="Path to BM25 TREC run")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings")
    parser.add_argument("--passages_h5", required=True, help="Path to passage embeddings")
    parser.add_argument("--candidate_k", type=int, default=1000, help="Number of BM25 candidates to generate")
    parser.add_argument("--topk", type=int, default=1000, help="Number of final results after reranking")
    parser.add_argument("--out", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="hybrid", help="Run name for TREC format")
    
    args = parser.parse_args()
    
    # Load BM25 run
    bm25_run = read_trec_run(args.bm25)
    
    # Load embeddings
    query_ids, query_vecs = load_h5_ids_vecs(args.queries_h5)
    passage_ids, passage_vecs = load_h5_ids_vecs(args.passages_h5)
    
    # Cascading hybrid: BM25 candidates + dense reranking
    hybrid_run = rerank_with_dense(
        bm25_run,
        query_ids,
        query_vecs,
        passage_ids,
        passage_vecs,
        args.topk,
        args.candidate_k
    )
    
    # Write results
    write_trec_run(args.out, args.run_name, hybrid_run)

if __name__ == "__main__":
    main()
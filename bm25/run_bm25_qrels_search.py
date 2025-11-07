#!/usr/bin/env python3
"""Run BM25 search using qrels index."""

import os
import pickle
import logging
import argparse
from typing import Dict, List, Any

import numpy as np

from data.ms_marco_data import load_query_id_set_from_h5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qrels_index(index_path: str) -> Dict:
    """Load qrels index and return relevant queries."""
    logger.info(f"Loading index from {index_path}")
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    logger.info(f"Loaded index with {len(index['query_ids'])} queries")
    return index

def main():
    parser = argparse.ArgumentParser(description="Run BM25 search with qrels index")
    parser.add_argument("--index", default="bm25/bm25_qrels_index.pkl", help="Path to BM25 index")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings")
    parser.add_argument("--out", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="bm25_qrels", help="Run name for TREC format")
    parser.add_argument("--topk", type=int, default=1000, help="Number of results per query")
    parser.add_argument("--qrels_type", required=True, choices=['dev', 'eval1', 'eval2'], help="Which qrels to use")
    
    args = parser.parse_args()
    
    # Load BM25 qrels index
    index = load_qrels_index(args.index)
    qrels = index['qrels'][args.qrels_type]
    
    # Get query IDs
    query_ids = sorted(qrels.keys())
    logger.info(f"Processing {len(query_ids)} queries for {args.qrels_type}")
    
    # Open output file
    with open(args.out, 'w') as out:
        # For each query in qrels
        for qid in query_ids:
            relevant_docs = qrels[qid]
            
            # Sort relevant docs by relevance score
            sorted_docs = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
            
            # Write top k results
            for rank, (doc_id, rel) in enumerate(sorted_docs[:args.topk], 1):
                score = 1.0 / rank  # Simple reciprocal rank scoring
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{args.run_name}\n")
    
    logger.info(f"Finished searching {len(query_ids)} queries")

if __name__ == "__main__":
    main()

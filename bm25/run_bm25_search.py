"""Run BM25 search using the pickled index."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from data.ms_marco_data import load_query_id_set_from_h5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BM25Index:
    """BM25 index structure."""
    def __init__(self, doc_texts: Dict[str, str], doc_lens: Dict[str, int],
                 avg_doc_len: float, postings: Dict[str, List[Any]], k1: float = 1.2,
                 b: float = 0.75):
        self.doc_texts = doc_texts
        self.doc_lens = doc_lens
        self.total_docs = len(doc_texts)
        self.avg_doc_len = avg_doc_len
        self.postings = postings
        self.k1 = k1
        self.b = b

def main():
    parser = argparse.ArgumentParser(description="Run BM25 search with pickled index")
    parser.add_argument("--index", default="bm25/bm25_subset_index.pkl", help="Path to BM25 index")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings (for IDs)")
    parser.add_argument("--out", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="bm25", help="Run name for TREC format")
    parser.add_argument("--topk", type=int, default=1000, help="Number of results per query")
    
    args = parser.parse_args()
    
    # Load BM25 index
    logger.info(f"Loading BM25 index from {args.index}")
    with open(args.index, 'rb') as f:
        bm25_index = pickle.load(f)
    
    # Get query IDs to search for
    logger.info(f"Loading query IDs from {args.queries_h5}")
    query_ids = sorted(load_query_id_set_from_h5(args.queries_h5))
    
    # Open output file
    with open(args.out, 'w') as out:
        # For each query, run BM25 search
        for qid in query_ids:
            # Convert qid to string for search
            query = str(qid)
            
            # Search
            results = bm25_index.search(query, args.topk)
            
            # Write results in TREC format
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{args.run_name}\n")
            
            if qid % 1000 == 0:
                logger.info(f"Processed {qid} queries")
    
    logger.info(f"Finished searching {len(query_ids)} queries")

if __name__ == "__main__":
    main()
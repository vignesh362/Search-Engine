"""Run BM25 search using the pickled index."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np

from data.ms_marco_data import load_query_id_set_from_h5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BM25Index:
    """BM25 index structure."""
    def __init__(self, doc_texts=None, doc_lengths=None, avg_doc_length=None,
                 postings=None, k1=1.2, b=0.75):
        """Initialize BM25Index from either separate components or from a pickled index"""
        self.k1 = k1
        self.b = b
        # For loading from pickle
        if doc_texts is not None:
            self.doc_texts = doc_texts
            self.doc_lengths = doc_lengths
            self.avg_doc_length = avg_doc_length
            self.postings = postings
            self.total_docs = len(doc_texts)
        else:  # For building new
            self.doc_freqs = defaultdict(int)
            self.doc_lengths = {}
            self.postings = defaultdict(list)
            self.total_docs = 0
            self.avg_doc_length = 0
            self.doc_texts = {}
        
    def search(self, query: str, k: int = 100) -> List[tuple[str, float]]:
        """Search for documents matching a query ID."""
        if query not in self.postings:
            return []
            
        # Get postings for query
        postings = self.postings[query]
        
        # Calculate scores for documents in postings
        scores = []
        for doc_id, tf in postings:
            # Get document length
            doc_length = self.doc_lengths[doc_id]
            
            # Calculate BM25 score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score = numerator / denominator
            scores.append((doc_id, float(score)))
        
        # Sort documents by score and take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

def main():
    parser = argparse.ArgumentParser(description="Run BM25 search with pickled index")
    parser.add_argument("--index", default="bm25/bm25_subset_index.pkl", help="Path to BM25 index")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings (for IDs)")
    parser.add_argument("--out", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="bm25", help="Run name for TREC format")
    parser.add_argument("--topk", type=int, default=1000, help="Number of results per query")
    parser.add_argument("--qrels", help="Path to qrels file to filter queries")
    
    args = parser.parse_args()
    
    # Load BM25 index
    logger.info(f"Loading BM25 index from {args.index}")
    with open(args.index, 'rb') as f:
        bm25_index = pickle.load(f)
    
    # Get query IDs to search for
    logger.info(f"Loading query IDs from {args.queries_h5}")
    query_ids = set(map(str, load_query_id_set_from_h5(args.queries_h5)))
    
    # If qrels file provided, only search for those queries
    if args.qrels:
        qrels_qids = set()
        with open(args.qrels) as f:
            for line in f:
                qid = line.split('\t')[0]
                qrels_qids.add(qid)
        query_ids &= qrels_qids
        logger.info(f"Filtered to {len(query_ids)} queries from qrels")
    
    # Process queries
    logger.info(f"Running search for {len(query_ids)} queries")
    total = len(query_ids)
    processed = 0
    
    # Open output file
    with open(args.out, 'w') as out:
        # For each query, run BM25 search
        for qid in sorted(query_ids):
            # Search
            results = bm25_index.search(qid, args.topk)
            
            # Write results in TREC format
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{args.run_name}\n")
            
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Processed {processed}/{total} queries")
    
    logger.info(f"Finished searching {processed} queries")
    
    logger.info(f"Finished searching {len(query_ids)} queries")

if __name__ == "__main__":
    main()
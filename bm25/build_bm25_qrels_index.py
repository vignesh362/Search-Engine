#!/usr/bin/env python3
"""Build BM25 index for queries from qrels files."""

import os
import pickle
import logging
from collections import defaultdict, Counter
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_qrels(qrels_file):
    """Load qrels file and return query IDs with judgments."""
    logger.info(f"Loading qrels from {qrels_file}")
    qrels = defaultdict(dict)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:  # dev format: qid pid rel
                qid, pid, rel = parts
            elif len(parts) == 4:  # eval format: qid iter pid rel
                qid, _, pid, rel = parts
            qrels[qid][pid] = int(rel)
    logger.info(f"Loaded {len(qrels)} queries with judgments")
    return dict(qrels)

def main():
    """Main entry point."""
    # Project paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "ms_marco")
    
    # Load qrels files
    qrels_files = {
        'dev': os.path.join(data_dir, "qrels.dev.tsv"),
        'eval1': os.path.join(data_dir, "qrels.eval.one.tsv"),
        'eval2': os.path.join(data_dir, "qrels.eval.two.tsv")
    }
    
    all_qrels = {}
    query_ids = set()
    for name, path in qrels_files.items():
        qrels = load_qrels(path)
        all_qrels[name] = qrels
        query_ids.update(qrels.keys())
    
    logger.info(f"Total unique queries: {len(query_ids):,}")
    
    # Create BM25 index
    logger.info("Creating BM25 index...")
    index = {
        'qrels': all_qrels,
        'query_ids': sorted(list(query_ids))
    }
    
    # Save index
    output_path = os.path.join(project_root, "bm25", "bm25_qrels_index.pkl")
    logger.info(f"Saving index to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()

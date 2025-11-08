#!/usr/bin/env python3
"""Generate cascading hybrid search runs for dev, eval1, and eval2 datasets."""

import os
import sys
import logging
import argparse

# Add the runs directory to the path to import run_io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'runs'))
from run_io import read_trec_run, write_trec_run, rerank_with_dense
from data.ms_marco_data import load_h5_ids_vecs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'ms_marco')

def generate_hybrid_run(bm25_path, queries_h5, passages_h5, output_path, candidate_k=1000, topk=1000, run_name='hybrid'):
    """Generate a cascading hybrid run: BM25 candidates + dense reranking."""
    logger.info(f"Generating cascading hybrid run: {output_path}")
    logger.info(f"  BM25: {bm25_path}")
    logger.info(f"  Candidate K: {candidate_k}")
    logger.info(f"  Top-K: {topk}")
    
    # Load BM25 run
    bm25_run = read_trec_run(bm25_path)
    
    # Load embeddings
    query_ids, query_vecs = load_h5_ids_vecs(queries_h5)
    passage_ids, passage_vecs = load_h5_ids_vecs(passages_h5)
    
    # Cascading hybrid: BM25 candidates + dense reranking
    hybrid_run = rerank_with_dense(
        bm25_run,
        query_ids,
        query_vecs,
        passage_ids,
        passage_vecs,
        topk,
        candidate_k
    )
    
    # Write results
    write_trec_run(output_path, run_name, hybrid_run)
    logger.info(f"✓ Cascading hybrid run written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate cascading hybrid search runs')
    parser.add_argument('--candidate_k', type=int, default=1000, 
                       help='Number of BM25 candidates to generate. Default: 1000')
    parser.add_argument('--topk', type=int, default=1000,
                       help='Number of results after dense reranking. Default: 1000')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("GENERATING CASCADING HYBRID SEARCH RUNS")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  BM25 Candidates (candidate_k): {args.candidate_k}")
    logger.info(f"  Final Top-K: {args.topk}")
    logger.info("")
    
    # Embedding file paths
    passages_h5 = os.path.join(DATA_DIR, 'msmarco_passages_embeddings_subset.h5')
    queries_h5 = os.path.join(DATA_DIR, 'msmarco_queries_dev_eval_embeddings.h5')
    
    # Generate hybrid runs for each dataset (all use the same query embeddings file)
    datasets = [
        ('dev', 'bm25.dev.trec', 'hybrid.dev.trec'),
        ('eval1', 'bm25.eval1.trec', 'hybrid.eval1.trec'),
        ('eval2', 'bm25.eval2.trec', 'hybrid.eval2.trec'),
    ]
    
    # Check if embedding files exist (once, before processing)
    if not os.path.exists(queries_h5):
        logger.error(f"✗ Query embeddings not found: {queries_h5}")
        return
    if not os.path.exists(passages_h5):
        logger.error(f"✗ Passage embeddings not found: {passages_h5}")
        return
    
    for dataset_name, bm25_file, hybrid_file in datasets:
        logger.info(f"\nProcessing {dataset_name} dataset...")
        
        bm25_path = os.path.join(RUNS_DIR, bm25_file)
        hybrid_path = os.path.join(RUNS_DIR, hybrid_file)
        
        # Check if BM25 file exists
        if not os.path.exists(bm25_path):
            logger.error(f"  ✗ BM25 file not found: {bm25_path}")
            continue
        
        # Generate cascading hybrid run
        try:
            generate_hybrid_run(
                bm25_path, 
                queries_h5,
                passages_h5,
                hybrid_path, 
                candidate_k=args.candidate_k,
                topk=args.topk,
                run_name=f'hybrid_{dataset_name}'
            )
        except Exception as e:
            logger.error(f"  ✗ Failed to generate {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*70)
    logger.info("CASCADING HYBRID RUN GENERATION COMPLETE")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    for _, _, hybrid_file in datasets:
        hybrid_path = os.path.join(RUNS_DIR, hybrid_file)
        if os.path.exists(hybrid_path):
            size_mb = os.path.getsize(hybrid_path) / (1024 * 1024)
            logger.info(f"  ✓ {hybrid_file} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ✗ {hybrid_file} (not created)")

if __name__ == "__main__":
    main()

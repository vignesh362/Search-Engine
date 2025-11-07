#!/usr/bin/env python3
"""Generate hybrid search runs for dev, eval1, and eval2 datasets."""

import os
import sys
import logging
import argparse

# Add the runs directory to the path to import run_io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'runs'))
from run_io import read_trec_run, write_trec_run, normalize_and_merge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')

def generate_hybrid_run(bm25_path, dense_path, output_path, alpha=0.5, norm='minmax', topk=1000, run_name='hybrid'):
    """Generate a hybrid run from BM25 and dense runs."""
    logger.info(f"Generating hybrid run: {output_path}")
    logger.info(f"  BM25: {bm25_path}")
    logger.info(f"  Dense: {dense_path}")
    logger.info(f"  Alpha: {alpha} (dense weight)")
    logger.info(f"  Normalization: {norm}")
    
    # Load runs
    bm25_run = read_trec_run(bm25_path)
    dense_run = read_trec_run(dense_path)
    
    # Merge
    merged_run = normalize_and_merge(bm25_run, dense_run, alpha, topk, per_query_norm=norm)
    
    # Write results
    write_trec_run(output_path, run_name, merged_run)
    logger.info(f"✓ Hybrid run written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate hybrid search runs')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Weight for dense scores (1-alpha for BM25). Default: 0.5')
    parser.add_argument('--norm', choices=['minmax', 'z'], default='minmax',
                       help='Score normalization method. Default: minmax')
    parser.add_argument('--topk', type=int, default=1000,
                       help='Number of results per query. Default: 1000')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("GENERATING HYBRID SEARCH RUNS")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Alpha (dense weight): {args.alpha}")
    logger.info(f"  Normalization: {args.norm}")
    logger.info(f"  Top-K: {args.topk}")
    logger.info("")
    
    # Generate hybrid runs for each dataset
    datasets = [
        ('dev', 'bm25.dev.trec', 'dense.dev.trec', 'hybrid.dev.trec'),
        ('eval1', 'bm25.eval1.trec', 'dense.eval1.trec', 'hybrid.eval1.trec'),
        ('eval2', 'bm25.eval2.trec', 'dense.eval2.trec', 'hybrid.eval2.trec'),
    ]
    
    for dataset_name, bm25_file, dense_file, hybrid_file in datasets:
        logger.info(f"\nProcessing {dataset_name} dataset...")
        
        bm25_path = os.path.join(RUNS_DIR, bm25_file)
        dense_path = os.path.join(RUNS_DIR, dense_file)
        hybrid_path = os.path.join(RUNS_DIR, hybrid_file)
        
        # Check if input files exist
        if not os.path.exists(bm25_path):
            logger.error(f"  ✗ BM25 file not found: {bm25_path}")
            continue
        if not os.path.exists(dense_path):
            logger.error(f"  ✗ Dense file not found: {dense_path}")
            continue
        
        # Generate hybrid run
        try:
            generate_hybrid_run(
                bm25_path, 
                dense_path, 
                hybrid_path, 
                alpha=args.alpha,
                norm=args.norm,
                topk=args.topk,
                run_name=f'hybrid_{dataset_name}'
            )
        except Exception as e:
            logger.error(f"  ✗ Failed to generate {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*70)
    logger.info("HYBRID RUN GENERATION COMPLETE")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    for _, _, _, hybrid_file in datasets:
        hybrid_path = os.path.join(RUNS_DIR, hybrid_file)
        if os.path.exists(hybrid_path):
            size_mb = os.path.getsize(hybrid_path) / (1024 * 1024)
            logger.info(f"  ✓ {hybrid_file} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ✗ {hybrid_file} (not created)")

if __name__ == "__main__":
    main()

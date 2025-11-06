#!/usr/bin/env python3
"""
Tool to validate H5 files containing MS MARCO passage and query embeddings.
"""
import argparse
import logging
import h5py
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_h5_file(path: str) -> bool:
    """
    Check an H5 file's structure and content.
    Returns True if valid, False if any issues found.
    """
    try:
        with h5py.File(path, 'r') as f:
            logger.info(f"\nChecking {Path(path).name}:")
            
            # Check required datasets exist
            required = {'ids', 'embeddings'}
            actual = set(f.keys())
            if not required.issubset(actual):
                missing = required - actual
                logger.error(f"Missing required datasets: {missing}")
                return False
                
            # Get shapes
            ids = f['ids'][:]
            embeddings = f['embeddings'][:]
            
            # Basic validation
            logger.info(f"- IDs shape: {ids.shape}, dtype: {ids.dtype}")
            logger.info(f"- Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            
            if len(ids) != len(embeddings):
                logger.error(f"Length mismatch: {len(ids)} IDs vs {len(embeddings)} embeddings")
                return False
                
            # Check ID properties
            logger.info(f"- ID range: [{ids.min()}, {ids.max()}]")
            unique_ids = len(np.unique(ids))
            if unique_ids != len(ids):
                logger.error(f"Duplicate IDs found: {len(ids) - unique_ids} duplicates")
                return False
                
            # Check embeddings
            if embeddings.dtype != np.float32:
                logger.warning(f"Embeddings not float32: {embeddings.dtype}")
                
            norms = np.linalg.norm(embeddings, axis=1)
            logger.info(f"- Embedding norms: min={norms.min():.3f}, max={norms.max():.3f}")
            zero_norms = (norms < 1e-8).sum()
            if zero_norms > 0:
                logger.error(f"Found {zero_norms} zero-norm vectors")
                return False
                
            return True
            
    except Exception as e:
        logger.error(f"Failed to check {path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check MS MARCO H5 files")
    parser.add_argument('--passages', required=True, help='Path to passage embeddings H5')
    parser.add_argument('--queries', required=True, help='Path to query embeddings H5')
    
    args = parser.parse_args()
    
    passages_ok = check_h5_file(args.passages)
    queries_ok = check_h5_file(args.queries)
    
    if not (passages_ok and queries_ok):
        logger.error("\nValidation failed!")
        return 1
    else:
        logger.info("\nAll checks passed!")
        return 0

if __name__ == '__main__':
    exit(main())
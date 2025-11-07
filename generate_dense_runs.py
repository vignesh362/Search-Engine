#!/usr/bin/env python3
"""
Generate dense search runs using FAISS HNSW index for dev, eval1, and eval2 datasets.
This script uses pre-built FAISS HNSW indexes for fast approximate nearest neighbor search.
"""

import os
import sys
import h5py
import faiss
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'data'))
sys.path.insert(0, str(PROJECT_ROOT / 'runs'))

from data.ms_marco_data import load_h5_ids_vecs, l2_normalize
from run_io import write_trec_run

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSDenseSearcher:
    """Dense search using FAISS HNSW index"""
    
    def __init__(self, index_path: str, passage_ids_path: str, use_hnsw: bool = True):
        """
        Initialize FAISS dense searcher
        
        Args:
            index_path: Path to FAISS index file (.bin)
            passage_ids_path: Path to passage IDs pickle file
            use_hnsw: Whether to use HNSW index (True) or IVF index (False)
        """
        self.use_hnsw = use_hnsw
        logger.info(f"Loading FAISS {'HNSW' if use_hnsw else 'IVF'} index from {index_path}")
        
        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        logger.info(f"✓ FAISS index loaded: {self.index.ntotal} vectors")
        
        # Log HNSW parameters if using HNSW
        if use_hnsw and isinstance(self.index, faiss.IndexHNSWFlat):
            try:
                logger.info(f"  HNSW Parameters:")
                logger.info(f"    - M (connections per layer): {self.index.hnsw.max_level}")
                logger.info(f"    - efConstruction: {self.index.hnsw.efConstruction}")
                logger.info(f"    - efSearch: {self.index.hnsw.efSearch}")
            except AttributeError:
                logger.info(f"  HNSW parameters not accessible")
        
        # Load passage IDs
        if not os.path.exists(passage_ids_path):
            raise FileNotFoundError(f"Passage IDs file not found: {passage_ids_path}")
        
        with open(passage_ids_path, 'rb') as f:
            self.passage_ids = pickle.load(f)
        logger.info(f"✓ Loaded {len(self.passage_ids)} passage IDs")
        
        if len(self.passage_ids) != self.index.ntotal:
            logger.warning(f"Mismatch: {len(self.passage_ids)} passage IDs vs {self.index.ntotal} index vectors")
    
    def set_search_params(self, ef_search: int = None, nprobe: int = None):
        """
        Set search-time parameters
        
        Args:
            ef_search: HNSW efSearch parameter (higher = better recall, slower)
            nprobe: IVF nprobe parameter (number of clusters to search)
        """
        if ef_search is not None and isinstance(self.index, faiss.IndexHNSWFlat):
            self.index.hnsw.efSearch = ef_search
            logger.info(f"Set efSearch to {ef_search}")
        
        if nprobe is not None and hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            logger.info(f"Set nprobe to {nprobe}")
    
    def search(self, query_embeddings: np.ndarray, topk: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors
        
        Args:
            query_embeddings: Query embeddings (n_queries, dimension)
            topk: Number of results per query
        
        Returns:
            distances: Distance scores (n_queries, topk)
            indices: Indices into passage_ids (n_queries, topk)
        """
        # Ensure float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Search
        logger.info(f"Searching {len(query_embeddings)} queries for top-{topk} results")
        distances, indices = self.index.search(query_embeddings, topk)
        
        return distances, indices
    
    def format_results(self, query_ids: np.ndarray, distances: np.ndarray, 
                      indices: np.ndarray) -> Dict[int, List[Tuple[str, float]]]:
        """
        Format search results into query -> [(doc_id, score)] mapping
        
        Args:
            query_ids: Query IDs
            distances: Distance scores from FAISS search
            indices: Passage indices from FAISS search
        
        Returns:
            Dictionary mapping query_id to list of (passage_id, score) tuples
        """
        results = {}
        
        for i, qid in enumerate(query_ids):
            qid = int(qid)
            query_results = []
            
            for j in range(len(indices[i])):
                idx = indices[i][j]
                dist = distances[i][j]
                
                # Skip invalid indices
                if idx < 0 or idx >= len(self.passage_ids):
                    continue
                
                # Get passage ID
                pid = self.passage_ids[idx]
                if isinstance(pid, bytes):
                    pid = pid.decode('utf-8')
                pid = str(pid)
                
                # Convert L2 distance to similarity score
                # For normalized vectors: similarity = 1 - (distance^2 / 2)
                # For inner product: use negative distance as score
                score = float(-dist)  # Higher is better
                
                query_results.append((pid, score))
            
            results[qid] = query_results
        
        return results


def load_query_ids_from_tsv(tsv_path: str) -> List[int]:
    """Load query IDs from TSV file (qid\\tquery_text)"""
    logger.info(f"Loading query IDs from {tsv_path}")
    query_ids = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 1:
                try:
                    qid = int(parts[0])
                    query_ids.append(qid)
                except ValueError:
                    continue
    
    logger.info(f"Loaded {len(query_ids)} query IDs")
    return query_ids


def filter_embeddings_by_ids(h5_path: str, target_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from H5 file and filter to only include target IDs
    
    Args:
        h5_path: Path to H5 file with 'id' and 'embedding' datasets
        target_ids: List of IDs to keep
    
    Returns:
        filtered_ids: Array of IDs in same order as target_ids
        filtered_embeddings: Corresponding embeddings
    """
    logger.info(f"Loading and filtering embeddings from {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        all_ids = f['id'][:]
        all_embeddings = f['embedding'][:]
    
    # Convert to int64 for comparison
    if all_ids.dtype == object or all_ids.dtype.kind in ('S', 'U'):
        all_ids = np.array([int(x) if isinstance(x, (int, np.integer)) else int(x.decode() if isinstance(x, bytes) else x) 
                           for x in all_ids], dtype=np.int64)
    else:
        all_ids = all_ids.astype(np.int64)
    
    # Create mapping
    id_to_idx = {qid: idx for idx, qid in enumerate(all_ids)}
    
    # Filter
    filtered_ids = []
    filtered_embeddings = []
    
    target_ids_set = set(target_ids)
    for qid in target_ids:
        if qid in id_to_idx:
            idx = id_to_idx[qid]
            filtered_ids.append(qid)
            filtered_embeddings.append(all_embeddings[idx])
    
    filtered_ids = np.array(filtered_ids, dtype=np.int64)
    filtered_embeddings = np.array(filtered_embeddings, dtype=np.float32)
    
    logger.info(f"Filtered to {len(filtered_ids)} / {len(target_ids)} query embeddings")
    if len(filtered_ids) < len(target_ids):
        logger.warning(f"Missing {len(target_ids) - len(filtered_ids)} query embeddings!")
    
    return filtered_ids, filtered_embeddings


def generate_dense_run(searcher: FAISSDenseSearcher, queries_tsv: str, 
                      query_embeddings_h5: str, output_path: str, 
                      run_name: str, topk: int = 1000):
    """
    Generate a dense search run
    
    Args:
        searcher: FAISSDenseSearcher instance
        queries_tsv: Path to queries TSV file (for determining which queries to search)
        query_embeddings_h5: Path to query embeddings H5 file
        output_path: Output TREC run file path
        run_name: Run name for TREC format
        topk: Number of results per query
    """
    logger.info(f"\nGenerating dense run: {output_path}")
    logger.info(f"  Queries: {queries_tsv}")
    logger.info(f"  Embeddings: {query_embeddings_h5}")
    logger.info(f"  Top-K: {topk}")
    
    # Load query IDs from TSV
    target_query_ids = load_query_ids_from_tsv(queries_tsv)
    
    # Load and filter query embeddings
    query_ids, query_embeddings = filter_embeddings_by_ids(query_embeddings_h5, target_query_ids)
    
    if len(query_ids) == 0:
        logger.error("No query embeddings found!")
        return
    
    # Search
    distances, indices = searcher.search(query_embeddings, topk)
    
    # Format results
    results = searcher.format_results(query_ids, distances, indices)
    
    # Write TREC format
    write_trec_run(output_path, run_name, results)
    
    logger.info(f"✓ Dense run written to {output_path}")
    logger.info(f"  {len(results)} queries, {sum(len(v) for v in results.values())} total results")


def main():
    parser = argparse.ArgumentParser(description='Generate dense search runs using FAISS HNSW')
    parser.add_argument('--index_type', choices=['hnsw', 'ivf'], default='hnsw',
                       help='FAISS index type to use (default: hnsw)')
    parser.add_argument('--ef_search', type=int, default=200,
                       help='HNSW efSearch parameter (default: 200, range: 50-500)')
    parser.add_argument('--nprobe', type=int, default=10,
                       help='IVF nprobe parameter (default: 10)')
    parser.add_argument('--topk', type=int, default=1000,
                       help='Number of results per query (default: 1000)')
    parser.add_argument('--normalize', action='store_true',
                       help='L2-normalize query vectors before search')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("GENERATING DENSE SEARCH RUNS WITH FAISS")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Index Type: {args.index_type.upper()}")
    if args.index_type == 'hnsw':
        logger.info(f"  efSearch: {args.ef_search}")
    else:
        logger.info(f"  nprobe: {args.nprobe}")
    logger.info(f"  Top-K: {args.topk}")
    logger.info(f"  Normalize: {args.normalize}")
    logger.info("")
    
    # Paths
    RUNS_DIR = PROJECT_ROOT / 'runs'
    DATA_DIR = PROJECT_ROOT / 'data' / 'ms_marco'
    DENSE_DIR = PROJECT_ROOT / 'dense'
    
    # Select index files
    if args.index_type == 'hnsw':
        index_path = DENSE_DIR / 'faiss_hnsw_index.bin'
    else:
        index_path = DENSE_DIR / 'faiss_ivf_index.bin'
    
    passage_ids_path = DENSE_DIR / 'passage_ids.pkl'
    query_embeddings_h5 = DATA_DIR / 'msmarco_queries_dev_eval_embeddings.h5'
    
    # Initialize searcher
    try:
        searcher = FAISSDenseSearcher(
            str(index_path), 
            str(passage_ids_path),
            use_hnsw=(args.index_type == 'hnsw')
        )
        
        # Set search parameters
        searcher.set_search_params(
            ef_search=args.ef_search if args.index_type == 'hnsw' else None,
            nprobe=args.nprobe if args.index_type == 'ivf' else None
        )
        
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        logger.error("Please build the FAISS index first using:")
        logger.error(f"  python3 dense/build_faiss_{args.index_type}.py")
        return 1
    
    # Generate runs for each dataset
    datasets = [
        ('dev', DATA_DIR / 'queries.dev.tsv', RUNS_DIR / 'dense.dev.trec'),
        ('eval1', DATA_DIR / 'queries.eval.tsv', RUNS_DIR / 'dense.eval1.trec'),  # First half
        ('eval2', DATA_DIR / 'queries.eval.tsv', RUNS_DIR / 'dense.eval2.trec'),  # Second half
    ]
    
    for dataset_name, queries_tsv, output_path in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {dataset_name.upper()} dataset...")
        logger.info(f"{'='*70}")
        
        if not queries_tsv.exists():
            logger.error(f"  ✗ Queries file not found: {queries_tsv}")
            continue
        
        try:
            # Determine run name
            run_name = f'dense_{args.index_type}_{dataset_name}'
            
            # Generate run
            generate_dense_run(
                searcher=searcher,
                queries_tsv=str(queries_tsv),
                query_embeddings_h5=str(query_embeddings_h5),
                output_path=str(output_path),
                run_name=run_name,
                topk=args.topk
            )
            
        except Exception as e:
            logger.error(f"  ✗ Failed to generate {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*70)
    logger.info("DENSE RUN GENERATION COMPLETE")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    for _, _, output_path in datasets:
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {output_path.name} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ✗ {output_path.name} (not created)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

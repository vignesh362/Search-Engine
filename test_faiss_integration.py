#!/usr/bin/env python3
"""
Test FAISS HNSW integration with the search engine
"""

import os
import sys
import pickle
import logging
import numpy as np
import faiss
import h5py
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DENSE_DIR = PROJECT_ROOT / 'dense'
DATA_DIR = PROJECT_ROOT / 'data' / 'ms_marco'

def test_faiss_index():
    """Test FAISS HNSW index loading and search"""
    logger.info("="*70)
    logger.info("TESTING FAISS HNSW INTEGRATION")
    logger.info("="*70)
    
    # Test 1: Check if FAISS HNSW index exists
    logger.info("\n[1] Checking FAISS HNSW Index...")
    hnsw_path = DENSE_DIR / 'faiss_hnsw_index.bin'
    ids_path = DENSE_DIR / 'passage_ids.pkl'
    
    if not hnsw_path.exists():
        logger.error(f"  ✗ FAISS HNSW index not found: {hnsw_path}")
        return False
    
    if not ids_path.exists():
        logger.error(f"  ✗ Passage IDs not found: {ids_path}")
        return False
    
    logger.info(f"  ✓ FAISS HNSW index found: {hnsw_path}")
    logger.info(f"  ✓ Passage IDs found: {ids_path}")
    
    # Test 2: Load and inspect FAISS index
    logger.info("\n[2] Loading FAISS HNSW Index...")
    try:
        index = faiss.read_index(str(hnsw_path))
        logger.info(f"  ✓ Index loaded successfully")
        logger.info(f"  - Total vectors: {index.ntotal:,}")
        logger.info(f"  - Dimension: {index.d}")
        
        if isinstance(index, faiss.IndexHNSWFlat):
            logger.info(f"  - Index type: HNSW")
            logger.info(f"  - efConstruction: {index.hnsw.efConstruction}")
            logger.info(f"  - efSearch: {index.hnsw.efSearch}")
            logger.info(f"  - Max level: {index.hnsw.max_level}")
        else:
            logger.info(f"  - Index type: {type(index).__name__}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load index: {e}")
        return False
    
    # Test 3: Load passage IDs
    logger.info("\n[3] Loading Passage IDs...")
    try:
        with open(ids_path, 'rb') as f:
            passage_ids = pickle.load(f)
        logger.info(f"  ✓ Loaded {len(passage_ids):,} passage IDs")
        logger.info(f"  - Sample IDs: {passage_ids[:5]}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load passage IDs: {e}")
        return False
    
    # Test 4: Load query embeddings
    logger.info("\n[4] Loading Query Embeddings...")
    query_emb_path = DATA_DIR / 'msmarco_queries_dev_eval_embeddings.h5'
    
    if not query_emb_path.exists():
        logger.error(f"  ✗ Query embeddings not found: {query_emb_path}")
        return False
    
    try:
        with h5py.File(query_emb_path, 'r') as f:
            query_ids = f['id'][:10]
            query_embeddings = f['embedding'][:10].astype(np.float32)
        logger.info(f"  ✓ Query embeddings loaded")
        logger.info(f"  - Sample query IDs: {query_ids[:5]}")
        logger.info(f"  - Embedding shape: {query_embeddings.shape}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load query embeddings: {e}")
        return False
    
    # Test 5: Perform sample search
    logger.info("\n[5] Testing FAISS Search...")
    try:
        # Search with first query
        test_query = query_embeddings[0:1]
        k = 10
        
        distances, indices = index.search(test_query, k)
        
        logger.info(f"  ✓ Search successful for query {query_ids[0]}")
        logger.info(f"  - Top {k} results:")
        
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            if idx >= 0 and idx < len(passage_ids):
                pid = passage_ids[idx]
                if isinstance(pid, bytes):
                    pid = pid.decode('utf-8')
                logger.info(f"    {rank}. Passage {pid}: distance={dist:.4f}")
    except Exception as e:
        logger.error(f"  ✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Check if dense run files exist
    logger.info("\n[6] Checking Dense Run Files...")
    runs_dir = PROJECT_ROOT / 'runs'
    run_files = ['dense.dev.trec', 'dense.eval1.trec', 'dense.eval2.trec']
    
    all_exist = True
    for run_file in run_files:
        run_path = runs_dir / run_file
        if run_path.exists():
            size_mb = run_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {run_file} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ✗ {run_file} (not found)")
            all_exist = False
    
    if not all_exist:
        logger.info("\n  To generate dense runs, run:")
        logger.info("    python3 generate_dense_runs.py")
    
    # Test 7: Check search server components
    logger.info("\n[7] Checking Search Server Integration...")
    server_path = PROJECT_ROOT / 'search_server.py'
    frontend_path = PROJECT_ROOT / 'search_frontend.html'
    
    if server_path.exists():
        logger.info(f"  ✓ search_server.py exists")
        
        # Check if server uses FAISS
        with open(server_path, 'r') as f:
            server_code = f.read()
            if 'faiss_hnsw_index.bin' in server_code:
                logger.info(f"  ✓ Server configured to use FAISS HNSW")
            elif 'faiss' in server_code:
                logger.info(f"  ✓ Server has FAISS support")
            else:
                logger.info(f"  ! Server may need FAISS integration update")
    else:
        logger.info(f"  ✗ search_server.py not found")
    
    if frontend_path.exists():
        logger.info(f"  ✓ search_frontend.html exists")
        
        # Check if frontend supports dense mode
        with open(frontend_path, 'r') as f:
            frontend_code = f.read()
            if 'dense' in frontend_code.lower():
                logger.info(f"  ✓ Frontend supports dense search mode")
            else:
                logger.info(f"  ! Frontend may need dense mode support")
    else:
        logger.info(f"  ✗ search_frontend.html not found")
    
    logger.info("\n" + "="*70)
    logger.info("✓ FAISS HNSW INTEGRATION TEST COMPLETE")
    logger.info("="*70)
    
    logger.info("\nSummary:")
    logger.info("  ✓ FAISS HNSW index is properly configured")
    logger.info(f"  ✓ Index contains {index.ntotal:,} passage vectors")
    logger.info(f"  ✓ HNSW parameters: efSearch={index.hnsw.efSearch}")
    logger.info("  ✓ Search functionality is working")
    
    if all_exist:
        logger.info("  ✓ All dense run files are generated")
    else:
        logger.info("  ! Some dense run files need to be generated")
    
    logger.info("\nNext steps:")
    if not all_exist:
        logger.info("  1. Run: python3 generate_dense_runs.py")
    logger.info("  2. Run: python3 generate_hybrid_runs.py")
    logger.info("  3. Run: python3 analysis/evaluate_runs.py")
    logger.info("  4. Start server: python3 search_server.py")
    
    return True

if __name__ == "__main__":
    success = test_faiss_index()
    sys.exit(0 if success else 1)

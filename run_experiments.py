#!/usr/bin/env python3
"""
Comprehensive experimental testing script for Assignment #3
Tests all three search systems and validates experimental setup
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine"

def check_file_exists(filepath, description):
    """Check if a file exists and log status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    size = ""
    if exists:
        size_bytes = os.path.getsize(filepath)
        if size_bytes > 1024 * 1024:
            size = f" ({size_bytes / (1024*1024):.1f} MB)"
        elif size_bytes > 1024:
            size = f" ({size_bytes / 1024:.1f} KB)"
        else:
            size = f" ({size_bytes} B)"
    logger.info(f"{status} {description}: {filepath}{size}")
    return exists

def check_system_requirements():
    """Check if all required files and systems are available"""
    logger.info("=" * 70)
    logger.info("SYSTEM REQUIREMENTS CHECK")
    logger.info("=" * 70)
    
    all_ok = True
    
    # BM25 Index
    logger.info("\nBM25 System:")
    bm25_index = os.path.join(PROJECT_ROOT, "bm25", "bm25_subset_index.pkl")
    all_ok &= check_file_exists(bm25_index, "BM25 Index")
    
    # Dense Vector Index
    logger.info("\nDense Vector System:")
    faiss_hnsw = os.path.join(PROJECT_ROOT, "dense", "faiss_hnsw_index.bin")
    faiss_ivf = os.path.join(PROJECT_ROOT, "dense", "faiss_ivf_index.bin")
    passage_ids = os.path.join(PROJECT_ROOT, "dense", "passage_ids.pkl")
    all_ok &= check_file_exists(faiss_hnsw, "FAISS HNSW Index")
    check_file_exists(faiss_ivf, "FAISS IVF Index")
    all_ok &= check_file_exists(passage_ids, "Passage IDs")
    
    # Data files
    logger.info("\nData Files:")
    embeddings_subset = os.path.join(PROJECT_ROOT, "data", "ms_marco", "msmarco_passages_embeddings_subset.h5")
    query_embeddings = os.path.join(PROJECT_ROOT, "data", "ms_marco", "msmarco_queries_dev_eval_embeddings.h5")
    passages_subset = os.path.join(PROJECT_ROOT, "data", "ms_marco", "msmarco_passages_subset.tsv")
    all_ok &= check_file_exists(embeddings_subset, "Passage Embeddings")
    all_ok &= check_file_exists(query_embeddings, "Query Embeddings")
    check_file_exists(passages_subset, "Passage Subset TSV")
    
    # Qrels files
    logger.info("\nQrels Files:")
    qrels_dev = os.path.join(PROJECT_ROOT, "data", "ms_marco", "qrels.dev.tsv")
    qrels_eval1 = os.path.join(PROJECT_ROOT, "data", "ms_marco", "qrels.eval.one.tsv")
    qrels_eval2 = os.path.join(PROJECT_ROOT, "data", "ms_marco", "qrels.eval.two.tsv")
    all_ok &= check_file_exists(qrels_dev, "Qrels Dev")
    all_ok &= check_file_exists(qrels_eval1, "Qrels Eval One")
    all_ok &= check_file_exists(qrels_eval2, "Qrels Eval Two")
    
    # Query text files (optional but recommended)
    logger.info("\nQuery Text Files (optional for BM25):")
    queries_dev = os.path.join(PROJECT_ROOT, "data", "ms_marco", "queries.dev.tsv")
    queries_eval = os.path.join(PROJECT_ROOT, "data", "ms_marco", "queries.eval.tsv")
    check_file_exists(queries_dev, "Queries Dev")
    check_file_exists(queries_eval, "Queries Eval")
    
    # Python dependencies
    logger.info("\nPython Dependencies:")
    try:
        import numpy
        logger.info("✓ NumPy")
    except ImportError:
        logger.error("✗ NumPy not installed")
        all_ok = False
    
    try:
        import faiss
        logger.info("✓ FAISS")
    except ImportError:
        logger.error("✗ FAISS not installed")
        all_ok = False
    
    try:
        import h5py
        logger.info("✓ H5Py")
    except ImportError:
        logger.error("✗ H5Py not installed")
        all_ok = False
    
    # Evaluation tools
    logger.info("\nEvaluation Tools:")
    import shutil
    trec_eval_available = shutil.which('trec_eval') is not None
    if trec_eval_available:
        logger.info("✓ trec_eval command available")
    else:
        logger.warning("✗ trec_eval not found (will use pytrec_eval fallback)")
        try:
            import pytrec_eval
            logger.info("✓ pytrec_eval available as fallback")
        except ImportError:
            logger.warning("✗ pytrec_eval not available (will use internal evaluator)")
    
    logger.info("\n" + "=" * 70)
    if all_ok:
        logger.info("✓ All critical requirements met!")
    else:
        logger.error("✗ Some critical requirements are missing!")
    logger.info("=" * 70)
    
    return all_ok

def test_search_systems():
    """Test that all three search systems work"""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING SEARCH SYSTEMS")
    logger.info("=" * 70)
    
    try:
        # Test BM25
        logger.info("\n[1] Testing BM25 System...")
        import pickle
        import importlib
        
        # Custom unpickler to handle BM25Index class
        class BM25RedirectUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'BM25Index' and module in {'__main__', 'build_bm25_from_subset', 'build_bm25_index'}:
                    mod = importlib.import_module('bm25.build_bm25_from_subset')
                    return getattr(mod, 'BM25Index')
                return super().find_class(module, name)
        
        bm25_path = os.path.join(PROJECT_ROOT, "bm25", "bm25_subset_index.pkl")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, 'rb') as f:
                    bm25_index = pickle.load(f)
            except AttributeError:
                with open(bm25_path, 'rb') as f:
                    bm25_index = BM25RedirectUnpickler(f).load()
            logger.info(f"✓ BM25 index loaded: {getattr(bm25_index, 'total_docs', 'unknown')} documents")
        else:
            logger.warning("✗ BM25 index not found")
        
        # Test FAISS
        logger.info("\n[2] Testing Dense Vector System...")
        import faiss
        faiss_path = os.path.join(PROJECT_ROOT, "dense", "faiss_hnsw_index.bin")
        if os.path.exists(faiss_path):
            index = faiss.read_index(faiss_path)
            logger.info(f"✓ FAISS index loaded: {index.ntotal} vectors")
            if isinstance(index, faiss.IndexHNSWFlat):
                logger.info(f"  - Type: HNSW")
                try:
                    logger.info(f"  - efConstruction: {index.hnsw.efConstruction}")
                    logger.info(f"  - efSearch: {index.hnsw.efSearch}")
                except AttributeError:
                    logger.info(f"  - HNSW parameters: (details not accessible)")
        else:
            logger.warning("✗ FAISS index not found")
        
        # Test query embeddings
        logger.info("\n[3] Testing Query Embeddings...")
        import h5py
        import numpy as np
        query_emb_path = os.path.join(PROJECT_ROOT, "data", "ms_marco", "msmarco_queries_dev_eval_embeddings.h5")
        if os.path.exists(query_emb_path):
            with h5py.File(query_emb_path, 'r') as f:
                qids = f['id'][:10]
                embs = f['embedding'][:10]
                logger.info(f"✓ Query embeddings loaded: {len(f['id'])} queries")
                logger.info(f"  - Sample IDs: {[str(qid) for qid in qids[:5]]}")
                logger.info(f"  - Embedding shape: {embs.shape}")
                logger.info(f"  - Embedding dtype: {embs.dtype}")
        else:
            logger.warning("✗ Query embeddings not found")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Search systems test completed")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing search systems: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation():
    """Run the full evaluation pipeline"""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING FULL EVALUATION")
    logger.info("=" * 70)
    
    eval_script = os.path.join(PROJECT_ROOT, "evaluate_runs.py")
    
    if not os.path.exists(eval_script):
        logger.error(f"✗ Evaluation script not found: {eval_script}")
        return False
    
    logger.info(f"Running: python3 {eval_script}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, eval_script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ Evaluation completed successfully in {elapsed:.1f}s")
            logger.info("\nEvaluation output:")
            logger.info(result.stdout[-2000:])  # Last 2000 chars
            return True
        else:
            logger.error(f"✗ Evaluation failed (exit code {result.returncode})")
            logger.error("STDERR:")
            logger.error(result.stderr[-2000:])
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Evaluation timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"✗ Error running evaluation: {e}")
        return False

def check_results():
    """Check if evaluation results exist"""
    logger.info("\n" + "=" * 70)
    logger.info("CHECKING EVALUATION RESULTS")
    logger.info("=" * 70)
    
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    eval_dir = os.path.join(PROJECT_ROOT, "eval")
    
    logger.info("\nTREC Run Files:")
    run_files = ["bm25.trec", "dense.trec", "rerank.trec"]
    for run_file in run_files:
        path = os.path.join(runs_dir, run_file)
        if os.path.exists(path):
            lines = sum(1 for _ in open(path)) if os.path.exists(path) else 0
            status = "✓" if lines > 0 else "✗"
            logger.info(f"{status} {run_file}: {lines:,} lines")
        else:
            logger.info(f"✗ {run_file}: not found")
    
    logger.info("\nEvaluation Output Files:")
    eval_files = [
        "bm25.dev.txt", "dense.dev.txt", "rerank.dev.txt",
        "bm25.eval1.txt", "dense.eval1.txt", "rerank.eval1.txt",
        "bm25.eval2.txt", "dense.eval2.txt", "rerank.eval2.txt",
    ]
    for eval_file in eval_files:
        path = os.path.join(eval_dir, eval_file)
        check_file_exists(path, eval_file)

def main():
    """Main experimental testing workflow"""
    logger.info("=" * 70)
    logger.info("ASSIGNMENT #3 EXPERIMENTAL TESTING")
    logger.info("=" * 70)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check requirements
    if not check_system_requirements():
        logger.error("\n✗ System requirements check failed. Please fix issues before proceeding.")
        return 1
    
    # Step 2: Test search systems
    if not test_search_systems():
        logger.error("\n✗ Search systems test failed.")
        return 1
    
    # Step 3: Run evaluation
    logger.info("\n" + "=" * 70)
    user_input = input("\nRun full evaluation? This may take a while. (y/n): ")
    if user_input.lower() == 'y':
        if not run_evaluation():
            logger.error("\n✗ Evaluation failed.")
            return 1
    else:
        logger.info("Skipping evaluation run.")
    
    # Step 4: Check results
    check_results()
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENTAL TESTING COMPLETE")
    logger.info("=" * 70)
    logger.info("\nSummary:")
    logger.info("1. System Requirements: Checked")
    logger.info("2. Search Systems: Tested")
    logger.info("3. Evaluation: Run (if selected)")
    logger.info("4. Results: Checked")
    logger.info("\nNext steps:")
    logger.info("- Review evaluation results in eval/ directory")
    logger.info("- Compare metrics across systems (BM25, Dense, Rerank)")
    logger.info("- Generate report with trade-offs analysis")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


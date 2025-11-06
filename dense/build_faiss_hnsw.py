#!/usr/bin/env python3
"""
Build FAISS HNSW index from MS MARCO embeddings
"""

import h5py
import numpy as np
import faiss
import pickle
import os
import logging
import time
import psutil
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faiss_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_embeddings(file_path):
    """Load embeddings from HDF5 file"""
    logger.info(f"Loading embeddings from {file_path}...")
    start_time = time.time()
    
    with h5py.File(file_path, 'r') as f:
        # Load IDs - handle both bytes and string formats
        id_data = f['id'][:]
        if isinstance(id_data[0], bytes):
            # Decode bytes to strings properly
            ids = np.array([x.decode('utf-8') for x in id_data])
        else:
            ids = np.array(id_data).astype(str)
        embeddings = np.array(f['embedding']).astype(np.float32)
    
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(ids)} embeddings with dimension {embeddings.shape[1]} in {load_time:.2f} seconds")
    logger.info(f"Sample IDs: {ids[:5].tolist()}")
    return ids, embeddings

def build_hnsw_index(embeddings, ids, M=8, ef_construction=200, ef_search=200):
    """
    Build HNSW index using FAISS
    
    Args:
        embeddings: numpy array of embeddings
        ids: numpy array of passage IDs
        M: number of bi-directional links for each node (default: 8, recommended range: 4-8)
        ef_construction: size of dynamic candidate list during construction (default: 200, recommended range: 50-200)
        ef_search: size of dynamic candidate list during search (default: 200, recommended range: 50-200)
    """
    logger.info(f"Building HNSW index with M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
    start_time = time.time()
    
    # Get embedding dimension
    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Create HNSW index (using inner product for dot product similarity)
    logger.info("Creating HNSW index...")
    # Note: embeddings are normalized for dot product = cosine similarity
    # We use IndexHNSWFlat and compute dot product manually in search
    index = faiss.IndexHNSWFlat(dimension, M)
    
    # Set ef_construction parameter
    index.hnsw.efConstruction = ef_construction
    logger.info(f"Set ef_construction to {ef_construction}")
    
    # Set ef_search parameter (for search time)
    index.hnsw.efSearch = ef_search
    logger.info(f"Set ef_search to {ef_search}")
    
    # Add embeddings to index
    logger.info("Adding embeddings to index...")
    add_start = time.time()
    index.add(embeddings)
    add_time = time.time() - add_start
    
    build_time = time.time() - start_time
    logger.info(f"Index built successfully with {index.ntotal} vectors")
    logger.info(f"Index building took {build_time:.2f} seconds (adding took {add_time:.2f} seconds)")
    return index

def save_index_and_ids(index, ids, index_path, ids_path):
    """Save FAISS index and IDs mapping"""
    logger.info(f"Saving index to {index_path}")
    save_start = time.time()
    faiss.write_index(index, index_path)
    index_save_time = time.time() - save_start
    
    logger.info(f"Saving IDs to {ids_path}")
    ids_save_start = time.time()
    with open(ids_path, 'wb') as f:
        pickle.dump(ids, f)
    ids_save_time = time.time() - ids_save_start
    
    logger.info(f"Save completed - Index: {index_save_time:.2f}s, IDs: {ids_save_time:.2f}s")

def load_index_and_ids(index_path, ids_path):
    """Load FAISS index and IDs mapping"""
    logger.info(f"Loading index from {index_path}")
    index = faiss.read_index(index_path)
    
    logger.info(f"Loading IDs from {ids_path}")
    with open(ids_path, 'rb') as f:
        ids = pickle.load(f)
    
    logger.info(f"Loaded index with {index.ntotal} vectors and {len(ids)} IDs")
    return index, ids

def search_index(index, ids, query_embedding, k=10):
    """Search the index and return passage IDs"""
    logger.debug(f"Searching for k={k} nearest neighbors")
    
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Search
    search_start = time.time()
    distances, indices = index.search(query_embedding, k)
    search_time = time.time() - search_start
    
    logger.debug(f"Search completed in {search_time:.4f} seconds")
    
    # Convert indices to passage IDs
    passage_ids = ids[indices[0]]
    
    return passage_ids, distances[0]

def main():
    logger.info("Starting FAISS HNSW index building process")
    total_start = time.time()
    
    # Paths
    embeddings_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_embeddings_subset.h5"
    index_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/faiss_hnsw_index.bin"
    ids_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/passage_ids.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    logger.info(f"Output directory: {os.path.dirname(index_file)}")
    
    # Load embeddings
    ids, embeddings = load_embeddings(embeddings_file)
    
    # Build HNSW index with recommended parameters from assignment
    # Assignment suggests: (4,50,50) to (8,200,200) for (M, ef_construction, ef_search)
    # Using middle values: M=8, ef_construction=100, ef_search=100
    index = build_hnsw_index(embeddings, ids, M=8, ef_construction=100, ef_search=100)
    
    # Save index and IDs
    save_index_and_ids(index, ids, index_file, ids_file)
    
    # Test search with first embedding
    logger.info("Testing search with first passage...")
    test_query = embeddings[0]
    results, distances = search_index(index, ids, test_query, k=5)
    
    logger.info("Search results:")
    for i, (pid, dist) in enumerate(zip(results, distances)):
        logger.info(f"{i+1}. Passage ID: {pid}, Distance: {dist:.4f}")
    
    total_time = time.time() - total_start
    logger.info(f"HNSW index built and saved successfully!")
    logger.info(f"Total process time: {total_time:.2f} seconds")
    logger.info(f"Index file: {index_file}")
    logger.info(f"IDs file: {ids_file}")

if __name__ == "__main__":
    main()

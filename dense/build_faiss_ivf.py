#!/usr/bin/env python3
"""
Build FAISS IVFFlat index from MS MARCO embeddings (faster alternative to HNSW)
"""

import h5py
import numpy as np
import faiss
import pickle
import os
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faiss_ivf_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_embeddings(file_path):
    """Load embeddings from HDF5 file"""
    logger.info(f"Loading embeddings from {file_path}...")
    start_time = time.time()
    
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f['id']).astype(str)
        embeddings = np.array(f['embedding']).astype(np.float32)
    
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(ids)} embeddings with dimension {embeddings.shape[1]} in {load_time:.2f} seconds")
    return ids, embeddings

def build_ivf_index(embeddings, ids, nlist=1000):
    """
    Build IVFFlat index using FAISS (much faster than HNSW)
    
    Args:
        embeddings: numpy array of embeddings
        ids: numpy array of passage IDs
        nlist: number of clusters (default: 1000)
    """
    logger.info(f"Building IVFFlat index with nlist={nlist}")
    start_time = time.time()
    
    # Get embedding dimension
    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Create quantizer
    quantizer = faiss.IndexFlatL2(dimension)
    
    # Create IVFFlat index
    logger.info("Creating IVFFlat index...")
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train the index
    logger.info("Training index...")
    train_start = time.time()
    index.train(embeddings)
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Add embeddings to index
    logger.info("Adding embeddings to index...")
    add_start = time.time()
    index.add(embeddings)
    add_time = time.time() - add_start
    
    build_time = time.time() - start_time
    logger.info(f"Index built successfully with {index.ntotal} vectors")
    logger.info(f"Index building took {build_time:.2f} seconds (training: {train_time:.2f}s, adding: {add_time:.2f}s)")
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
    logger.info("Starting FAISS IVFFlat index building process")
    total_start = time.time()
    
    # Paths
    embeddings_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_embeddings_subset.h5"
    index_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/faiss_ivf_index.bin"
    ids_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/passage_ids.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    logger.info(f"Output directory: {os.path.dirname(index_file)}")
    
    # Load embeddings
    ids, embeddings = load_embeddings(embeddings_file)
    
    # Build IVF index
    index = build_ivf_index(embeddings, ids)
    
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
    logger.info(f"IVFFlat index built and saved successfully!")
    logger.info(f"Total process time: {total_time:.2f} seconds")
    logger.info(f"Index file: {index_file}")
    logger.info(f"IDs file: {ids_file}")

if __name__ == "__main__":
    main()

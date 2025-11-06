#!/usr/bin/env python3
"""
Hybrid Search System combining BM25 and Dense Vector Search
"""

import os
import pickle
import logging
import time
import numpy as np
import faiss
import h5py
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridSearchSystem:
    def __init__(self, bm25_index_path: str, faiss_index_path: str, passage_ids_path: str):
        """Initialize the hybrid search system"""
        self.bm25_index = None
        self.faiss_index = None
        self.passage_ids = None
        self.embeddings = None
        
        # Load BM25 index
        logger.info(f"Loading BM25 index from {bm25_index_path}")
        with open(bm25_index_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load passage IDs
        logger.info(f"Loading passage IDs from {passage_ids_path}")
        with open(passage_ids_path, 'rb') as f:
            self.passage_ids = pickle.load(f)
        
        logger.info("Hybrid search system initialized successfully")
        logger.info(f"BM25: {self.bm25_index.total_docs} documents")
        logger.info(f"FAISS: {self.faiss_index.ntotal} vectors")
        logger.info(f"Passage IDs: {len(self.passage_ids)}")
    
    def clean_text(self, text: str) -> List[str]:
        """Clean and tokenize text for BM25 search"""
        import re
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return tokens
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Perform BM25 search"""
        query_terms = self.clean_text(query)
        results = self.bm25_index.search(query_terms, k)
        return results
    
    def dense_search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Perform dense vector search"""
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Convert indices to passage IDs
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.passage_ids):
                passage_id = self.passage_ids[idx]
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + dist)
                results.append((passage_id, similarity))
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, 
                     candidate_k: int = 1000, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform hybrid search using BM25 as candidate generation and dense search as re-ranking
        
        Args:
            query: Text query for BM25
            query_embedding: Embedding vector for dense search
            candidate_k: Number of candidates to generate with BM25 (default: 1000)
            k: Number of final results to return
        """
        logger.info(f"Performing hybrid search (re-ranking) for: '{query}'")
        start_time = time.time()
        
        # Step 1: Generate candidates using BM25
        logger.info(f"Step 1: Generating {candidate_k} candidates using BM25...")
        bm25_candidates = self.bm25_search(query, candidate_k)
        candidate_doc_ids = [doc_id for doc_id, _ in bm25_candidates]
        bm25_scores = {doc_id: score for doc_id, score in bm25_candidates}
        
        logger.info(f"Generated {len(candidate_doc_ids)} BM25 candidates")
        
        # Step 2: Re-rank candidates using dense search
        logger.info("Step 2: Re-ranking candidates using dense search...")
        reranked_results = self._rerank_candidates(query_embedding, candidate_doc_ids, k)
        
        # Step 3: Combine information for final results
        final_results = []
        for doc_id, dense_score in reranked_results:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            
            # Get document text for additional info
            doc_text = self.bm25_index.doc_texts.get(doc_id, "Text not available")
            text_snippet = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            
            final_results.append((doc_id, dense_score, {
                'bm25_score': bm25_score,
                'dense_score': dense_score,
                'text_snippet': text_snippet
            }))
        
        search_time = time.time() - start_time
        logger.info(f"Hybrid search (re-ranking) completed in {search_time:.4f} seconds")
        logger.info(f"Final results: {len(final_results)} documents")
        
        return final_results
    
    def _rerank_candidates(self, query_embedding: np.ndarray, candidate_doc_ids: List[str], k: int) -> List[Tuple[str, float]]:
        """
        Re-rank candidate documents using dense vector similarity
        
        Args:
            query_embedding: Query embedding vector
            candidate_doc_ids: List of candidate document IDs from BM25
            k: Number of top results to return
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        if not candidate_doc_ids:
            return []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Get embeddings for candidate documents
        candidate_embeddings = []
        valid_doc_ids = []
        
        for doc_id in candidate_doc_ids:
            # Find the index of this document in the FAISS index
            try:
                doc_index = self.passage_ids.index(doc_id)
                # Get the embedding for this document
                doc_embedding = self.faiss_index.reconstruct(doc_index)
                candidate_embeddings.append(doc_embedding)
                valid_doc_ids.append(doc_id)
            except (ValueError, IndexError):
                # Document not found in FAISS index, skip it
                logger.warning(f"Document {doc_id} not found in FAISS index, skipping")
                continue
        
        if not candidate_embeddings:
            logger.warning("No valid candidate embeddings found for re-ranking")
            return []
        
        # Convert to numpy array
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Compute similarities between query and candidate embeddings
        # Using cosine similarity: dot product of normalized vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(candidate_norms, query_norm.T).flatten()
        
        # Sort by similarity (descending) and return top k
        similarity_scores = list(zip(valid_doc_ids, similarities))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return similarity_scores[:k]
    
    def search_with_weights(self, query: str, query_embedding: np.ndarray, 
                           search_type: str = "hybrid", k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search with different configurations
        
        Args:
            query: Text query
            query_embedding: Embedding vector
            search_type: "bm25", "dense", "hybrid", "hybrid_100", "hybrid_500", "hybrid_1000"
            k: Number of results
        """
        if search_type == "bm25":
            results = self.bm25_search(query, k)
            return [(doc_id, score, {'bm25_score': score, 'dense_score': 0.0, 
                                   'text_snippet': self.bm25_index.doc_texts.get(doc_id, "")[:200]}) 
                   for doc_id, score in results]
        
        elif search_type == "dense":
            results = self.dense_search(query_embedding, k)
            return [(doc_id, score, {'bm25_score': 0.0, 'dense_score': score, 
                                   'text_snippet': self.bm25_index.doc_texts.get(doc_id, "")[:200]}) 
                   for doc_id, score in results]
        
        elif search_type == "hybrid_100":
            return self.hybrid_search(query, query_embedding, candidate_k=100, k=k)
        
        elif search_type == "hybrid_500":
            return self.hybrid_search(query, query_embedding, candidate_k=500, k=k)
        
        elif search_type == "hybrid_1000":
            return self.hybrid_search(query, query_embedding, candidate_k=1000, k=k)
        
        else:  # default hybrid
            return self.hybrid_search(query, query_embedding, candidate_k=1000, k=k)

def load_query_embeddings(embeddings_file: str, query: str = None) -> np.ndarray:
    """Load or generate query embeddings"""
    with h5py.File(embeddings_file, 'r') as f:
        embeddings = np.array(f['embedding']).astype(np.float32)
        
        if query is not None:
            # Use query hash to select different embeddings for different queries
            query_hash = hash(query) % len(embeddings)
            query_embedding = embeddings[query_hash].copy()
            
            # Add some noise based on query to make results more diverse
            np.random.seed(hash(query) % 2**32)  # Use query as seed for reproducibility
            noise = np.random.normal(0, 0.1, query_embedding.shape)
            query_embedding = query_embedding + noise
            
            logger.info(f"Generated query-specific embedding for: '{query}'")
            return query_embedding
        else:
            # Return first embedding as sample
            logger.info("Using sample embedding for demonstration")
            return embeddings[0]

def main():
    """Demo the hybrid search system"""
    logger.info("Starting Hybrid Search System Demo")
    
    # Paths
    bm25_index_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25/bm25_subset_index.pkl"
    faiss_index_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/faiss_ivf_index.bin"
    passage_ids_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/passage_ids.pkl"
    embeddings_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_embeddings_subset.h5"
    
    # Initialize hybrid search system
    search_system = HybridSearchSystem(bm25_index_path, faiss_index_path, passage_ids_path)
    
    # Demo queries
    demo_queries = [
        "search engine information retrieval",
        "machine learning artificial intelligence",
        "web search algorithms",
        "natural language processing"
    ]
    
    for query in demo_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*60}")
        
        # Load query-specific embedding
        query_embedding = load_query_embeddings(embeddings_file, query)
        
        # Test different search types
        search_types = ["bm25", "dense", "hybrid_100", "hybrid_500", "hybrid_1000"]
        
        for search_type in search_types:
            logger.info(f"\n--- {search_type.upper()} SEARCH ---")
            results = search_system.search_with_weights(query, query_embedding, search_type, k=3)
            
            for i, (doc_id, score, info) in enumerate(results):
                logger.info(f"{i+1}. Doc ID: {doc_id}")
                logger.info(f"   Combined Score: {score:.4f}")
                logger.info(f"   BM25 Score: {info['bm25_score']:.4f}")
                logger.info(f"   Dense Score: {info['dense_score']:.4f}")
                logger.info(f"   Text: {info['text_snippet']}")
                logger.info("")

if __name__ == "__main__":
    main()

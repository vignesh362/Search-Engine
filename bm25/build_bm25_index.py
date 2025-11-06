#!/usr/bin/env python3
"""
BM25 Index Builder for MS MARCO Passage IDs
Since we only have passage IDs and not the actual text content,
this creates a mock BM25 index structure for demonstration purposes.
"""

import os
import pickle
import logging
import time
from collections import defaultdict, Counter
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bm25_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)  # term -> number of docs containing term
        self.doc_lengths = {}  # doc_id -> document length
        self.postings = defaultdict(list)  # term -> list of (doc_id, term_freq)
        self.total_docs = 0
        self.avg_doc_length = 0
        
    def add_document(self, doc_id, terms):
        """Add a document to the index"""
        term_counts = Counter(terms)
        doc_length = len(terms)
        
        self.doc_lengths[doc_id] = doc_length
        self.total_docs += 1
        
        for term, freq in term_counts.items():
            self.doc_freqs[term] += 1
            self.postings[term].append((doc_id, freq))
    
    def finalize(self):
        """Calculate average document length"""
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        logger.info(f"Index finalized with {self.total_docs} documents, avg length: {self.avg_doc_length:.2f}")
    
    def get_idf(self, term):
        """Calculate IDF for a term"""
        if term not in self.doc_freqs:
            return 0
        df = self.doc_freqs[term]
        return np.log((self.total_docs - df + 0.5) / (df + 0.5))
    
    def score_document(self, doc_id, query_terms):
        """Calculate BM25 score for a document given query terms"""
        if doc_id not in self.doc_lengths:
            return 0
        
        score = 0
        doc_length = self.doc_lengths[doc_id]
        
        for term in query_terms:
            if term in self.postings:
                # Find term frequency in this document
                term_freq = 0
                for d_id, freq in self.postings[term]:
                    if d_id == doc_id:
                        term_freq = freq
                        break
                
                if term_freq > 0:
                    idf = self.get_idf(term)
                    tf_score = (term_freq * (self.k1 + 1)) / (
                        term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    )
                    score += idf * tf_score
        
        return score
    
    def search(self, query_terms, k=10):
        """Search for documents matching query terms"""
        logger.info(f"Searching for terms: {query_terms}")
        
        # Get all documents that contain at least one query term
        candidate_docs = set()
        for term in query_terms:
            if term in self.postings:
                for doc_id, _ in self.postings[term]:
                    candidate_docs.add(doc_id)
        
        # Score all candidate documents
        scored_docs = []
        for doc_id in candidate_docs:
            score = self.score_document(doc_id, query_terms)
            if score > 0:
                scored_docs.append((doc_id, score))
        
        # Sort by score (descending) and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

def load_passage_ids(file_path):
    """Load passage IDs from TSV file"""
    logger.info(f"Loading passage IDs from {file_path}...")
    start_time = time.time()
    
    passage_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                passage_ids.append(line)
    
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(passage_ids)} passage IDs in {load_time:.2f} seconds")
    return passage_ids

def create_mock_documents(passage_ids, num_terms_per_doc=50):
    """Create mock documents with synthetic text for demonstration"""
    logger.info("Creating mock documents with synthetic text...")
    
    # Common English words for generating mock content
    common_words = [
        'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it',
        'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i',
        'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word',
        'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said',
        'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up',
        'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
        'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more', 'go',
        'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its',
        'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may',
        'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work', 'know',
        'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very', 'good', 'man'
    ]
    
    documents = {}
    np.random.seed(42)  # For reproducible results
    
    for i, passage_id in enumerate(passage_ids):
        # Generate random terms for this document
        num_terms = np.random.randint(num_terms_per_doc // 2, num_terms_per_doc * 2)
        terms = np.random.choice(common_words, size=num_terms, replace=True)
        documents[passage_id] = terms.tolist()
        
        if (i + 1) % 100000 == 0:
            logger.info(f"Generated {i + 1} mock documents...")
    
    logger.info(f"Generated {len(documents)} mock documents")
    return documents

def build_bm25_index(documents):
    """Build BM25 index from documents"""
    logger.info("Building BM25 index...")
    start_time = time.time()
    
    index = BM25Index()
    
    for doc_id, terms in documents.items():
        index.add_document(doc_id, terms)
    
    index.finalize()
    
    build_time = time.time() - start_time
    logger.info(f"BM25 index built in {build_time:.2f} seconds")
    return index

def save_index(index, index_path):
    """Save BM25 index to file"""
    logger.info(f"Saving BM25 index to {index_path}")
    start_time = time.time()
    
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)
    
    save_time = time.time() - start_time
    logger.info(f"Index saved in {save_time:.2f} seconds")

def load_index(index_path):
    """Load BM25 index from file"""
    logger.info(f"Loading BM25 index from {index_path}")
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def main():
    logger.info("Starting BM25 index building process")
    total_start = time.time()
    
    # Paths
    passage_ids_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_subset.tsv"
    index_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25/bm25_index.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    logger.info(f"Output directory: {os.path.dirname(index_file)}")
    
    # Load passage IDs
    passage_ids = load_passage_ids(passage_ids_file)
    
    # Create mock documents (since we don't have actual passage text)
    documents = create_mock_documents(passage_ids)
    
    # Build BM25 index
    index = build_bm25_index(documents)
    
    # Save index
    save_index(index, index_file)
    
    # Test search
    logger.info("Testing search with sample query...")
    test_query = ['the', 'search', 'engine', 'information', 'retrieval']
    results = index.search(test_query, k=5)
    
    logger.info("Search results:")
    for i, (doc_id, score) in enumerate(results):
        logger.info(f"{i+1}. Document ID: {doc_id}, Score: {score:.4f}")
    
    total_time = time.time() - total_start
    logger.info(f"BM25 index built and saved successfully!")
    logger.info(f"Total process time: {total_time:.2f} seconds")
    logger.info(f"Index file: {index_file}")

if __name__ == "__main__":
    main()

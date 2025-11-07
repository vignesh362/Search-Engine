#!/usr/bin/env python3
"""
BM25 Index Builder for MS MARCO Passage Subset
Uses passage IDs from msmarco_passages_subset.tsv to filter collection.tsv
and build BM25 index for only those passages.
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
        logging.FileHandler('bm25_subset_build.log'),
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
        self.doc_texts = {}  # doc_id -> original text
        
    def add_document(self, doc_id, text, terms):
        """Add a document to the index"""
        term_counts = Counter(terms)
        doc_length = len(terms)
        
        self.doc_lengths[doc_id] = doc_length
        self.doc_texts[doc_id] = text
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
    
    def search(self, query_terms, k=10, conjunctive=False):
        """Search for documents matching query terms
        
        Args:
            query_terms: List of search terms
            k: Number of top results to return
            conjunctive: If True, require all terms (AND); if False, require any term (OR)
        """
        logger.info(f"Searching for terms: {query_terms} ({'AND' if conjunctive else 'OR'} mode)")
        
        if conjunctive:
            # AND mode: Get documents that contain ALL query terms
            if not query_terms:
                return []
            
            # Start with documents containing the first term
            candidate_docs = None
            for term in query_terms:
                if term in self.postings:
                    term_docs = {doc_id for doc_id, _ in self.postings[term]}
                    if candidate_docs is None:
                        candidate_docs = term_docs
                    else:
                        candidate_docs = candidate_docs.intersection(term_docs)
                else:
                    # If any term is not in index, no documents can match all terms
                    return []
            
            if candidate_docs is None:
                candidate_docs = set()
        else:
            # OR mode: Get all documents that contain at least one query term
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

def clean_text(text):
    """Clean and tokenize text for indexing"""
    import re
    # Convert to lowercase and extract alphanumeric tokens
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return tokens

def load_passage_ids(file_path):
    """Load passage IDs from subset TSV file"""
    logger.info(f"Loading passage IDs from {file_path}...")
    start_time = time.time()
    
    passage_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                passage_ids.add(line)
    
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(passage_ids)} unique passage IDs in {load_time:.2f} seconds")
    return passage_ids

def load_collection_subset(collection_file, passage_ids):
    """Load collection and filter to only include passages from the subset"""
    logger.info(f"Loading collection from {collection_file} and filtering by subset...")
    start_time = time.time()
    
    documents = {}
    total_lines = 0
    matched_lines = 0
    
    with open(collection_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            if line_num % 100000 == 0:
                logger.info(f"Processed {line_num} lines, matched {matched_lines} passages...")
            
            line = line.strip()
            if not line:
                continue
            
            # Parse TSV format: passage_id \t passage_text
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            passage_id, passage_text = parts
            
            # Check if this passage ID is in our subset
            if passage_id in passage_ids:
                documents[passage_id] = passage_text
                matched_lines += 1
    
    load_time = time.time() - start_time
    logger.info(f"Collection loading completed in {load_time:.2f} seconds")
    logger.info(f"Total lines processed: {total_lines}")
    logger.info(f"Matched passages: {matched_lines}")
    logger.info(f"Expected passages: {len(passage_ids)}")
    
    return documents

def build_bm25_index(documents):
    """Build BM25 index from documents"""
    logger.info("Building BM25 index...")
    start_time = time.time()
    
    index = BM25Index()
    
    for i, (doc_id, text) in enumerate(documents.items()):
        # Clean and tokenize text
        terms = clean_text(text)
        
        # Add document to index
        index.add_document(doc_id, text, terms)
        
        if (i + 1) % 10000 == 0:
            logger.info(f"Indexed {i + 1} documents...")
    
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
    logger.info("Starting BM25 index building process for MS MARCO subset")
    total_start = time.time()
    
    # Paths
    passage_ids_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_subset.tsv"
    collection_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/collection.tsv"
    index_file = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25/bm25_subset_index.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    logger.info(f"Output directory: {os.path.dirname(index_file)}")
    
    # Check if files exist
    if not os.path.exists(passage_ids_file):
        logger.error(f"Passage IDs file not found: {passage_ids_file}")
        return
    
    if not os.path.exists(collection_file):
        logger.error(f"Collection file not found: {collection_file}")
        return
    
    # Load passage IDs
    passage_ids = load_passage_ids(passage_ids_file)
    
    # Load and filter collection
    documents = load_collection_subset(collection_file, passage_ids)
    
    if not documents:
        logger.error("No documents found matching the passage IDs!")
        return
    
    # Build BM25 index
    index = build_bm25_index(documents)
    
    # Save index
    save_index(index, index_file)
    
    # Test search
    logger.info("Testing search with sample query...")
    test_query = ['search', 'engine', 'information', 'retrieval']
    results = index.search(test_query, k=5)
    
    logger.info("Search results:")
    for i, (doc_id, score) in enumerate(results):
        # Show snippet of the document text
        text_snippet = index.doc_texts[doc_id][:100] + "..." if len(index.doc_texts[doc_id]) > 100 else index.doc_texts[doc_id]
        logger.info(f"{i+1}. Document ID: {doc_id}, Score: {score:.4f}")
        logger.info(f"   Text: {text_snippet}")
    
    total_time = time.time() - total_start
    logger.info(f"BM25 subset index built and saved successfully!")
    logger.info(f"Total process time: {total_time:.2f} seconds")
    logger.info(f"Index file: {index_file}")

if __name__ == "__main__":
    main()

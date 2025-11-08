#!/usr/bin/env python3
"""
Test script to compare BM25, Dense, and Hybrid search across multiple queries.
Shows differences in ranking, scores, and document retrieval.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import argparse

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
from data.ms_marco_data import load_h5_ids_vecs
from runs.run_io import read_trec_run, write_trec_run, rerank_with_dense

# Test queries - mix of short and long, different topics
TEST_QUERIES = [
    # Short queries
    "what is python",
    "covid symptoms",
    "best laptop",
    "weather forecast",
    "machine learning",
    "restaurant near me",
    "stock market",
    "headache causes",
    "quick recipes",
    "fitness tips",
    
    # Long queries
    "how to train a neural network for image classification",
    "what are the health benefits of drinking green tea daily",
    "step by step guide to building a website from scratch",
    "differences between classical and quantum computing systems",
    "how does climate change affect global ocean temperatures",
    "what causes inflation and how does it impact the economy",
    "best practices for writing clean and maintainable code",
    "how to prepare for a technical job interview at big tech",
    "understanding the relationship between diet and mental health",
    "what are the main differences between supervised and unsupervised learning",
]

def load_bm25_index():
    """Load BM25 index for searching"""
    try:
        import subprocess
        return True
    except:
        return False

def search_bm25(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Run BM25 search using QueryProcessing binary"""
    import subprocess
    import json
    
    query_processor = './bm25/QueryProcessing'
    index_dir = './bm25/index_output'
    
    if not os.path.exists(query_processor):
        print(f"Warning: BM25 QueryProcessing binary not found at {query_processor}")
        return []
    
    try:
        # Run the query processor
        cmd = [query_processor, '--index-dir', index_dir, '--query', query, '--k', str(k), '--mode', 'disjunctive']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"BM25 search failed for query: {query}")
            return []
        
        # Parse output (assuming it returns doc_id score pairs)
        results = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    doc_id = parts[0]
                    score = float(parts[1])
                    results.append((doc_id, score))
        
        return results[:k]
    except Exception as e:
        print(f"Error running BM25 search: {e}")
        return []

def search_dense(query: str, query_vecs: np.ndarray, passage_ids: np.ndarray, 
                 passage_vecs: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Run dense search using embeddings"""
    # For simplicity, use a hash-based query embedding (in real scenario, encode the query)
    # Here we'll just use a random query vector as placeholder
    query_hash = hash(query) % len(query_vecs)
    query_vec = query_vecs[query_hash].reshape(1, -1)
    
    # Compute similarities with all passages
    similarities = np.dot(passage_vecs, query_vec.T).flatten()
    
    # Get top-k
    top_k_idx = np.argsort(-similarities)[:k]
    
    results = [(str(passage_ids[i]), float(similarities[i])) for i in top_k_idx]
    return results

def search_hybrid(query: str, bm25_results: List[Tuple[str, float]], 
                  query_vecs: np.ndarray, passage_ids: np.ndarray,
                  passage_vecs: np.ndarray, k: int = 10, 
                  candidate_k: int = 100) -> List[Tuple[str, float]]:
    """Run cascading hybrid search"""
    if not bm25_results:
        return []
    
    # Get BM25 candidates
    candidate_ids = [doc_id for doc_id, _ in bm25_results[:candidate_k]]
    
    # Create mapping
    pid_to_idx = {str(pid): idx for idx, pid in enumerate(passage_ids)}
    
    # Get query vector (using hash as placeholder)
    query_hash = hash(query) % len(query_vecs)
    query_vec = query_vecs[query_hash].reshape(1, -1)
    
    # Get candidate vectors and rerank
    valid_ids = []
    valid_vecs = []
    for doc_id in candidate_ids:
        if doc_id in pid_to_idx:
            valid_ids.append(doc_id)
            valid_vecs.append(passage_vecs[pid_to_idx[doc_id]])
    
    if not valid_ids:
        return []
    
    valid_vecs = np.array(valid_vecs)
    scores = np.dot(valid_vecs, query_vec.T).flatten()
    
    # Sort and return top-k
    sorted_idx = np.argsort(-scores)[:k]
    results = [(valid_ids[i], float(scores[i])) for i in sorted_idx]
    
    return results

def calculate_overlap(results1: List[Tuple[str, float]], 
                     results2: List[Tuple[str, float]]) -> Tuple[int, float]:
    """Calculate overlap between two result sets"""
    ids1 = set(doc_id for doc_id, _ in results1)
    ids2 = set(doc_id for doc_id, _ in results2)
    
    overlap = len(ids1 & ids2)
    overlap_pct = (overlap / len(ids1) * 100) if ids1 else 0
    
    return overlap, overlap_pct

def calculate_rank_correlation(results1: List[Tuple[str, float]], 
                               results2: List[Tuple[str, float]]) -> float:
    """Calculate rank correlation (Kendall's tau approximation)"""
    # Get common documents
    ids1 = {doc_id: rank for rank, (doc_id, _) in enumerate(results1)}
    ids2 = {doc_id: rank for rank, (doc_id, _) in enumerate(results2)}
    
    common_ids = set(ids1.keys()) & set(ids2.keys())
    
    if len(common_ids) < 2:
        return 0.0
    
    # Simple rank correlation
    rank_diffs = [abs(ids1[doc_id] - ids2[doc_id]) for doc_id in common_ids]
    avg_diff = sum(rank_diffs) / len(rank_diffs)
    
    # Normalize to 0-1 scale (1 = perfect agreement, 0 = no agreement)
    max_possible_diff = len(results1)
    correlation = 1.0 - (avg_diff / max_possible_diff)
    
    return correlation

def print_results_comparison(query: str, bm25_results: List[Tuple[str, float]],
                            dense_results: List[Tuple[str, float]],
                            hybrid_results: List[Tuple[str, float]],
                            k: int = 10):
    """Print detailed comparison of results"""
    print("\n" + "="*100)
    print(f"QUERY: {query}")
    print("="*100)
    
    # Print top results side by side
    print(f"\n{'Rank':<6} {'BM25 (Score)':<35} {'Dense (Score)':<35} {'Hybrid (Score)':<35}")
    print("-"*100)
    
    for i in range(k):
        bm25_entry = f"{bm25_results[i][0]} ({bm25_results[i][1]:.4f})" if i < len(bm25_results) else "-"
        dense_entry = f"{dense_results[i][0]} ({dense_results[i][1]:.4f})" if i < len(dense_results) else "-"
        hybrid_entry = f"{hybrid_results[i][0]} ({hybrid_results[i][1]:.4f})" if i < len(hybrid_results) else "-"
        
        print(f"{i+1:<6} {bm25_entry:<35} {dense_entry:<35} {hybrid_entry:<35}")
    
    # Calculate overlaps
    print("\n" + "-"*100)
    print("OVERLAP ANALYSIS:")
    print("-"*100)
    
    bm25_dense_overlap, bm25_dense_pct = calculate_overlap(bm25_results, dense_results)
    bm25_hybrid_overlap, bm25_hybrid_pct = calculate_overlap(bm25_results, hybrid_results)
    dense_hybrid_overlap, dense_hybrid_pct = calculate_overlap(dense_results, hybrid_results)
    
    print(f"BM25 vs Dense:   {bm25_dense_overlap}/{k} documents ({bm25_dense_pct:.1f}% overlap)")
    print(f"BM25 vs Hybrid:  {bm25_hybrid_overlap}/{k} documents ({bm25_hybrid_pct:.1f}% overlap)")
    print(f"Dense vs Hybrid: {dense_hybrid_overlap}/{k} documents ({dense_hybrid_pct:.1f}% overlap)")
    
    # Rank correlation
    print("\n" + "-"*100)
    print("RANK CORRELATION (1.0 = perfect agreement, 0.0 = no agreement):")
    print("-"*100)
    
    bm25_dense_corr = calculate_rank_correlation(bm25_results, dense_results)
    bm25_hybrid_corr = calculate_rank_correlation(bm25_results, hybrid_results)
    dense_hybrid_corr = calculate_rank_correlation(dense_results, hybrid_results)
    
    print(f"BM25 vs Dense:   {bm25_dense_corr:.3f}")
    print(f"BM25 vs Hybrid:  {bm25_hybrid_corr:.3f}")
    print(f"Dense vs Hybrid: {dense_hybrid_corr:.3f}")
    
    # Unique documents
    print("\n" + "-"*100)
    print("UNIQUE DOCUMENTS (not in other systems):")
    print("-"*100)
    
    bm25_ids = set(doc_id for doc_id, _ in bm25_results)
    dense_ids = set(doc_id for doc_id, _ in dense_results)
    hybrid_ids = set(doc_id for doc_id, _ in hybrid_results)
    
    bm25_unique = bm25_ids - dense_ids - hybrid_ids
    dense_unique = dense_ids - bm25_ids - hybrid_ids
    hybrid_unique = hybrid_ids - bm25_ids - dense_ids
    
    print(f"BM25 only:   {len(bm25_unique)} documents")
    print(f"Dense only:  {len(dense_unique)} documents")
    print(f"Hybrid only: {len(hybrid_unique)} documents")
    
    # Score statistics
    print("\n" + "-"*100)
    print("SCORE STATISTICS:")
    print("-"*100)
    
    if bm25_results:
        bm25_scores = [score for _, score in bm25_results]
        print(f"BM25:   min={min(bm25_scores):.4f}, max={max(bm25_scores):.4f}, "
              f"mean={np.mean(bm25_scores):.4f}, std={np.std(bm25_scores):.4f}")
    
    if dense_results:
        dense_scores = [score for _, score in dense_results]
        print(f"Dense:  min={min(dense_scores):.4f}, max={max(dense_scores):.4f}, "
              f"mean={np.mean(dense_scores):.4f}, std={np.std(dense_scores):.4f}")
    
    if hybrid_results:
        hybrid_scores = [score for _, score in hybrid_results]
        print(f"Hybrid: min={min(hybrid_scores):.4f}, max={max(hybrid_scores):.4f}, "
              f"mean={np.mean(hybrid_scores):.4f}, std={np.std(hybrid_scores):.4f}")

def print_summary_statistics(all_comparisons: List[Dict]):
    """Print summary statistics across all queries"""
    print("\n\n" + "="*100)
    print("SUMMARY STATISTICS ACROSS ALL QUERIES")
    print("="*100)
    
    # Aggregate statistics
    avg_bm25_dense_overlap = np.mean([c['bm25_dense_overlap_pct'] for c in all_comparisons])
    avg_bm25_hybrid_overlap = np.mean([c['bm25_hybrid_overlap_pct'] for c in all_comparisons])
    avg_dense_hybrid_overlap = np.mean([c['dense_hybrid_overlap_pct'] for c in all_comparisons])
    
    avg_bm25_dense_corr = np.mean([c['bm25_dense_corr'] for c in all_comparisons])
    avg_bm25_hybrid_corr = np.mean([c['bm25_hybrid_corr'] for c in all_comparisons])
    avg_dense_hybrid_corr = np.mean([c['dense_hybrid_corr'] for c in all_comparisons])
    
    print("\nAVERAGE OVERLAP:")
    print(f"  BM25 vs Dense:   {avg_bm25_dense_overlap:.1f}%")
    print(f"  BM25 vs Hybrid:  {avg_bm25_hybrid_overlap:.1f}%")
    print(f"  Dense vs Hybrid: {avg_dense_hybrid_overlap:.1f}%")
    
    print("\nAVERAGE RANK CORRELATION:")
    print(f"  BM25 vs Dense:   {avg_bm25_dense_corr:.3f}")
    print(f"  BM25 vs Hybrid:  {avg_bm25_hybrid_corr:.3f}")
    print(f"  Dense vs Hybrid: {avg_dense_hybrid_corr:.3f}")
    
    # Query type analysis
    short_queries = [c for c in all_comparisons if c['query_length'] <= 3]
    long_queries = [c for c in all_comparisons if c['query_length'] > 3]
    
    if short_queries and long_queries:
        print("\n" + "-"*100)
        print("QUERY LENGTH ANALYSIS:")
        print("-"*100)
        
        print(f"\nShort Queries (≤3 words, n={len(short_queries)}):")
        print(f"  Avg BM25-Dense overlap: {np.mean([c['bm25_dense_overlap_pct'] for c in short_queries]):.1f}%")
        print(f"  Avg BM25-Hybrid overlap: {np.mean([c['bm25_hybrid_overlap_pct'] for c in short_queries]):.1f}%")
        
        print(f"\nLong Queries (>3 words, n={len(long_queries)}):")
        print(f"  Avg BM25-Dense overlap: {np.mean([c['bm25_dense_overlap_pct'] for c in long_queries]):.1f}%")
        print(f"  Avg BM25-Hybrid overlap: {np.mean([c['bm25_hybrid_overlap_pct'] for c in long_queries]):.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Compare BM25, Dense, and Hybrid search systems')
    parser.add_argument('--k', type=int, default=10, help='Number of results to retrieve (default: 10)')
    parser.add_argument('--candidate-k', type=int, default=100, 
                       help='Number of BM25 candidates for hybrid (default: 100)')
    parser.add_argument('--queries', nargs='+', help='Custom queries to test (overrides default 20)')
    args = parser.parse_args()
    
    # Use custom queries or defaults
    queries = args.queries if args.queries else TEST_QUERIES
    
    print("="*100)
    print("SEARCH SYSTEM COMPARISON TEST")
    print("="*100)
    print(f"\nTesting {len(queries)} queries across BM25, Dense, and Hybrid search systems")
    print(f"Retrieving top-{args.k} results per query")
    print(f"Using candidate_k={args.candidate_k} for hybrid search")
    print("\nLoading embeddings...")
    
    # Load embeddings
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ms_marco')
    queries_h5 = os.path.join(DATA_DIR, 'msmarco_queries_dev_eval_embeddings.h5')
    passages_h5 = os.path.join(DATA_DIR, 'msmarco_passages_embeddings_subset.h5')
    
    query_ids, query_vecs = load_h5_ids_vecs(queries_h5)
    passage_ids, passage_vecs = load_h5_ids_vecs(passages_h5)
    
    print(f"Loaded {len(query_ids)} query embeddings and {len(passage_ids)} passage embeddings")
    
    # Store comparisons for summary
    all_comparisons = []
    
    # Test each query
    for i, query in enumerate(queries, 1):
        print(f"\n\nProcessing query {i}/{len(queries)}: {query}")
        
        # Run searches
        bm25_results = search_bm25(query, k=max(args.k, args.candidate_k))
        dense_results = search_dense(query, query_vecs, passage_ids, passage_vecs, k=args.k)
        hybrid_results = search_hybrid(query, bm25_results, query_vecs, passage_ids, 
                                      passage_vecs, k=args.k, candidate_k=args.candidate_k)
        
        # Print comparison
        print_results_comparison(query, bm25_results[:args.k], dense_results, 
                                hybrid_results, k=args.k)
        
        # Store for summary
        bm25_dense_overlap, bm25_dense_pct = calculate_overlap(bm25_results[:args.k], dense_results)
        bm25_hybrid_overlap, bm25_hybrid_pct = calculate_overlap(bm25_results[:args.k], hybrid_results)
        dense_hybrid_overlap, dense_hybrid_pct = calculate_overlap(dense_results, hybrid_results)
        
        all_comparisons.append({
            'query': query,
            'query_length': len(query.split()),
            'bm25_dense_overlap_pct': bm25_dense_pct,
            'bm25_hybrid_overlap_pct': bm25_hybrid_pct,
            'dense_hybrid_overlap_pct': dense_hybrid_pct,
            'bm25_dense_corr': calculate_rank_correlation(bm25_results[:args.k], dense_results),
            'bm25_hybrid_corr': calculate_rank_correlation(bm25_results[:args.k], hybrid_results),
            'dense_hybrid_corr': calculate_rank_correlation(dense_results, hybrid_results),
        })
    
    # Print summary
    print_summary_statistics(all_comparisons)
    
    print("\n" + "="*100)
    print("TEST COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()

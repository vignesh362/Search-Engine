#!/usr/bin/env python3
"""
Test script to compare BM25, Dense, and Hybrid search using existing run files.
Shows differences in ranking, scores, and document retrieval across 20 sample queries.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import argparse
import random

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
from runs.run_io import read_trec_run

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
    """Calculate rank correlation (based on average rank difference)"""
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
    max_possible_diff = max(len(results1), len(results2))
    correlation = max(0.0, 1.0 - (avg_diff / max_possible_diff))
    
    return correlation

def print_results_comparison(qid: str, bm25_results: List[Tuple[str, float]],
                            dense_results: List[Tuple[str, float]],
                            hybrid_results: List[Tuple[str, float]],
                            k: int = 10):
    """Print detailed comparison of results"""
    print("\n" + "="*120)
    print(f"QUERY ID: {qid}")
    print("="*120)
    
    # Print top results side by side
    print(f"\n{'Rank':<6} {'BM25 (Doc ID / Score)':<40} {'Dense (Doc ID / Score)':<40} {'Hybrid (Doc ID / Score)':<40}")
    print("-"*120)
    
    for i in range(k):
        bm25_entry = f"{bm25_results[i][0]} / {bm25_results[i][1]:.4f}" if i < len(bm25_results) else "-"
        dense_entry = f"{dense_results[i][0]} / {dense_results[i][1]:.4f}" if i < len(dense_results) else "-"
        hybrid_entry = f"{hybrid_results[i][0]} / {hybrid_results[i][1]:.4f}" if i < len(hybrid_results) else "-"
        
        # Highlight common documents across all three
        bm25_id = bm25_results[i][0] if i < len(bm25_results) else None
        dense_id = dense_results[i][0] if i < len(dense_results) else None
        hybrid_id = hybrid_results[i][0] if i < len(hybrid_results) else None
        
        marker = ""
        if bm25_id and bm25_id == dense_id == hybrid_id:
            marker = " ★"  # All three agree
        elif bm25_id and (bm25_id == dense_id or bm25_id == hybrid_id):
            marker = " •"  # Two agree
        
        print(f"{i+1:<6} {bm25_entry:<40} {dense_entry:<40} {hybrid_entry:<40}{marker}")
    
    print("\n★ = All three systems agree  • = Two systems agree")
    
    # Calculate overlaps
    print("\n" + "-"*120)
    print("OVERLAP ANALYSIS (Top-10 Documents):")
    print("-"*120)
    
    bm25_dense_overlap, bm25_dense_pct = calculate_overlap(bm25_results, dense_results)
    bm25_hybrid_overlap, bm25_hybrid_pct = calculate_overlap(bm25_results, hybrid_results)
    dense_hybrid_overlap, dense_hybrid_pct = calculate_overlap(dense_results, hybrid_results)
    
    print(f"BM25 vs Dense:   {bm25_dense_overlap:2d}/{k} documents ({bm25_dense_pct:5.1f}% overlap)")
    print(f"BM25 vs Hybrid:  {bm25_hybrid_overlap:2d}/{k} documents ({bm25_hybrid_pct:5.1f}% overlap)")
    print(f"Dense vs Hybrid: {dense_hybrid_overlap:2d}/{k} documents ({dense_hybrid_pct:5.1f}% overlap)")
    
    # Rank correlation
    print("\n" + "-"*120)
    print("RANK CORRELATION (1.0 = perfect agreement, 0.0 = no agreement):")
    print("-"*120)
    
    bm25_dense_corr = calculate_rank_correlation(bm25_results, dense_results)
    bm25_hybrid_corr = calculate_rank_correlation(bm25_results, hybrid_results)
    dense_hybrid_corr = calculate_rank_correlation(dense_results, hybrid_results)
    
    print(f"BM25 vs Dense:   {bm25_dense_corr:.3f}")
    print(f"BM25 vs Hybrid:  {bm25_hybrid_corr:.3f}")
    print(f"Dense vs Hybrid: {dense_hybrid_corr:.3f}")
    
    # Unique documents
    print("\n" + "-"*120)
    print("UNIQUE DOCUMENTS (only found by one system, not the other two):")
    print("-"*120)
    
    bm25_ids = set(doc_id for doc_id, _ in bm25_results)
    dense_ids = set(doc_id for doc_id, _ in dense_results)
    hybrid_ids = set(doc_id for doc_id, _ in hybrid_results)
    
    bm25_unique = bm25_ids - dense_ids - hybrid_ids
    dense_unique = dense_ids - bm25_ids - hybrid_ids
    hybrid_unique = hybrid_ids - bm25_ids - dense_ids
    
    print(f"BM25 only:   {len(bm25_unique):2d} documents", end="")
    if bm25_unique:
        print(f" (e.g., {', '.join(str(x) for x in list(bm25_unique)[:3])})")
    else:
        print()
    
    print(f"Dense only:  {len(dense_unique):2d} documents", end="")
    if dense_unique:
        print(f" (e.g., {', '.join(str(x) for x in list(dense_unique)[:3])})")
    else:
        print()
    
    print(f"Hybrid only: {len(hybrid_unique):2d} documents", end="")
    if hybrid_unique:
        print(f" (e.g., {', '.join(str(x) for x in list(hybrid_unique)[:3])})")
    else:
        print()
    
    # Score statistics
    print("\n" + "-"*120)
    print("SCORE STATISTICS:")
    print("-"*120)
    
    if bm25_results:
        bm25_scores = [score for _, score in bm25_results]
        print(f"BM25:   min={min(bm25_scores):7.4f}, max={max(bm25_scores):7.4f}, "
              f"mean={np.mean(bm25_scores):7.4f}, std={np.std(bm25_scores):7.4f}")
    
    if dense_results:
        dense_scores = [score for _, score in dense_results]
        print(f"Dense:  min={min(dense_scores):7.4f}, max={max(dense_scores):7.4f}, "
              f"mean={np.mean(dense_scores):7.4f}, std={np.std(dense_scores):7.4f}")
    
    if hybrid_results:
        hybrid_scores = [score for _, score in hybrid_results]
        print(f"Hybrid: min={min(hybrid_scores):7.4f}, max={max(hybrid_scores):7.4f}, "
              f"mean={np.mean(hybrid_scores):7.4f}, std={np.std(hybrid_scores):7.4f}")
    
    # Check for perfect matches
    if bm25_results and dense_results and hybrid_results:
        perfect_matches = 0
        for i in range(min(len(bm25_results), len(dense_results), len(hybrid_results))):
            if bm25_results[i][0] == dense_results[i][0] == hybrid_results[i][0]:
                perfect_matches += 1
        
        if perfect_matches > 0:
            print(f"\n🎯 {perfect_matches} document(s) at the same rank in all three systems!")

def print_summary_statistics(all_comparisons: List[Dict]):
    """Print summary statistics across all queries"""
    print("\n\n" + "="*120)
    print("SUMMARY STATISTICS ACROSS ALL QUERIES")
    print("="*120)
    
    # Aggregate statistics
    avg_bm25_dense_overlap = np.mean([c['bm25_dense_overlap_pct'] for c in all_comparisons])
    avg_bm25_hybrid_overlap = np.mean([c['bm25_hybrid_overlap_pct'] for c in all_comparisons])
    avg_dense_hybrid_overlap = np.mean([c['dense_hybrid_overlap_pct'] for c in all_comparisons])
    
    avg_bm25_dense_corr = np.mean([c['bm25_dense_corr'] for c in all_comparisons])
    avg_bm25_hybrid_corr = np.mean([c['bm25_hybrid_corr'] for c in all_comparisons])
    avg_dense_hybrid_corr = np.mean([c['dense_hybrid_corr'] for c in all_comparisons])
    
    print("\nOVERAGE OVERLAP ACROSS ALL QUERIES:")
    print("-"*120)
    print(f"  BM25 vs Dense:   {avg_bm25_dense_overlap:5.1f}% (range: {min(c['bm25_dense_overlap_pct'] for c in all_comparisons):.1f}% - {max(c['bm25_dense_overlap_pct'] for c in all_comparisons):.1f}%)")
    print(f"  BM25 vs Hybrid:  {avg_bm25_hybrid_overlap:5.1f}% (range: {min(c['bm25_hybrid_overlap_pct'] for c in all_comparisons):.1f}% - {max(c['bm25_hybrid_overlap_pct'] for c in all_comparisons):.1f}%)")
    print(f"  Dense vs Hybrid: {avg_dense_hybrid_overlap:5.1f}% (range: {min(c['dense_hybrid_overlap_pct'] for c in all_comparisons):.1f}% - {max(c['dense_hybrid_overlap_pct'] for c in all_comparisons):.1f}%)")
    
    print("\nAVERAGE RANK CORRELATION:")
    print("-"*120)
    print(f"  BM25 vs Dense:   {avg_bm25_dense_corr:.3f}")
    print(f"  BM25 vs Hybrid:  {avg_bm25_hybrid_corr:.3f}")
    print(f"  Dense vs Hybrid: {avg_dense_hybrid_corr:.3f}")
    
    # Distribution analysis
    print("\n" + "-"*120)
    print("OVERLAP DISTRIBUTION:")
    print("-"*120)
    
    high_overlap_bm25_hybrid = sum(1 for c in all_comparisons if c['bm25_hybrid_overlap_pct'] >= 70)
    medium_overlap_bm25_hybrid = sum(1 for c in all_comparisons if 30 <= c['bm25_hybrid_overlap_pct'] < 70)
    low_overlap_bm25_hybrid = sum(1 for c in all_comparisons if c['bm25_hybrid_overlap_pct'] < 30)
    
    print(f"BM25 vs Hybrid:")
    print(f"  High overlap (≥70%):     {high_overlap_bm25_hybrid:2d} queries")
    print(f"  Medium overlap (30-70%): {medium_overlap_bm25_hybrid:2d} queries")
    print(f"  Low overlap (<30%):      {low_overlap_bm25_hybrid:2d} queries")
    
    high_overlap_bm25_dense = sum(1 for c in all_comparisons if c['bm25_dense_overlap_pct'] >= 70)
    medium_overlap_bm25_dense = sum(1 for c in all_comparisons if 30 <= c['bm25_dense_overlap_pct'] < 70)
    low_overlap_bm25_dense = sum(1 for c in all_comparisons if c['bm25_dense_overlap_pct'] < 30)
    
    print(f"\nBM25 vs Dense:")
    print(f"  High overlap (≥70%):     {high_overlap_bm25_dense:2d} queries")
    print(f"  Medium overlap (30-70%): {medium_overlap_bm25_dense:2d} queries")
    print(f"  Low overlap (<30%):      {low_overlap_bm25_dense:2d} queries")
    
    # Key insights
    print("\n" + "="*120)
    print("KEY INSIGHTS:")
    print("="*120)
    
    print(f"\n1. Hybrid retrieves on average {avg_bm25_hybrid_overlap:.1f}% of the same documents as BM25")
    print(f"   → This makes sense since Hybrid reranks BM25 candidates")
    
    print(f"\n2. BM25 and Dense only agree on {avg_bm25_dense_overlap:.1f}% of documents on average")
    print(f"   → Shows different retrieval strategies (term-based vs semantic)")
    
    print(f"\n3. Dense and Hybrid agree on {avg_dense_hybrid_overlap:.1f}% of documents")
    print(f"   → Hybrid uses dense scoring, so higher agreement expected")
    
    if avg_bm25_hybrid_corr > 0.5:
        print(f"\n4. Rank correlation BM25-Hybrid is {avg_bm25_hybrid_corr:.3f}")
        print(f"   → Cascading preserves some BM25 ranking structure")
    else:
        print(f"\n4. Rank correlation BM25-Hybrid is {avg_bm25_hybrid_corr:.3f}")
        print(f"   → Dense reranking significantly changes document order")

def main():
    parser = argparse.ArgumentParser(description='Compare BM25, Dense, and Hybrid search systems using run files')
    parser.add_argument('--dataset', choices=['dev', 'eval1', 'eval2'], default='dev',
                       help='Dataset to use (default: dev)')
    parser.add_argument('--k', type=int, default=10, 
                       help='Number of results to compare (default: 10)')
    parser.add_argument('--num-queries', type=int, default=20,
                       help='Number of random queries to sample (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for query sampling (default: 42)')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*120)
    print("SEARCH SYSTEM COMPARISON TEST (Using Existing Run Files)")
    print("="*120)
    print(f"\nDataset: {args.dataset}")
    print(f"Comparing top-{args.k} results per query")
    print(f"Analyzing {args.num_queries} randomly sampled queries")
    
    # Load run files
    RUNS_DIR = os.path.join(os.path.dirname(__file__), 'runs')
    
    bm25_file = os.path.join(RUNS_DIR, f'bm25.{args.dataset}.trec')
    dense_file = os.path.join(RUNS_DIR, f'dense.{args.dataset}.trec')
    hybrid_file = os.path.join(RUNS_DIR, f'hybrid.{args.dataset}.trec')
    
    print(f"\nLoading run files...")
    print(f"  BM25:   {bm25_file}")
    print(f"  Dense:  {dense_file}")
    print(f"  Hybrid: {hybrid_file}")
    
    bm25_run = read_trec_run(bm25_file)
    dense_run = read_trec_run(dense_file)
    hybrid_run = read_trec_run(hybrid_file)
    
    # Find common queries
    common_qids = set(bm25_run.keys()) & set(dense_run.keys()) & set(hybrid_run.keys())
    print(f"\nFound {len(common_qids)} queries common to all three systems")
    
    # Sample random queries
    sampled_qids = sorted(random.sample(list(common_qids), min(args.num_queries, len(common_qids))))
    print(f"Sampled {len(sampled_qids)} queries for analysis")
    
    # Store comparisons for summary
    all_comparisons = []
    
    # Test each query
    for i, qid in enumerate(sampled_qids, 1):
        print(f"\n\nProcessing query {i}/{len(sampled_qids)}: QID {qid}")
        
        bm25_results = bm25_run[qid][:args.k]
        dense_results = dense_run[qid][:args.k]
        hybrid_results = hybrid_run[qid][:args.k]
        
        # Print comparison
        print_results_comparison(qid, bm25_results, dense_results, hybrid_results, k=args.k)
        
        # Store for summary
        bm25_dense_overlap, bm25_dense_pct = calculate_overlap(bm25_results, dense_results)
        bm25_hybrid_overlap, bm25_hybrid_pct = calculate_overlap(bm25_results, hybrid_results)
        dense_hybrid_overlap, dense_hybrid_pct = calculate_overlap(dense_results, hybrid_results)
        
        all_comparisons.append({
            'qid': qid,
            'bm25_dense_overlap_pct': bm25_dense_pct,
            'bm25_hybrid_overlap_pct': bm25_hybrid_pct,
            'dense_hybrid_overlap_pct': dense_hybrid_pct,
            'bm25_dense_corr': calculate_rank_correlation(bm25_results, dense_results),
            'bm25_hybrid_corr': calculate_rank_correlation(bm25_results, hybrid_results),
            'dense_hybrid_corr': calculate_rank_correlation(dense_results, hybrid_results),
        })
    
    # Print summary
    print_summary_statistics(all_comparisons)
    
    print("\n" + "="*120)
    print("TEST COMPLETE")
    print("="*120)

if __name__ == "__main__":
    main()

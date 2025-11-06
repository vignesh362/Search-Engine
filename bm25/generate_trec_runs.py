#!/usr/bin/env python3
"""Generate BM25 run files for TREC eval queries"""

import subprocess
import json
import os
from pathlib import Path
import h5py
import numpy as np

def run_bm25_query(query, topk=1000):
    """Run BM25 search for a query"""
    cmd = ['./bm25/QueryProcessing', query, '-k', str(topk)]
    try:
        output = subprocess.check_output(cmd, text=True)
        results = []
        for line in output.split('\n'):
            if 'doc=' in line and 'score=' in line:
                parts = line.split()
                doc_id = parts[1].split('=')[1]
                score = float(parts[2].split('=')[1])
                results.append((doc_id, score))
        return results
    except subprocess.CalledProcessError as e:
        print(f"Error processing query: {query}")
        print(f"Error details: {e}")
        return []

def get_eval_queries():
    """Get example queries for TREC eval"""
    return {
        19335: "what is a bank transit number",
        47923: "who is robert gray the poet",
        87181: "what causes cavities between teeth",
    }

def run_bm25_query(query, topk=1000):
    """Run BM25 search for a query"""
    cmd = ['./bm25/QueryProcessing', query, '-k', str(topk)]
    try:
        output = subprocess.check_output(cmd, text=True)
        results = []
        for line in output.split('\n'):
            if 'doc=' in line and 'score=' in line:
                parts = line.split()
                doc_id = parts[1].split('=')[1]
                score = float(parts[2].split('=')[1])
                results.append((doc_id, score))
        return results
    except subprocess.CalledProcessError as e:
        print(f"Error processing query: {query}")
        print(f"Error details: {e}")
        return []

def main():
    base_dir = Path(__file__).parent.parent
    
    # Load example queries
    queries = get_eval_queries()
    total_queries = len(queries)
    print(f"Processing {total_queries} evaluation queries")
    
    # Generate run file with example queries
    run_file = base_dir / 'runs/bm25.trec'
    with open(run_file, 'w') as out:
        for i, (qid, query) in enumerate(queries.items(), 1):
            print(f"\nProcessing query {i}/{total_queries}: {qid} - {query}")
            
            results = run_bm25_query(query)
            print(f"Found {len(results)} results")
            
            # Write results in TREC format
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score} bm25\n")
    
    # Generate run file
    with open('runs/bm25.trec', 'w') as out:
        for qid in eval_qids:
            if qid not in queries:
                print(f"Warning: Query {qid} not found in H5 file")
                continue
                
            query = queries[qid]
            results = run_bm25_query(query)
            
            # Write results in TREC format
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score} bm25\n")
            
            print(f"Processed query {qid}: {len(results)} results")

if __name__ == '__main__':
    main()
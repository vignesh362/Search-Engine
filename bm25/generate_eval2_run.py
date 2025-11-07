#!/usr/bin/env python3
"""Generate BM25 run file for eval2 queries using QueryProcessing binary."""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_eval2_queries(queries_file: str) -> Dict[int, str]:
    """Load eval2 queries from TSV file."""
    queries = {}
    logger.info(f"Loading queries from {queries_file}")
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = int(parts[0])
                query_text = parts[1]
                queries[qid] = query_text
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries

def run_bm25_query(query_text: str, topk: int = 1000) -> List[Tuple[str, float]]:
    """Run BM25 search using QueryProcessing binary."""
    cmd = ['./bm25/QueryProcessing', query_text, '-k', str(topk)]
    
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        results = []
        
        for line in output.split('\n'):
            # Parse output like: "doc=7068445 score=12.34"
            if 'doc=' in line and 'score=' in line:
                parts = line.split()
                doc_id = None
                score = None
                
                for part in parts:
                    if part.startswith('doc='):
                        doc_id = part.split('=')[1]
                    elif part.startswith('score='):
                        score = float(part.split('=')[1])
                
                if doc_id and score is not None:
                    results.append((doc_id, score))
        
        return results
    
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error processing query '{query_text}': {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error for query '{query_text}': {e}")
        return []

def load_eval2_qrels_queries(qrels_file: str) -> set:
    """Load query IDs from eval2 qrels file."""
    query_ids = set()
    logger.info(f"Loading qrels from {qrels_file}")
    
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid = int(parts[0])
                query_ids.add(qid)
    
    logger.info(f"Found {len(query_ids)} unique queries in qrels")
    return query_ids

def main():
    base_dir = Path(__file__).parent.parent
    
    # Paths
    queries_file = base_dir / 'data/ms_marco/eval2-queries.tsv'
    qrels_file = base_dir / 'data/ms_marco/qrels.eval.two.tsv'
    output_file = base_dir / 'runs/bm25.eval2.trec'
    
    # Check if QueryProcessing binary exists
    query_binary = base_dir / 'bm25/QueryProcessing'
    if not query_binary.exists():
        logger.error(f"QueryProcessing binary not found at {query_binary}")
        logger.error("Please build the BM25 index first using: cd bm25 && ./build_search_index.sh")
        return 1
    
    # Load all eval2 queries
    all_queries = load_eval2_queries(str(queries_file))
    
    # Load query IDs from qrels to know which queries to process
    qrels_query_ids = load_eval2_qrels_queries(str(qrels_file))
    
    # Filter queries to only those in qrels
    queries_to_process = {qid: text for qid, text in all_queries.items() if qid in qrels_query_ids}
    
    logger.info(f"Processing {len(queries_to_process)} queries that have qrels")
    
    # Generate run file
    with open(output_file, 'w') as out:
        processed = 0
        for qid in sorted(queries_to_process.keys()):
            query_text = queries_to_process[qid]
            
            # Run BM25 search
            results = run_bm25_query(query_text, topk=1000)
            
            if not results:
                logger.warning(f"No results for query {qid}: '{query_text}'")
                continue
            
            # Write results in TREC format
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\tbm25\n")
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{len(queries_to_process)} queries")
    
    logger.info(f"Successfully generated run file: {output_file}")
    logger.info(f"Total queries processed: {processed}/{len(queries_to_process)}")
    
    return 0

if __name__ == '__main__':
    exit(main())

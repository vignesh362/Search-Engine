#!/usr/bin/env python3
"""Generate BM25 run files for all query sets (dev, eval.one, eval.two)."""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import re
import string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_query_term(term: str) -> str:
    """Clean a single query term by removing punctuation (except hyphens within words)."""
    # Remove leading/trailing punctuation
    term = term.strip(string.punctuation)
    # Replace contractions/possessives: can't -> cant, mcentire's -> mcentire
    term = term.replace("'", "")
    # Remove remaining punctuation except hyphens
    term = re.sub(r'[^\w\s-]', '', term)
    return term.lower()

def load_queries(queries_file: str) -> Dict[int, str]:
    """Load queries from TSV file."""
    queries = {}
    logger.info(f"Loading queries from {queries_file}")
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = int(parts[0])
                query_text = '\t'.join(parts[1:])  # Handle queries with tabs
                queries[qid] = query_text
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries

def load_qrels_query_ids(qrels_file: str) -> set:
    """Load query IDs from qrels file."""
    query_ids = set()
    
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid = int(parts[0])
                query_ids.add(qid)
    
    return query_ids

def run_bm25_query(query_text: str, topk: int = 1000) -> List[Tuple[str, float]]:
    """Run BM25 search using QueryProcessing binary."""
    # Tokenize and clean query - QueryProcessing expects each word as separate argument
    query_terms = [clean_query_term(term) for term in query_text.split()]
    # Remove empty terms
    query_terms = [t for t in query_terms if t]
    
    if not query_terms:
        logger.warning(f"No valid terms after cleaning: '{query_text[:50]}...'")
        return []
    
    cmd = ['./QueryProcessing'] + query_terms + ['-k', str(topk)]
    
    try:
        # Note: QueryProcessing outputs to stdout, diagnostics to stderr
        # Must run from bm25/ directory where index_output/ exists
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd='bm25')
        results = []
        
        for line in result.stdout.splitlines():
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
    
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout for query: '{query_text[:50]}...'")
        return []
    except Exception as e:
        logger.warning(f"Error processing query '{query_text[:50]}...': {e}")
        return []

def generate_run_file(queries_file: str, qrels_file: str, output_file: str, run_name: str):
    """Generate a BM25 run file for a specific query set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating {run_name} run file")
    logger.info(f"{'='*60}")
    
    # Load all queries
    all_queries = load_queries(queries_file)
    
    # Load query IDs from qrels
    qrels_query_ids = load_qrels_query_ids(qrels_file)
    
    # Filter queries to only those in qrels
    queries_to_process = {qid: text for qid, text in all_queries.items() if qid in qrels_query_ids}
    
    logger.info(f"Found {len(queries_to_process)}/{len(qrels_query_ids)} queries in query file")
    
    if len(queries_to_process) == 0:
        logger.error("No matching queries found!")
        return False
    
    # Generate run file
    with open(output_file, 'w') as out:
        processed = 0
        failed = 0
        
        for qid in sorted(queries_to_process.keys()):
            query_text = queries_to_process[qid]
            
            # Run BM25 search
            results = run_bm25_query(query_text, topk=1000)
            
            if not results:
                logger.warning(f"No results for query {qid}: '{query_text[:50]}...'")
                failed += 1
                continue
            
            # Write results in TREC format
            for rank, (doc_id, score) in enumerate(results, 1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n")
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{len(queries_to_process)} queries")
    
    logger.info(f"Successfully generated: {output_file}")
    logger.info(f"Processed: {processed}/{len(queries_to_process)} queries")
    logger.info(f"Failed: {failed} queries")
    
    return processed > 0

def main():
    base_dir = Path(__file__).parent.parent
    
    # Check if QueryProcessing binary exists
    query_binary = base_dir / 'bm25/QueryProcessing'
    if not query_binary.exists():
        logger.error(f"QueryProcessing binary not found at {query_binary}")
        logger.error("Please build the BM25 index first using: cd bm25 && ./build_search_index.sh")
        return 1
    
    # Configuration for each run
    runs_config = [
        {
            'name': 'dev',
            'queries_file': base_dir / 'data/ms_marco/queries.dev.tsv',
            'qrels_file': base_dir / 'data/ms_marco/qrels.dev.tsv',
            'output_file': base_dir / 'runs/bm25.dev.trec',
            'run_name': 'bm25'
        },
        {
            'name': 'eval.one',
            'queries_file': base_dir / 'data/ms_marco/queries.eval.tsv',
            'qrels_file': base_dir / 'data/ms_marco/qrels.eval.one.tsv',
            'output_file': base_dir / 'runs/bm25.eval1.trec',
            'run_name': 'bm25'
        },
        {
            'name': 'eval.two',
            'queries_file': base_dir / 'data/ms_marco/queries.eval.tsv',
            'qrels_file': base_dir / 'data/ms_marco/qrels.eval.two.tsv',
            'output_file': base_dir / 'runs/bm25.eval2.trec',
            'run_name': 'bm25'
        }
    ]
    
    # Generate each run file
    success_count = 0
    for config in runs_config:
        try:
            if generate_run_file(
                str(config['queries_file']),
                str(config['qrels_file']),
                str(config['output_file']),
                config['run_name']
            ):
                success_count += 1
        except Exception as e:
            logger.error(f"Error generating {config['name']} run: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary: Generated {success_count}/{len(runs_config)} run files successfully")
    logger.info(f"{'='*60}")
    
    return 0 if success_count == len(runs_config) else 1

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Extract passages for the 1M subset from full collection."""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load subset IDs
    logger.info("Loading subset passage IDs...")
    subset_ids = set()
    with open('data/ms_marco/msmarco_passages_subset.tsv', 'r') as f:
        for line in f:
            pid = line.strip()
            if pid:
                subset_ids.add(pid)
    
    logger.info(f"Loaded {len(subset_ids):,} passage IDs from subset")
    
    # Extract passages from full collection
    logger.info("Extracting passages from full collection...")
    output_file = 'bm25/collection_subset.tsv'
    matches = 0
    
    with open('data/collection.tsv', 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line_num, line in enumerate(fin, 1):
                if line_num % 1000000 == 0:
                    logger.info(f"Processed {line_num:,} documents, found {matches:,} matches")
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    if doc_id in subset_ids:
                        fout.write(line)
                        matches += 1
                        
                if matches == len(subset_ids):
                    break
    
    logger.info(f"Extraction complete!")
    logger.info(f"Wrote {matches:,} passages to {output_file}")

if __name__ == '__main__':
    main()

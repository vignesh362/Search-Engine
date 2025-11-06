"""Dense search using FAISS indexes with various configurations."""

import argparse
import faiss
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List

from data.ms_marco_data import load_h5_ids_vecs, l2_normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_index(
    vectors: np.ndarray,
    index_kind: str,
    nlist: int = 100,
    nprobe: int = 10,
    hnsw_m: int = 16,
    ef_construction: int = 100,
    ef_search: int = 100
) -> faiss.Index:
    """Build a FAISS index of the specified type."""
    d = vectors.shape[1]
    
    if index_kind == 'flat_ip':
        index = faiss.IndexFlatIP(d)
    elif index_kind == 'flat_l2':
        index = faiss.IndexFlatL2(d)
    elif index_kind == 'ivf_ip':
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
    elif index_kind == 'ivf_l2':
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.nprobe = nprobe
    elif index_kind == 'hnsw_ip':
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    elif index_kind == 'hnsw_l2':
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_L2)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"Unknown index kind: {index_kind}")
    
    if 'ivf' in index_kind:
        logger.info("Training IVF index...")
        index.train(vectors)
    
    logger.info("Adding vectors to index...")
    index.add(vectors)
    return index

def search_queries(
    index: faiss.Index,
    query_vecs: np.ndarray,
    topk: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Search for nearest neighbors of query vectors."""
    return index.search(query_vecs, topk)

def write_trec_run(
    qids: np.ndarray,
    pids: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    run_path: str,
    run_name: str
):
    """Write search results in TREC format."""
    with open(run_path, 'w') as f:
        for qix, qid in enumerate(qids):
            # Sort by score descending
            sorted_idx = np.argsort(-D[qix])
            for rank, idx in enumerate(sorted_idx, 1):
                score = D[qix][idx]
                pid = pids[I[qix][idx]]
                f.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score}\t{run_name}\n")
    
    logger.info(f"Wrote TREC run to {run_path}")

def main():
    parser = argparse.ArgumentParser(description="Dense search using FAISS")
    parser.add_argument("--passages_h5", required=True, help="Path to passage embeddings")
    parser.add_argument("--queries_h5", required=True, help="Path to query embeddings")
    parser.add_argument(
        "--index_kind",
        choices=['flat_ip', 'ivf_ip', 'hnsw_ip', 'flat_l2', 'ivf_l2', 'hnsw_l2'],
        default='flat_ip',
        help="Type of FAISS index to build"
    )
    parser.add_argument("--normalize", action='store_true', help="L2-normalize vectors")
    parser.add_argument("--topk", type=int, default=1000, help="Number of results per query")
    parser.add_argument("--run_path", required=True, help="Output TREC run path")
    parser.add_argument("--run_name", default="dense", help="Run name for TREC format")
    
    # ANN parameters
    parser.add_argument("--nlist", type=int, default=100, help="IVF: Number of clusters")
    parser.add_argument("--nprobe", type=int, default=10, help="IVF: Number of clusters to search")
    parser.add_argument("--hnsw_m", type=int, default=16, help="HNSW: Connections per layer")
    parser.add_argument("--ef_search", type=int, default=100, help="HNSW: Search time expansion factor")
    parser.add_argument("--ef_construction", type=int, default=100, help="HNSW: Build time expansion factor")
    
    args = parser.parse_args()
    
    # Load data
    passage_ids, passage_vecs = load_h5_ids_vecs(args.passages_h5)
    query_ids, query_vecs = load_h5_ids_vecs(args.queries_h5)
    
    # Normalize if requested
    if args.normalize:
        logger.info("L2-normalizing vectors")
        passage_vecs = l2_normalize(passage_vecs)
        query_vecs = l2_normalize(query_vecs)
    
    # Build index
    index = build_index(
        passage_vecs,
        args.index_kind,
        args.nlist,
        args.nprobe,
        args.hnsw_m,
        args.ef_construction,
        args.ef_search
    )
    
    # Search
    logger.info(f"Searching {len(query_ids)} queries...")
    D, I = search_queries(index, query_vecs, args.topk)
    
    # Write results
    write_trec_run(query_ids, passage_ids, D, I, args.run_path, args.run_name)

if __name__ == "__main__":
    main()

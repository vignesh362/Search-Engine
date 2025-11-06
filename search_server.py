#!/usr/bin/env python3
"""
Unified Search Backend Server
Handles search requests using BM25, Dense Vector search, and Hybrid search systems
Combines functionality from all previous search server implementations
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import re
import os
import time
import logging
import sys
import pickle
import numpy as np
import faiss
import h5py
from urllib.parse import parse_qs
from typing import List, Dict, Any, Tuple

# Try to import requests for LM Studio API calls
try:
    import requests
except ImportError:
    requests = None

# Import BM25Index class to fix pickle loading
from bm25.build_bm25_from_subset import BM25Index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedSearchHandler(BaseHTTPRequestHandler):
    
    def __init__(self, *args, **kwargs):
        # Initialize search systems
        self.bm25_index = None
        self.faiss_index = None
        self.passage_ids = None
        self.embeddings = None
        self.load_search_systems()
        super().__init__(*args, **kwargs)
    
    def load_search_systems(self):
        """Load both BM25 and FAISS search systems"""
        try:
            # Load BM25 index
            bm25_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25/bm25_subset_index.pkl"
            if os.path.exists(bm25_path):
                logger.info("Loading BM25 index...")
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                logger.info(f"BM25 index loaded: {self.bm25_index.total_docs} documents")
            else:
                logger.warning("BM25 index not found, BM25 search will be disabled")
            
            # Load FAISS index
            faiss_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/faiss_ivf_index.bin"
            ids_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/passage_ids.pkl"
            
            if os.path.exists(faiss_path) and os.path.exists(ids_path):
                logger.info("Loading FAISS index...")
                self.faiss_index = faiss.read_index(faiss_path)
                with open(ids_path, 'rb') as f:
                    self.passage_ids = pickle.load(f)
                logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
            else:
                logger.warning("FAISS index not found, dense search will be disabled")
            
            # Load embeddings for query encoding
            embeddings_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_embeddings_subset.h5"
            if os.path.exists(embeddings_path):
                logger.info("Loading embeddings for query encoding...")
                with h5py.File(embeddings_path, 'r') as f:
                    self.embeddings = np.array(f['embedding']).astype(np.float32)
                logger.info(f"Embeddings loaded: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error loading search systems: {e}")
    
    def do_GET(self):
        """Serve the HTML frontend"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open('search_frontend.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle search requests"""
        if self.path == '/search':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request = json.loads(post_data.decode('utf-8'))
                query = request.get('query', '').strip()
                topk = request.get('topk', 10)
                conjunctive = request.get('conjunctive', False)
                mode = request.get('mode', 'bm25')  # 'bm25', 'dense', 'hybrid', 'compare'
                
                if not query:
                    self.send_json_response({'error': 'Empty query'}, 400)
                    return
                
                mode_str = "AND (conjunctive)" if conjunctive else "OR (disjunctive)"
                logger.info(f"Search request: '{query}', mode: {mode}, topk: {topk}, snippets: True")
                
                # Perform search
                results, timings = self.search(query, topk, conjunctive, mode)
                
                self.send_json_response({
                    'query': query,
                    'results': results,
                    'timings': timings,
                    'count': len(results)
                })
                
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                self.send_json_response({'error': str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()
    
    def clean_text(self, text: str) -> List[str]:
        """Clean and tokenize text for BM25 search"""
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return tokens
    
    def bm25_search(self, query: str, k: int = 10) -> Tuple[List[Tuple[str, float]], float]:
        """Perform BM25 search using Python index"""
        if self.bm25_index is None:
            raise Exception("BM25 index not loaded")
        
        start_time = time.time()
        query_terms = self.clean_text(query)
        results = self.bm25_index.search(query_terms, k)
        search_time = time.time() - start_time
        
        return results, search_time
    
    def bm25_search_cpp(self, query: str, k: int = 10, conjunctive: bool = False) -> Tuple[List[int], Dict[int, float], float]:
        """Perform BM25 search using C++ QueryProcessing executable"""
        start_time = time.time()
        query_terms = query.strip().split()
        if not query_terms:
            return [], {}, 0.0
        
        query_cmd = ['./QueryProcessing'] + query_terms + ['-k', str(k)]
        if conjunctive:
            query_cmd.append('-and')
        
        try:
            out = subprocess.check_output(query_cmd, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore')
            doc_pattern = re.compile(r'doc=(\d+)\s+score=([\d.]+)')
            matches = doc_pattern.findall(out)
            doc_ids = [int(m[0]) for m in matches]
            scores = {int(m[0]): float(m[1]) for m in matches}
            search_time = time.time() - start_time
            return doc_ids, scores, search_time
        except subprocess.CalledProcessError as e:
            logger.warning(f"C++ BM25 search failed: {e}, falling back to Python BM25")
            return self.bm25_search(query, k)
    
    def dense_search(self, query: str, k: int = 10) -> Tuple[List[Tuple[str, float]], float]:
        """Perform dense vector search using LM Studio embeddings"""
        if self.faiss_index is None or self.passage_ids is None:
            raise Exception("FAISS index or passage IDs not loaded")
        
        start_time = time.time()
        
        # Get query embedding from LM Studio
        query_embedding = self.get_query_embedding(query)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Convert results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.passage_ids):
                passage_id = self.passage_ids[idx]
                # Convert distance to similarity score
                similarity = float(1.0 / (1.0 + dist))
                results.append((passage_id, similarity))
        
        search_time = time.time() - start_time
        return results, search_time
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding from LM Studio API"""
        embed_endpoint = "http://127.0.0.1:1234/v1/embeddings"
        embed_model = "gaianet/text-embedding-all-minilm-l6-v2-embedding"
        
        payload = {
            "model": embed_model,
            "input": query
        }
        
        try:
            import requests
            response = requests.post(embed_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception:
            # Fallback to urllib if requests not available
            try:
                from urllib.request import Request, urlopen
                import json as _json
                req = Request(embed_endpoint, data=_json.dumps(payload).encode('utf-8'), 
                            headers={'Content-Type': 'application/json'})
                with urlopen(req, timeout=30) as r:
                    data = _json.load(r)
            except Exception as e:
                logger.warning(f"LM Studio API failed: {e}, falling back to hash-based approach")
                return self.get_fallback_embedding(query)
        
        # Extract embedding from response
        embedding = None
        if isinstance(data, dict):
            # OpenAI-like: { data: [ { embedding: [...] } ] }
            if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0 and 'embedding' in data['data'][0]:
                embedding = data['data'][0]['embedding']
            # LM Studio simple: { embedding: [...] } or { data: { embedding: [...] } }
            elif 'embedding' in data and isinstance(data['embedding'], list):
                embedding = data['embedding']
            elif 'data' in data and isinstance(data['data'], dict) and 'embedding' in data['data']:
                embedding = data['data']['embedding']
        
        if embedding is None:
            logger.warning("Could not extract embedding from LM Studio response, using fallback")
            return self.get_fallback_embedding(query)
        
        try:
            return np.array(embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logger.warning(f"Failed to parse embedding vector: {e}, using fallback")
            return self.get_fallback_embedding(query)
    
    def get_fallback_embedding(self, query: str) -> np.ndarray:
        """Fallback to hash-based embedding if LM Studio is unavailable"""
        if self.embeddings is not None:
            # Use query hash to select different embeddings for different queries
            query_hash = hash(query) % len(self.embeddings)
            query_embedding = self.embeddings[query_hash].reshape(1, -1)
            
            # Add some noise based on query to make results more diverse
            np.random.seed(hash(query) % 2**32)  # Use query as seed for reproducibility
            noise = np.random.normal(0, 0.1, query_embedding.shape)
            query_embedding = query_embedding + noise
            return query_embedding
        else:
            raise Exception("No embeddings available for query encoding")
    
    def hybrid_search(self, query: str, k: int = 10, candidate_k: int = 1000) -> Tuple[List[Tuple[str, float, Dict]], float]:
        """Perform hybrid search using BM25 as candidate generation and dense search as re-ranking"""
        start_time = time.time()
        
        # Step 1: Generate candidates using BM25
        bm25_results, bm25_time = self.bm25_search(query, candidate_k)
        candidate_doc_ids = [doc_id for doc_id, _ in bm25_results]
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # Step 2: Re-rank candidates using dense search
        reranked_results = self._rerank_candidates(query, candidate_doc_ids, k)
        
        # Step 3: Combine information for final results
        final_results = []
        for doc_id, dense_score in reranked_results:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            final_results.append((doc_id, dense_score, {
                'bm25_score': bm25_score,
                'dense_score': dense_score
            }))
        
        search_time = time.time() - start_time
        return final_results, search_time
    
    def _rerank_candidates(self, query: str, candidate_doc_ids: List[str], k: int) -> List[Tuple[str, float]]:
        """Re-rank candidate documents using dense vector similarity"""
        if not candidate_doc_ids or self.faiss_index is None or self.passage_ids is None:
            return []
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Get embeddings for candidate documents
        candidate_embeddings = []
        valid_doc_ids = []
        
        for doc_id in candidate_doc_ids:
            try:
                doc_index = self.passage_ids.index(doc_id)
                doc_embedding = self.faiss_index.reconstruct(doc_index)
                candidate_embeddings.append(doc_embedding)
                valid_doc_ids.append(doc_id)
            except (ValueError, IndexError):
                continue
        
        if not candidate_embeddings:
            return []
        
        # Convert to numpy array
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(candidate_norms, query_norm.T).flatten()
        
        # Sort by similarity and return top k
        similarity_scores = list(zip(valid_doc_ids, similarities))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return similarity_scores[:k]
    
    def generate_snippets(self, query: str, doc_id: str) -> List[str]:
        """Generate snippets using SnippetExtractor.cpp or fallback to simple extraction"""
        try:
            # First try SnippetExtractor.cpp if C++ index files exist
            snippet_extractor_path = '/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25/SnippetExtractor'
            store_dir = '/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25'
            offsets_file = os.path.join(store_dir, 'offsets.bin')
            
            if os.path.exists(snippet_extractor_path) and os.path.exists(offsets_file):
                # Run SnippetExtractor with correct format
                cmd = [
                    snippet_extractor_path,
                    '--store-dir', store_dir,
                    '--query', query,
                    '--docs', doc_id,
                    '--windowsz', '40',
                    '--per', '2'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Parse snippet output
                    snippets = []
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            try:
                                snippet_data = json.loads(line)
                                if snippet_data.get('docID') == doc_id:
                                    snippets.extend(snippet_data.get('snippets', []))
                            except json.JSONDecodeError:
                                continue
                    if snippets:
                        return snippets
            
            # Fallback to simple snippet generation using BM25 index
            if self.bm25_index and doc_id in self.bm25_index.doc_texts:
                text = self.bm25_index.doc_texts[doc_id]
                # Simple snippet: find query terms and extract context
                query_terms = query.lower().split()
                text_lower = text.lower()
                
                # Find the first occurrence of any query term
                best_pos = -1
                for term in query_terms:
                    pos = text_lower.find(term)
                    if pos != -1 and (best_pos == -1 or pos < best_pos):
                        best_pos = pos
                
                if best_pos != -1:
                    # Extract 200 characters around the match
                    start = max(0, best_pos - 100)
                    end = min(len(text), best_pos + 100)
                    snippet = text[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(text):
                        snippet = snippet + "..."
                    return [snippet]
                else:
                    # No query terms found, return first 200 chars
                    return [text[:200] + "..." if len(text) > 200 else text]
            
            # If document not found, return a generic snippet
            return [f"Document {doc_id} - Content not available for snippet generation"]
            
        except Exception as e:
            logger.warning(f"Snippet generation failed for doc {doc_id}: {e}")
            return [f"Document {doc_id} - Snippet generation error"]
    
    def search(self, query: str, topk: int, conjunctive: bool = False, mode: str = 'bm25') -> Tuple[List[Dict], Dict]:
        """Execute search based on mode"""
        timings = {
            'bm25_time': None,
            'dense_time': None,
            'snippet_time': None,
            'total_time': None
        }
        
        start_total = time.time()
        
        try:
            if mode == 'bm25':
                # Try C++ BM25 first, fallback to Python BM25
                try:
                    doc_ids, scores, search_time = self.bm25_search_cpp(query, topk, conjunctive)
                    timings['bm25_time'] = search_time
                    
                    # Format results
                    formatted_results = []
                    for doc_id in doc_ids:
                        snippets = self.generate_snippets(query, str(doc_id))
                        formatted_results.append({
                            'docID': str(doc_id),
                            'score': float(scores.get(doc_id, 0.0)),
                            'snippets': snippets
                        })
                    
                except Exception as e:
                    logger.warning(f"C++ BM25 failed: {e}, using Python BM25")
                    results, search_time = self.bm25_search(query, topk)
                    timings['bm25_time'] = search_time
                    
                    # Format results
                    formatted_results = []
                    for doc_id, score in results:
                        snippets = self.generate_snippets(query, str(doc_id))
                        formatted_results.append({
                            'docID': str(doc_id),
                            'score': score,
                            'snippets': snippets
                        })
                
            elif mode == 'dense':
                results, search_time = self.dense_search(query, topk)
                timings['dense_time'] = search_time
                
                # Format results
                formatted_results = []
                for doc_id, score in results:
                    snippets = self.generate_snippets(query, str(doc_id))
                    formatted_results.append({
                        'docID': str(doc_id),
                        'score': float(score),
                        'snippets': snippets
                    })
                
            elif mode == 'hybrid':
                results, search_time = self.hybrid_search(query, topk)
                timings['bm25_time'] = search_time  # Hybrid includes both
                timings['dense_time'] = search_time
                
                # Format results
                formatted_results = []
                for doc_id, score, info in results:
                    snippets = self.generate_snippets(query, str(doc_id))
                    formatted_results.append({
                        'docID': str(doc_id),
                        'score': float(score),
                        'snippets': snippets
                    })
                
            elif mode == 'compare':
                # Run both BM25 and dense search separately for comparison
                bm25_results, bm25_time = self.bm25_search(query, topk)
                dense_results, dense_time = self.dense_search(query, topk)
                
                timings['bm25_time'] = bm25_time
                timings['dense_time'] = dense_time
                
                # Combine results for comparison
                all_results = {}
                for doc_id, score in bm25_results:
                    all_results[doc_id] = {'bm25_score': score, 'dense_score': 0.0}
                
                for doc_id, score in dense_results:
                    if doc_id in all_results:
                        all_results[doc_id]['dense_score'] = score
                    else:
                        all_results[doc_id] = {'bm25_score': 0.0, 'dense_score': score}
                
                # Format results
                formatted_results = []
                for doc_id, scores in all_results.items():
                    snippets = self.generate_snippets(query, str(doc_id))
                    # Calculate combined score for display
                    combined_score = scores['bm25_score'] + scores['dense_score']
                    formatted_results.append({
                        'docID': str(doc_id),
                        'score': float(combined_score),
                        'bm25_score': float(scores['bm25_score']),
                        'dense_score': float(scores['dense_score']),
                        'snippets': snippets
                    })
                
            else:
                raise Exception(f"Unknown search mode: {mode}")
            
            timings['total_time'] = time.time() - start_total
            return formatted_results, timings
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise e
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        logger.info(f"[Server] {format % args}")

def run_server(port=8080):
    """Start the unified search server"""
    
    # Check if required files exist
    required_files = [
        'search_frontend.html'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, UnifiedSearchHandler)
    
    print("=" * 60)
    print("🔍 UNIFIED SEARCH SERVER")
    print("=" * 60)
    print(f"✓ Server running on http://localhost:{port}")
    print(f"✓ BM25 + Dense Vector + Hybrid Search")
    print(f"✓ C++ and Python implementations")
    print(f"✓ Open your browser and navigate to: http://localhost:{port}")
    print(f"✓ Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[Server] Shutting down...")
        httpd.shutdown()
        print("[Server] Server stopped.")

if __name__ == '__main__':
    run_server(port=8080)
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
import sys
import time
import logging
import pickle
import numpy as np
import faiss
import h5py
from urllib.parse import parse_qs
from typing import List, Dict, Any, Tuple

# Add bm25 directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bm25'))

# Try to import requests for LM Studio API calls
try:
    import requests
except ImportError:
    requests = None

# Import BM25Index class - try multiple approaches
try:
    from build_bm25_from_subset import BM25Index
except ImportError:
    try:
        import build_bm25_from_subset
        BM25Index = build_bm25_from_subset.BM25Index
    except:
        # Define a dummy class if import fails
        class BM25Index:
            pass
        logger = logging.getLogger(__name__)
        logger.warning("Could not import BM25Index, BM25 search may not work")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReuseAddrHTTPServer(HTTPServer):
    """HTTPServer with SO_REUSEADDR to avoid 'Address already in use' errors"""
    allow_reuse_address = True

# Global variables for search systems (shared across all handler instances)
_bm25_index = None
_faiss_index = None
_passage_ids = None
_embeddings = None
_systems_loaded = False

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle BM25Index class from different modules"""
    def find_class(self, module, name):
        if name == 'BM25Index':
            # Return the BM25Index class regardless of original module
            try:
                from build_bm25_from_subset import BM25Index
                return BM25Index
            except:
                pass
        return super().find_class(module, name)

def load_search_systems():
    """Load both BM25 and FAISS search systems once"""
    global _bm25_index, _faiss_index, _passage_ids, _embeddings, _systems_loaded
    
    if _systems_loaded:
        return
    
    try:
        # Load BM25 index with custom unpickler
        bm25_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/bm25/bm25_subset_index.pkl"
        if os.path.exists(bm25_path):
            logger.info("Loading BM25 index...")
            with open(bm25_path, 'rb') as f:
                _bm25_index = CustomUnpickler(f).load()
            logger.info(f"BM25 index loaded: {_bm25_index.total_docs} documents")
        else:
            logger.warning("BM25 index not found, BM25 search will be disabled")
        
        # Load FAISS HNSW index (prefer HNSW over IVF for better quality)
        hnsw_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/faiss_hnsw_index.bin"
        ivf_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/faiss_ivf_index.bin"
        ids_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/dense/passage_ids.pkl"
        
        # Prefer HNSW if available, fallback to IVF
        faiss_path = hnsw_path if os.path.exists(hnsw_path) else ivf_path
        
        if os.path.exists(faiss_path) and os.path.exists(ids_path):
            logger.info(f"Loading FAISS index from {os.path.basename(faiss_path)}...")
            _faiss_index = faiss.read_index(faiss_path)
            
            # Configure FAISS index parameters for optimal search
            if isinstance(_faiss_index, faiss.IndexHNSWFlat):
                # Set HNSW search parameters for better recall
                _faiss_index.hnsw.efSearch = 200  # Higher = better recall but slower
                logger.info(f"FAISS HNSW index loaded: {_faiss_index.ntotal} vectors")
                logger.info(f"  efSearch set to: {_faiss_index.hnsw.efSearch}")
            elif hasattr(_faiss_index, 'nprobe'):
                # Set IVF search parameters
                _faiss_index.nprobe = 10  # Number of clusters to search
                logger.info(f"FAISS IVF index loaded: {_faiss_index.ntotal} vectors")
                logger.info(f"  nprobe set to: {_faiss_index.nprobe}")
            else:
                logger.info(f"FAISS index loaded: {_faiss_index.ntotal} vectors")
            
            # Load passage IDs
            with open(ids_path, 'rb') as f:
                _passage_ids = pickle.load(f)
            logger.info(f"Passage IDs loaded: {len(_passage_ids)} passages")
        else:
            logger.warning("FAISS index not found, dense search will be disabled")
            logger.warning(f"  Checked: {hnsw_path}")
            logger.warning(f"  Checked: {ivf_path}")
        
        # Load embeddings for query encoding
        embeddings_path = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_passages_embeddings_subset.h5"
        if os.path.exists(embeddings_path):
            logger.info("Loading embeddings for query encoding...")
            with h5py.File(embeddings_path, 'r') as f:
                _embeddings = np.array(f['embedding']).astype(np.float32)
            logger.info(f"Embeddings loaded: {_embeddings.shape}")
        
        _systems_loaded = True
        
    except Exception as e:
        logger.error(f"Error loading search systems: {e}")

class UnifiedSearchHandler(BaseHTTPRequestHandler):
    
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
    
    def bm25_search(self, query: str, k: int = 10, conjunctive: bool = False) -> Tuple[List[Tuple[str, float]], float]:
        """Perform BM25 search using Python index"""
        if _bm25_index is None:
            raise Exception("BM25 index not loaded")
        
        start_time = time.time()
        query_terms = self.clean_text(query)
        results = _bm25_index.search(query_terms, k, conjunctive=conjunctive)
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
            # Convert Python BM25 results to C++ format
            results, search_time = self.bm25_search(query, k, conjunctive)
            doc_ids = [int(doc_id) for doc_id, _ in results]
            scores = {int(doc_id): score for doc_id, score in results}
            return doc_ids, scores, search_time
    
    def dense_search(self, query: str, k: int = 10) -> Tuple[List[Tuple[str, float]], float]:
        """Perform dense vector search using FAISS HNSW/IVF index"""
        if _faiss_index is None or _passage_ids is None:
            raise Exception("FAISS index or passage IDs not loaded")
        
        start_time = time.time()
        
        # Get query embedding from LM Studio
        query_embedding = self.get_query_embedding(query)
        
        # Ensure query embedding is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        distances, indices = _faiss_index.search(query_embedding, k)
        
        # Convert results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= 0 and idx < len(_passage_ids):
                passage_id = _passage_ids[idx]
                
                # Handle bytes to string conversion
                if isinstance(passage_id, bytes):
                    passage_id = passage_id.decode('utf-8')
                passage_id = str(passage_id)
                
                # Convert distance to similarity score
                # For L2 distance: smaller is better, convert to similarity
                # For Inner Product: higher (more negative distance) is better
                if isinstance(_faiss_index, faiss.IndexHNSWFlat):
                    # HNSW uses L2 distance by default
                    similarity = float(1.0 / (1.0 + dist))
                else:
                    # For inner product or other metrics
                    similarity = float(-dist)  # Negative distance as score
                
                results.append((passage_id, similarity))
        
        search_time = time.time() - start_time
        logger.debug(f"Dense search completed in {search_time:.3f}s, found {len(results)} results")
        return results, search_time
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding from LM Studio API"""
        embed_endpoint = "http://127.0.0.1:1234/v1/embeddings"
        embed_model = "second-state/text-embedding-all-minilm-l6-v2-embedding"
        
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
        if _embeddings is not None:
            # Use query hash to select different embeddings for different queries
            query_hash = hash(query) % len(_embeddings)
            query_embedding = _embeddings[query_hash].reshape(1, -1)
            
            # Add some noise based on query to make results more diverse
            np.random.seed(hash(query) % 2**32)  # Use query as seed for reproducibility
            noise = np.random.normal(0, 0.1, query_embedding.shape)
            query_embedding = query_embedding + noise
            return query_embedding
        else:
            raise Exception("No embeddings available for query encoding")
    
    def hybrid_search(self, query: str, k: int = 10, conjunctive: bool = False, candidate_k: int = 1000) -> Tuple[List[Tuple[str, float, Dict]], float]:
        """Perform hybrid search using BM25 as candidate generation and dense search as re-ranking
        
        Args:
            query: Search query
            k: Number of final results to return
            conjunctive: Whether to use AND mode for BM25 candidate generation
            candidate_k: Number of candidates to generate with BM25 before re-ranking
        """
        start_time = time.time()
        
        # Step 1: Generate candidates using BM25
        bm25_results, bm25_time = self.bm25_search(query, candidate_k, conjunctive)
        candidate_doc_ids = [doc_id for doc_id, _ in bm25_results]
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        logger.info(f"Hybrid search: BM25 generated {len(candidate_doc_ids)} candidates")
        
        # Step 2: Re-rank candidates using dense search
        reranked_results = self._rerank_candidates(query, candidate_doc_ids, k)
        
        logger.info(f"Hybrid search: Re-ranked to {len(reranked_results)} final results")
        
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
        if not candidate_doc_ids or _embeddings is None or _passage_ids is None:
            return []
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Create a mapping from doc_id to index for faster lookup
        passage_id_to_index = {doc_id: idx for idx, doc_id in enumerate(_passage_ids)}
        
        # Get embeddings for candidate documents
        candidate_embeddings = []
        valid_doc_ids = []
        
        for doc_id in candidate_doc_ids:
            if doc_id in passage_id_to_index:
                doc_index = passage_id_to_index[doc_id]
                try:
                    # Use embeddings instead of reconstructing from FAISS
                    doc_embedding = _embeddings[doc_index]
                    candidate_embeddings.append(doc_embedding)
                    valid_doc_ids.append(doc_id)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not get embedding for doc {doc_id}: {e}")
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
            if _bm25_index and doc_id in _bm25_index.doc_texts:
                text = _bm25_index.doc_texts[doc_id]
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
                    results, search_time = self.bm25_search(query, topk, conjunctive)
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
                results, search_time = self.hybrid_search(query, topk, conjunctive)
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
                bm25_results, bm25_time = self.bm25_search(query, topk, conjunctive)
                dense_results, dense_time = self.dense_search(query, topk)
                
                timings['bm25_time'] = bm25_time
                timings['dense_time'] = dense_time
                
                # Combine results for comparison
                all_results = {}
                for doc_id, score in bm25_results:
                    all_results[doc_id] = {'bm25': score, 'dense': 0.0}
                
                for doc_id, score in dense_results:
                    if doc_id in all_results:
                        all_results[doc_id]['dense'] = score
                    else:
                        all_results[doc_id] = {'bm25': 0.0, 'dense': score}
                
                # Format results - sort by combined score
                formatted_results = []
                for doc_id, scores in all_results.items():
                    snippets = self.generate_snippets(query, str(doc_id))
                    # Calculate combined score for sorting
                    combined_score = scores['bm25'] + scores['dense']
                    formatted_results.append({
                        'docID': str(doc_id),
                        'score': {
                            'bm25': float(scores['bm25']),
                            'dense': float(scores['dense'])
                        },
                        'snippets': snippets
                    })
                
                # Sort by combined score
                formatted_results.sort(key=lambda x: x['score']['bm25'] + x['score']['dense'], reverse=True)
                
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

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    try:
        # Find process using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    logger.info(f"Killing process {pid} on port {port}")
                    subprocess.run(['kill', '-9', pid], check=True)
                    time.sleep(0.5)  # Give it a moment to release the port
                except subprocess.CalledProcessError:
                    pass
    except Exception as e:
        logger.warning(f"Could not check/kill process on port {port}: {e}")

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
    
    # Kill any existing process on the port
    kill_process_on_port(port)
    
    # Load search systems before starting server
    print("=" * 60)
    print("🔍 UNIFIED SEARCH SERVER - INITIALIZING")
    print("=" * 60)
    load_search_systems()
    print()
    
    server_address = ('', port)
    
    # Use HTTPServer with socket reuse enabled
    httpd = ReuseAddrHTTPServer(server_address, UnifiedSearchHandler)
    
    print("=" * 60)
    print("🔍 UNIFIED SEARCH SERVER - READY")
    print("=" * 60)
    print(f"✓ Server running on http://localhost:{port}")
    print(f"✓ BM25 + Dense Vector + Hybrid Search")
    if _bm25_index:
        print(f"✓ BM25 Index: {_bm25_index.total_docs:,} documents loaded")
    if _faiss_index:
        print(f"✓ FAISS Index: {_faiss_index.ntotal:,} vectors loaded")
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
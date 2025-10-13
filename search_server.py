#!/usr/bin/env python3
"""
Genie Search Backend Server
Handles search requests and returns results with snippets
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import re
import os
from urllib.parse import parse_qs

class SearchHandler(BaseHTTPRequestHandler):
    
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
                
                if not query:
                    self.send_json_response({'error': 'Empty query'}, 400)
                    return
                
                mode = "AND (conjunctive)" if conjunctive else "OR (disjunctive)"
                print(f"\n[Search Request] Query: '{query}', Mode: {mode}, Top-K: {topk}")
                
                # Perform search
                results = self.search(query, topk, conjunctive)
                
                self.send_json_response({
                    'query': query,
                    'results': results,
                    'count': len(results)
                })
                
            except Exception as e:
                print(f"[Error] {str(e)}")
                self.send_json_response({'error': str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()
    
    def search(self, query, topk, conjunctive=False):
        """Execute search pipeline: QueryProcessing + SnippetExtractor"""
        
        # Step 1: Run QueryProcessing to get ranked documents
        mode_str = "conjunctive (AND)" if conjunctive else "disjunctive (OR)"
        print(f"[Step 1] Running QueryProcessing in {mode_str} mode...")
        try:
            # Split query into individual terms for proper processing
            query_terms = query.strip().split()
            if not query_terms:
                return []
            
            # Join terms with spaces for QueryProcessing
            query_for_search = ' '.join(query_terms)
            print(f"[Step 1] Query terms: {query_terms}")
            
            query_cmd = ['./QueryProcessing', query_for_search, '-k', str(topk)]
            if conjunctive:
                query_cmd.append('--and')
            
            query_output = subprocess.check_output(
                query_cmd, 
                stderr=subprocess.STDOUT
            ).decode('utf-8', errors='ignore')  # Ignore invalid UTF-8 bytes
            
            # Parse document IDs and scores from output
            # Format: " 1. doc=6233096 score=20.5415"
            doc_pattern = re.compile(r'doc=(\d+)\s+score=([\d.]+)')
            matches = doc_pattern.findall(query_output)
            
            if not matches:
                print("[Step 1] No results found")
                return []
            
            doc_ids = [int(m[0]) for m in matches]
            scores = {int(m[0]): float(m[1]) for m in matches}
            
            print(f"[Step 1] Found {len(doc_ids)} documents")
            
            # Step 2: Generate snippets for these documents
            print(f"[Step 2] Generating snippets...")
            doc_ids_str = ','.join(map(str, doc_ids))
            
            snippet_cmd = [
                './SnippetExtractor',
                '--store-dir', 'index_output',
                '--docs', doc_ids_str,
                '--query', query_for_search,  # Use the processed query for highlighting
                '--windowsz', '50',
                '--per', '2'
            ]
            
            snippet_output = subprocess.check_output(
                snippet_cmd,
                stderr=subprocess.STDOUT
            ).decode('utf-8', errors='ignore')  # Ignore invalid UTF-8 bytes
            
            # Parse JSON output from SnippetExtractor
            results = []
            for line in snippet_output.strip().split('\n'):
                if line.strip():
                    try:
                        snippet_data = json.loads(line)
                        doc_id = snippet_data['docID']
                        results.append({
                            'docID': doc_id,
                            'score': scores.get(doc_id, 0.0),
                            'snippets': snippet_data['snippets']
                        })
                    except json.JSONDecodeError:
                        print(f"[Warning] Failed to parse snippet line: {line[:100]}")
                        continue
            
            print(f"[Step 2] Generated snippets for {len(results)} documents")
            
            # Sort by original score order
            results.sort(key=lambda x: scores.get(x['docID'], 0), reverse=True)
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"[Error] Command failed: {e.output}")
            raise Exception(f"Search command failed: {e.output}")
        except FileNotFoundError as e:
            print(f"[Error] Command not found: {e}")
            raise Exception("Search executables not found. Please compile QueryProcessing and SnippetExtractor.")
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[Server] {format % args}")


def run_server(port=8080):
    """Start the search server"""
    
    # Check if required files exist
    if not os.path.exists('search_frontend.html'):
        print("Error: search_frontend.html not found!")
        return
    
    if not os.path.exists('QueryProcessing'):
        print("Error: QueryProcessing executable not found!")
        return
    
    if not os.path.exists('SnippetExtractor'):
        print("Error: SnippetExtractor executable not found!")
        return
    
    if not os.path.exists('index_output'):
        print("Error: index_output directory not found!")
        return
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, SearchHandler)
    
    print("=" * 60)
    print("🧞 GENIE SEARCH SERVER")
    print("=" * 60)
    print(f"✓ Server running on http://localhost:{port}")
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


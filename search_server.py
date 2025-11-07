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
import sys

IS_WINDOWS = sys.platform == "win32"
QUERY_PROC_EXEC = 'QueryProcessing.exe' if IS_WINDOWS else './QueryProcessing'
SNIPPET_EXEC = 'SnippetExtractor.exe' if IS_WINDOWS else './SnippetExtractor'


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
        
        mode_str = "conjunctive (AND)" if conjunctive else "disjunctive (OR)"
        print(f"[Step 1] Running QueryProcessing in {mode_str} mode...")
        try:
            query_terms = query.strip().split()
            if not query_terms:
                return []
            
            print(f"[Step 1] Query terms: {query_terms}")
            
            # --- MODIFICATION: Use the OS-aware executable name ---
            query_cmd = [QUERY_PROC_EXEC] + query_terms + ['-k', str(topk)]
            if conjunctive:
                query_cmd.append('-and')
            
            query_output = subprocess.check_output(
                query_cmd, 
                stderr=subprocess.STDOUT
            ).decode('utf-8', errors='ignore')
            
            doc_pattern = re.compile(r'doc=(\d+)\s+score=([\d.]+)')
            matches = doc_pattern.findall(query_output)
            
            if not matches:
                print("[Step 1] No results found")
                return []
            
            doc_ids = [int(m[0]) for m in matches]
            scores = {int(m[0]): float(m[1]) for m in matches}
            
            print(f"[Step 1] Found {len(doc_ids)} documents")
            
            print(f"[Step 2] Generating snippets...")
            doc_ids_str = ','.join(map(str, doc_ids))
            
            snippet_cmd = [
                SNIPPET_EXEC,
                '--store-dir', 'index_output',
                '--docs', doc_ids_str,
                '--query', query,
                '--windowsz', '50',
                '--per', '2'
            ]
            
            snippet_output = subprocess.check_output(
                snippet_cmd,
                stderr=subprocess.STDOUT
            ).decode('utf-8', errors='ignore')
            
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
            
            results.sort(key=lambda x: scores.get(x['docID'], 0), reverse=True)
            
            return results
            
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode('utf-8', errors='ignore') if e.output else "No output from command."
            print(f"[Error] Command failed with output:\n---\n{error_output}\n---")
            raise Exception(f"Search command failed.")
        except FileNotFoundError as e:
            print(f"[Error] Command not found: {e}. Ensure executables are in the correct path and compiled for your OS.")
            raise Exception("Search executables not found. Please compile them for your operating system.")
    
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        print(f"[Server] {format % args}")


def run_server(port=8080):
    """Start the search server"""
    
    if not os.path.exists('search_frontend.html'):
        print("Error: search_frontend.html not found!")
        return
    
    # --- MODIFICATION: Use OS-aware executable names for checks ---
    if not os.path.exists(QUERY_PROC_EXEC):
        print(f"Error: {QUERY_PROC_EXEC} executable not found!")
        return
    
    if not os.path.exists(SNIPPET_EXEC):
        print(f"Error: {SNIPPET_EXEC} executable not found!")
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
#!/bin/bash
# Genie Search Server Startup Script

cd "$(dirname "$0")"

echo "Starting Genie Search Server..."
echo "================================"
echo ""

# Activate virtual environment
source search_engine_env/bin/activate

# Start the server
python3 search_server.py

echo ""
echo "Server stopped."

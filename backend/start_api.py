#!/usr/bin/env python3
"""
Startup script for the AI Photo Search API Server

This script provides easy commands to start the FastAPI server with different configurations.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies found")
        return True
    except ImportError:
        print("❌ FastAPI dependencies missing. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"], check=True)
            print("✅ FastAPI dependencies installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def start_server(host="0.0.0.0", port=8000, reload=True, log_level="info"):
    """Start the FastAPI server"""
    
    # Check if the API server file exists
    if not os.path.exists("api_server.py"):
        print("❌ api_server.py not found in current directory")
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    print(f"🚀 Starting AI Photo Search API Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📊 Access API docs at: http://{host}:{port}/docs")
    print(f"🌐 Access API at: http://{host}:{port}")
    print("-" * 50)
    
    try:
        # Start the server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "api_server:app",
            "--host", host,
            "--port", str(port),
            "--log-level", log_level
        ]
        
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AI Photo Search API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    parser.add_argument("--local", action="store_true", help="Run on localhost only (127.0.0.1)")
    
    args = parser.parse_args()
    
    # Adjust host for local-only mode
    if args.local:
        args.host = "127.0.0.1"
    
    # Start the server
    success = start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

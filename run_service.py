"""
run_service.py — Service Launcher for FedTrust-Credit Web Dashboard & REST API
Usage:
    python run_service.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import os
import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

def main():
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", 8000))

    parser = argparse.ArgumentParser(description="Start FedTrust-Credit Service")
    parser.add_argument("--host", default=default_host, help=f"Host interface (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print("=" * 70)
    print(">> Starting FedTrust-Credit Web Dashboard & REST API Service")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   API Documentation: http://{args.host}:{args.port}/docs")
    print("=" * 70)

    uvicorn.run("service:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()

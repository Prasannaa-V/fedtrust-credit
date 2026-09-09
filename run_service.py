"""
run_service.py — Service Launcher for FedTrust-Credit Web Dashboard & REST API
Usage:
    python run_service.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description="Start FedTrust-Credit Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
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

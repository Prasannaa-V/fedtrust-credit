#!/usr/bin/env python3
"""
FedTrust-Credit — Remote Bank Client Daemon
Course: CLOUD COMPUTING (BITE412L)

This script runs on a secondary cloud instance (e.g., AWS EC2 Instance 2)
to simulate an isolated remote banking institution (Bank 3 - Subprime Portfolio).

Features:
- Zero external dependencies: Uses standard library (urllib, json, socket, platform, time).
- Discovers public IP automatically via AWS checkip service.
- Sends periodic heartbeats with hardware & portfolio telemetry to the Central Aggregator.
- Computes round-trip latency to prove true distributed cross-instance network traffic.
"""

import argparse
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def get_public_ip() -> str:
    """Discovers external public IP using lightweight public lookup services."""
    services = [
        "http://checkip.amazonaws.com",
        "http://api.ipify.org",
        "http://ifconfig.me/ip",
    ]
    for url in services:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                ip = resp.read().decode("utf-8").strip()
                if ip:
                    return ip
        except Exception:
            continue
    # Fallback to local hostname resolution
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def get_system_telemetry() -> Dict[str, Any]:
    """Collects system stats using only standard library."""
    load_1m = 0.0
    try:
        if hasattr(os, "getloadavg"):
            load_1m = round(os.getloadavg()[0], 2)
    except Exception:
        pass

    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "load_1m": load_1m,
    }


def send_heartbeat(
    aggregator_url: str,
    node_id: str,
    portfolio: str,
    public_ip: str,
    telemetry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Sends JSON heartbeat payload to Aggregator endpoint."""
    endpoint = f"{aggregator_url.rstrip('/')}/api/nodes/heartbeat"
    payload = {
        "node_id": node_id,
        "role": "remote_bank_client",
        "portfolio": portfolio,
        "ip_address": public_ip,
        "hostname": telemetry.get("hostname"),
        "status": "online",
        "metadata": {
            "os": telemetry.get("os"),
            "arch": telemetry.get("architecture"),
            "load_1m": telemetry.get("load_1m"),
            "timestamp": time.time(),
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "FedTrust-RemoteClient/1.0"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=5) as resp:
        rtt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        resp_data = json.loads(resp.read().decode("utf-8"))
        resp_data["_rtt_ms"] = rtt_ms
        return resp_data


def main():
    parser = argparse.ArgumentParser(
        description="FedTrust-Credit Remote Bank Client Node (AWS EC2 Instance 2)"
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="http://13.218.90.189:8000",
        help="Aggregator Base URL (e.g., http://13.218.90.189:8000)",
    )
    parser.add_argument(
        "--bank-id",
        type=str,
        default="bank_3",
        help="Bank Identifier (e.g. bank_3)",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default="Subprime Portfolio (Grades E–G)",
        help="Portfolio description for this banking node",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Heartbeat interval in seconds (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" FedTrust-Credit — Remote Bank Client Daemon (EC2 Node 2)")
    print(" Course: CLOUD COMPUTING (BITE412L) — Distributed FL Cluster")
    print("=" * 70)
    print(f"[*] Aggregator Target : {args.aggregator}")
    print(f"[*] Bank Identifier   : {args.bank_id}")
    print(f"[*] Portfolio         : {args.portfolio}")
    print(f"[*] Heartbeat Cadence : Every {args.interval}s")
    print("[*] Detecting public IP address...")
    public_ip = get_public_ip()
    print(f"[*] Node Public IPv4  : {public_ip}")
    print("[*] Starting heartbeat loop (Press Ctrl+C to terminate)...\n")

    seq = 0
    telemetry = get_system_telemetry()

    while True:
        seq += 1
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            res = send_heartbeat(
                aggregator_url=args.aggregator,
                node_id=args.bank_id,
                portfolio=args.portfolio,
                public_ip=public_ip,
                telemetry=telemetry,
            )
            rtt = res.get("_rtt_ms", 0.0) if res else 0.0
            print(
                f"[{curr_time}] [Beat #{seq:04d}] ACK from Aggregator | "
                f"RTT: {rtt:5.1f} ms | Node IP: {public_ip} | Status: ONLINE"
            )
        except urllib.error.URLError as e:
            print(
                f"[{curr_time}] [Beat #{seq:04d}] Connection failed to {args.aggregator}: {e.reason} "
                f"(Check Security Group / Port 8000)"
            )
        except Exception as e:
            print(f"[{curr_time}] [Beat #{seq:04d}] Error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Remote client stopped.")
        sys.exit(0)

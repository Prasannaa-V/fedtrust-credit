"""
deploy_cloudwatch_dashboard.py — Deploy/refresh CloudWatch dashboard (Task 4)
Uses boto3 cloudwatch.put_dashboard with the same 4-widget layout as
terraform/aws_monitoring.tf. Dry-run safe when AWS creds are missing.
"""

import argparse
import json
import os
import sys

WIDGETS = [
    {"title": "DefaultProbability (per prediction)",
     "metric": "DefaultProbability", "period": 60, "stat": "Average"},
    {"title": "ExplanationConsistencyScore (per round)",
     "metric": "ExplanationConsistencyScore", "period": 300, "stat": "Average"},
    {"title": "RoundLatencySeconds",
     "metric": "RoundLatencySeconds", "period": 300, "stat": "Average"},
]

NAMESPACE = "FedTrustCredit/FL"


def build_body(region):
    metric_widgets = []
    for i, w in enumerate(WIDGETS):
        metric_widgets.append({
            "type": "metric", "x": (i % 2) * 12, "y": (i // 2) * 6,
            "width": 12, "height": 6,
            "properties": {
                "title": w["title"],
                "metrics": [[NAMESPACE, w["metric"]]],
                "period": w["period"], "stat": w["stat"], "region": region,
            },
        })
    metric_widgets.append({
        "type": "alarm", "x": 12, "y": 6, "width": 12, "height": 6,
        "properties": {
            "title": "Alarm states (consistency drift + latency)",
            "alarms": [
                f"arn:aws:cloudwatch:{region}:*:alarm:fedtrust-consistency-drop-warning",
                f"arn:aws:cloudwatch:{region}:*:alarm:fedtrust-round-latency-high",
            ],
        },
    })
    return json.dumps({"widgets": metric_widgets})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard", default="fedtrust-credit")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    body = build_body(args.region)
    if args.dry_run:
        print(f"[dashboard] dry-run: {args.dashboard} in {args.region}")
        print(body[:500] + "...")
        return
    try:
        import boto3
    except ImportError:
        print("[dashboard] boto3 not installed; nothing deployed.")
        sys.exit(2)
    cw = boto3.client("cloudwatch", region_name=args.region)
    resp = cw.put_dashboard(DashboardName=args.dashboard, DashboardBody=body)
    print(f"[dashboard] deployed {args.dashboard}: {resp}")


if __name__ == "__main__":
    main()

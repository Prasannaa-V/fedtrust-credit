# AWS CloudWatch Screenshot Runbook — closes 9.1 (~10 min, credentialed env)

> Everything except the console screenshot is already done and tested:
> Terraform dashboard (`terraform/aws_monitoring.tf`), deploy script
> (`src/deploy_cloudwatch_dashboard.py`), latency telemetry
> (`APILatencySeconds`), local snapshot
> (`results/plots/cloudwatch_monitoring_snapshot.png`).
> This runbook produces the one artifact that cannot be generated without
> AWS credentials: a live CloudWatch console screenshot.

## 0. Prerequisites (2 min)

- AWS CLI + Terraform installed; credentials with `cloudwatch:PutDashboard`,
  `cloudwatch:PutMetricData`, `sns:Publish` (or an EC2 instance profile).
- Service running: `.venv/bin/uvicorn src.service:app --host 0.0.0.0 --port 8000`
- Env: `AWS_DEFAULT_REGION=us-east-1`, `FEDTRUST_S3_BUCKET`, `FEDTRUST_SNS_TOPIC_ARN`.

## 1. Apply monitoring stack (2 min)

```bash
cd terraform
terraform init
terraform apply -target=aws_cloudwatch_log_group.fedtrust_logs \
  -target=aws_cloudwatch_dashboard.fedtrust_dashboard
```

Expected: dashboard `fedtrust-credit` with 4 widgets (DefaultProbability,
ConsistencyScore, APILatencySeconds p95 bar histogram, alarm panel).

## 2. Push the dashboard definition (1 min)

```bash
.venv/bin/python src/deploy_cloudwatch_dashboard.py --region us-east-1
```

Expected: `put_dashboard` success response naming `fedtrust-credit`.

## 3. Generate live traffic (3 min)

```bash
for i in $(seq 1 12); do
  curl -s -X POST localhost:8000/api/predict/risk \
    -H 'Content-Type: application/json' \
    -d '{"int_rate": '$(python3 -c "print(8+$i*1.5)")'}' > /dev/null
  curl -s -X POST localhost:8000/api/federated/simulate-round \
    -H 'Content-Type: application/json' \
    -d '{"consistency_gain": 1.0}' > /dev/null
  sleep 5
done
```

Each call emits `DefaultProbability` / `ExplanationConsistencyScore` /
`APILatencySeconds` to `FedTrustCredit/FL`.

## 4. Capture + embed (2 min)

1. Open AWS Console → CloudWatch → Dashboards → `fedtrust-credit`.
2. Wait until all 4 widgets show datapoints; screenshot the full grid.
3. Save as `docs/aws_cloudwatch_dashboard.png` (repo root `docs/`).
4. Reference it from `docs/final_project_report_100pct.md` §6 next to the
   local snapshot, captioned with timestamp + region.
5. Commit the PNG (request an exception to image-size policy if needed).

## 5. Verify

- [ ] All 4 widgets render datapoints (no "no data" panels).
- [ ] Alarm panel shows both alarms in OK (or expected) state.
- [ ] PNG embedded in §6 with capture timestamp.

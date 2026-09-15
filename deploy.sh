#!/usr/bin/env bash
# =============================================================================
# deploy.sh — FedTrust-Credit EC2 Deployment Script
# Usage: bash deploy.sh [SNS_TOPIC_ARN]
# Example: bash deploy.sh arn:aws:sns:us-east-1:123456789012:fedtrust-risk-alerts
# =============================================================================

set -e

CONTAINER_NAME="fedtrust-dashboard"
IMAGE_NAME="fedtrust-credit"
S3_BUCKET="${FEDTRUST_S3_BUCKET:-fedtrust-models}"
SNS_ARN="${1:-${FEDTRUST_SNS_TOPIC_ARN:-}}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo ""
echo "======================================================================"
echo "  FedTrust-Credit — EC2 Deploy"
echo "  S3 Bucket : $S3_BUCKET"
echo "  Region    : $REGION"
echo "  SNS ARN   : ${SNS_ARN:-NOT SET}"
echo "======================================================================"
echo ""

# ── 1. Pull latest code ────────────────────────────────────────────────────
echo "[1/6] Pulling latest code from GitHub..."
git pull --ff-only

# ── 2. Stop & remove old container ────────────────────────────────────────
echo "[2/6] Stopping old container (if running)..."
sudo docker stop "$CONTAINER_NAME" 2>/dev/null || true
sudo docker rm   "$CONTAINER_NAME" 2>/dev/null || true

# ── 3. Start new container from existing image ─────────────────────────────
echo "[3/6] Starting container from image: $IMAGE_NAME ..."
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 8000:8000 \
  -e FEDTRUST_S3_BUCKET="$S3_BUCKET" \
  -e FEDTRUST_SNS_TOPIC_ARN="$SNS_ARN" \
  -e AWS_DEFAULT_REGION="$REGION" \
  "$IMAGE_NAME"

# ── 4. Inject boto3 + latest source files ─────────────────────────────────
echo "[4/6] Installing boto3 into container..."
sudo docker exec "$CONTAINER_NAME" pip install boto3 -q

echo "[5/6] Copying latest source files into container..."
sudo docker cp src/service.py        "$CONTAINER_NAME":/app/src/service.py
sudo docker cp src/aws_integrations.py "$CONTAINER_NAME":/app/src/aws_integrations.py

# ── 5. Restart to load new files ──────────────────────────────────────────
echo "[6/6] Restarting service to load updated files..."
sudo docker restart "$CONTAINER_NAME"

# ── 6. Health check ───────────────────────────────────────────────────────
echo ""
echo "Waiting for service to come up..."
for i in {1..12}; do
    sleep 2
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "✅ Service is up! (health check passed in ${i}x2s)"
        break
    fi
    echo "   ... waiting ($i/12)"
done

# ── 7. Print results ──────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  AWS Status:"
echo "======================================================================"
curl -s http://localhost:8000/api/aws/status | python3 -m json.tool 2>/dev/null || \
    echo "  ⚠ Service not responding yet — try: curl http://localhost:8000/api/aws/status"

echo ""
echo "======================================================================"
echo "  To test S3 upload:"
echo "  curl -X POST http://localhost:8000/api/aws/s3/upload \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"bucket_name\": \"$S3_BUCKET\"}' | python3 -m json.tool"
echo "======================================================================"

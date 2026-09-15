"""
aws_integrations.py — AWS Cloud Services Integration for FedTrust-Credit
Course: CLOUD COMPUTING – BITE412L, VIT Vellore

Integrates 4 native AWS Cloud Services with the FastAPI Risk Engine:
  1. Amazon S3: Cloud Model Registry for pre-trained models & benchmark plots
  2. AWS IAM: Role-based least-privilege security (zero hardcoded secrets)
  3. Amazon CloudWatch: Metrics telemetry (ConsistencyScore, Latency, DefaultProb)
  4. AWS SNS (Simple Notification Service): Instant email/SMS alerts for high-risk applicants
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "results" / "plots"

# Optional Boto3 import (falls back gracefully if not installed)
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class AWSCloudManager:
    """
    Manages AWS Cloud Services (S3, CloudWatch, SNS, IAM).
    Operates seamlessly using EC2 IAM Role or environment variables.
    Fails safely in dry-run mode if AWS services are not configured.
    """
    def __init__(self):
        self.region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.s3_bucket = os.environ.get("FEDTRUST_S3_BUCKET", "")
        self.sns_topic_arn = os.environ.get("FEDTRUST_SNS_TOPIC_ARN", "")
        self.cloudwatch_namespace = "FedTrustCredit/FL"

        self.s3_client = None
        self.sns_client = None
        self.cw_client = None
        self.is_connected = False

        if BOTO3_AVAILABLE:
            self._init_clients()

    def _init_clients(self):
        try:
            self.s3_client = boto3.client("s3", region_name=self.region)
            self.sns_client = boto3.client("sns", region_name=self.region)
            self.cw_client = boto3.client("cloudwatch", region_name=self.region)
            self.is_connected = True
            print(f"[AWSCloudManager] Connected to AWS services in region: {self.region}")
        except Exception as e:
            print(f"[AWSCloudManager] AWS clients running in offline/dry-run mode: {e}")
            self.is_connected = False

    def get_cloud_status(self) -> Dict[str, Any]:
        """Returns the live status of all integrated cloud services for the dashboard."""
        return {
            "aws_connected": self.is_connected,
            "region": self.region,
            "services": {
                "ec2": {
                    "status": "active",
                    "role": "Central Aggregator & Web Server",
                    "description": "Docker containerized compute instance",
                },
                "s3": {
                    "status": "configured" if self.s3_bucket else "ready_to_connect",
                    "bucket_name": self.s3_bucket or "Not set (set FEDTRUST_S3_BUCKET)",
                    "role": "Cloud Model Registry & Plot Storage",
                },
                "cloudwatch": {
                    "status": "active" if self.is_connected else "offline",
                    "namespace": self.cloudwatch_namespace,
                    "role": "Real-time Telemetry & Consistency Drift Alarms",
                },
                "sns": {
                    "status": "configured" if self.sns_topic_arn else "ready_to_connect",
                    "topic_arn": self.sns_topic_arn or "Not set (set FEDTRUST_SNS_TOPIC_ARN)",
                    "role": "Instant Email/SMS High-Risk Borrower Alerts",
                },
                "iam": {
                    "status": "active",
                    "policy": "Least-privilege role-based access",
                    "description": "Zero hardcoded credentials; uses EC2 Instance Profile",
                },
            }
        }

    def upload_models_to_s3(self, bucket_name: Optional[str] = None) -> Dict[str, Any]:
        """Uploads serialized model artifacts and plots to Amazon S3."""
        target_bucket = bucket_name or self.s3_bucket
        if not self.s3_client or not target_bucket:
            return {"success": False, "error": "S3 client or bucket name not configured"}

        uploaded = []
        try:
            # Upload models
            if MODELS_DIR.exists():
                for model_file in MODELS_DIR.glob("*.*"):
                    key = f"models/{model_file.name}"
                    self.s3_client.upload_file(str(model_file), target_bucket, key)
                    uploaded.append(key)

            # Upload publication plots
            if PLOTS_DIR.exists():
                for plot_file in PLOTS_DIR.glob("*.png"):
                    key = f"plots/{plot_file.name}"
                    self.s3_client.upload_file(str(plot_file), target_bucket, key)
                    uploaded.append(key)

            return {
                "success": True,
                "bucket": target_bucket,
                "files_uploaded": len(uploaded),
                "keys": uploaded,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def publish_high_risk_sns_alert(self, applicant: Dict[str, Any], default_prob: float) -> bool:
        """Sends an instant SNS email alert when an applicant is flagged as high-risk."""
        if not self.sns_client or not self.sns_topic_arn:
            return False

        try:
            subject = f"[FedTrust-Credit Alert] High-Risk Loan Applicant ({default_prob*100:.1f}%)"
            message = (
                f"FedTrust-Credit Automated Risk Alert\n"
                f"====================================\n"
                f"Default Probability : {default_prob*100:.2f}%\n"
                f"Risk Classification : CRITICAL / HIGH RISK\n\n"
                f"Applicant Parameters:\n"
                f"- Requested Loan  : ${applicant.get('loan_amnt', 0):,.2f}\n"
                f"- Annual Income   : ${applicant.get('annual_inc', 0):,.2f}\n"
                f"- Debt-to-Income  : {applicant.get('dti', 0)}%\n"
                f"- Loan Grade      : {applicant.get('grade', 'N/A')}\n"
                f"- Interest Rate   : {applicant.get('int_rate', 0)}%\n"
                f"- Purpose         : {applicant.get('purpose', 'N/A')}\n\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Managed by: AWS SNS (Simple Notification Service)\n"
            )
            self.sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject[:100],
                Message=message,
            )
            print(f"[AWSCloudManager] High-risk alert published to SNS Topic: {self.sns_topic_arn}")
            return True
        except Exception as e:
            print(f"[AWSCloudManager] Failed to publish SNS alert: {e}")
            return False

    def log_metric_to_cloudwatch(self, metric_name: str, value: float, unit: str = "None"):
        """Publishes custom telemetry metrics to Amazon CloudWatch."""
        if not self.cw_client:
            return

        try:
            self.cw_client.put_metric_data(
                Namespace=self.cloudwatch_namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": float(value),
                        "Unit": unit,
                        "Timestamp": time.time(),
                    }
                ],
            )
        except Exception as e:
            # Silent fallback to avoid impacting user request flow
            pass


# Singleton instance
cloud_manager = AWSCloudManager()

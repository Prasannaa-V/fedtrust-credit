# ==============================================================================
# AWS CloudWatch Monitoring & Metrics
# Fulfills Section 3.5: Amazon CloudWatch training-round metrics & latency tracking
# ==============================================================================

# CloudWatch Log Group for Federated Training Logs
resource "aws_cloudwatch_log_group" "fedtrust_logs" {
  name              = "/fedtrust-credit/training"
  retention_in_days = 30
  tags = {
    Name = "fedtrust-cloudwatch-log-group"
  }
}

# Metric Filter for Explanation Consistency Score
resource "aws_cloudwatch_log_metric_filter" "consistency_score_filter" {
  name           = "ExplanationConsistencyScore"
  pattern        = "[timestamp, round, consistency_score, ...]"
  log_group_name = aws_cloudwatch_log_group.fedtrust_logs.name

  metric_transformation {
    name          = "ExplanationConsistencyScore"
    namespace     = "FedTrustCredit/FL"
    value         = "$consistency_score"
    default_value = 0.0
  }
}

# Metric Alarm: Consistency Drift Alarm
resource "aws_cloudwatch_metric_alarm" "consistency_alarm" {
  count               = var.enable_cloudwatch_alarms ? 1 : 0
  alarm_name          = "fedtrust-consistency-drop-warning"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ExplanationConsistencyScore"
  namespace           = "FedTrustCredit/FL"
  period              = 300
  statistic           = "Average"
  threshold           = 0.70 # Warn if mean pairwise consistency drops below 0.70
  alarm_description   = "Triggered when federated client explanation consensus drops below 70%"
  treat_missing_data  = "notBreaching"
}

# Metric Alarm: Aggregator Training Round Latency
resource "aws_cloudwatch_metric_alarm" "round_latency_alarm" {
  count               = var.enable_cloudwatch_alarms ? 1 : 0
  alarm_name          = "fedtrust-round-latency-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RoundLatencySeconds"
  namespace           = "FedTrustCredit/FL"
  period              = 300
  statistic           = "Average"
  threshold           = 120 # Alert if a federated communication round exceeds 2 minutes
  alarm_description   = "Triggered when federated training round takes longer than 120 seconds"
  treat_missing_data  = "notBreaching"
}

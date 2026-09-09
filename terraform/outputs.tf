# ==============================================================================
# FedTrust-Credit — Multi-Cloud Outputs
# Outputs connectivity information, public endpoints, and resource identifiers
# ==============================================================================

# ── Aggregator Server & Web UI ────────────────────────────────────────────────
output "aggregator_public_ip" {
  description = "Public IP of the Federated Aggregator Server"
  value       = aws_instance.aggregator.public_ip
}

output "aggregator_dashboard_url" {
  description = "URL for the Live Interactive Credit Risk Assessment Web Dashboard"
  value       = "http://${aws_instance.aggregator.public_ip}:8000"
}

output "aggregator_swagger_docs" {
  description = "OpenAPI / Swagger REST documentation URL"
  value       = "http://${aws_instance.aggregator.public_ip}:8000/docs"
}

output "flower_fl_grpc_endpoint" {
  description = "Flower FL Aggregation gRPC address for client connections"
  value       = "${aws_instance.aggregator.public_ip}:8080"
}

# ── Client Compute Nodes ──────────────────────────────────────────────────────
output "aws_bank_client_1_ip" {
  description = "Public IP of Bank Client 1 (AWS VPC 1, Grades A-B)"
  value       = aws_instance.bank_client_1.public_ip
}

output "aws_bank_client_2_ip" {
  description = "Public IP of Bank Client 2 (AWS VPC 2, Grades C-D)"
  value       = aws_instance.bank_client_2.public_ip
}

output "azure_bank_client_3_ip" {
  description = "Public IP of Bank Client 3 (Azure VM, Grades E-G)"
  value       = azurerm_public_ip.bank3_pip.ip_address
}

# ── Isolated Storage ──────────────────────────────────────────────────────────
output "bank_1_s3_bucket" {
  description = "AWS S3 Encrypted Bucket for Bank Client 1"
  value       = aws_s3_bucket.bank1_bucket.bucket
}

output "bank_2_s3_bucket" {
  description = "AWS S3 Encrypted Bucket for Bank Client 2"
  value       = aws_s3_bucket.bank2_bucket.bucket
}

output "bank_3_azure_storage_account" {
  description = "Azure Storage Account for Bank Client 3"
  value       = azurerm_storage_account.bank3_storage.name
}

# ── Monitoring ────────────────────────────────────────────────────────────────
output "cloudwatch_log_group" {
  description = "Amazon CloudWatch Log Group for federated training metrics"
  value       = aws_cloudwatch_log_group.fedtrust_logs.name
}

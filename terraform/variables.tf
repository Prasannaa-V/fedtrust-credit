# ==============================================================================
# FedTrust-Credit — Terraform Variables
# Course: CLOUD COMPUTING – BITE412L
# ==============================================================================

variable "aws_region" {
  description = "AWS Primary Region for Aggregator, Bank 1, and Bank 2"
  type        = string
  default     = "us-east-1"
}

variable "azure_region" {
  description = "Azure Region for Bank 3 (geographically aligned with AWS us-east-1)"
  type        = string
  default     = "eastus"
}

variable "aws_instance_type" {
  description = "AWS EC2 instance type for Aggregator and Client nodes"
  type        = string
  default     = "t3.medium"
}

variable "azure_vm_size" {
  description = "Azure VM size for Bank Client 3"
  type        = string
  default     = "Standard_B2s"
}

variable "admin_username" {
  description = "Administrator username for cloud instances"
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key" {
  description = "Public SSH key for secure EC2 and Azure VM access"
  type        = string
  default     = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyFedTrustCredit"
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch metric alarms for federated training latency"
  type        = bool
  default     = true
}

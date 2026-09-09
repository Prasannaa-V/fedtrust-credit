# ==============================================================================
# FedTrust-Credit — Multi-Cloud Terraform Infrastructure
# Course: CLOUD COMPUTING – BITE412L
# Architecture:
#   - AWS us-east-1: Aggregator Node + Bank 1 VPC + Bank 2 VPC + S3 + CloudWatch
#   - Microsoft Azure East US: Bank 3 Virtual Machine + Azure Blob Storage
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "FedTrust-Credit"
      Course      = "BITE412L-Cloud-Computing"
      Environment = "Production-Evaluation"
      ManagedBy   = "Terraform"
    }
  }
}

# Azure Provider Configuration
provider "azurerm" {
  features {}
}

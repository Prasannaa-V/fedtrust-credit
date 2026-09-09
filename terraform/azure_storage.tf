# ==============================================================================
# Microsoft Azure Storage — Blob Storage for Bank Client 3 Data Isolation
# Fulfills Section 3.5: Azure Blob Storage for Azure-hosted client
# ==============================================================================

# Random string for Azure storage account name (must be globally unique and alphanumeric)
resource "random_string" "storage_suffix" {
  length  = 6
  special = false
  upper   = false
}

# Azure Storage Account
resource "azurerm_storage_account" "bank3_storage" {
  name                     = "stfedtrust${random_string.storage_suffix.result}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  tags = {
    Name   = "fedtrust-bank3-storage"
    Client = "bank_3"
  }
}

# Azure Blob Storage Container
resource "azurerm_storage_container" "bank3_container" {
  name                  = "bank3data"
  storage_account_name  = azurerm_storage_account.bank3_storage.name
  container_access_type = "private"
}

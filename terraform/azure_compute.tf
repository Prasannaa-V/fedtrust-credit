# ==============================================================================
# Microsoft Azure Compute — Bank Client 3 Virtual Machine
# Fulfills Section 3.5 & 4.1: Bank Client 3 on Azure VM in East US region
# ==============================================================================

# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = "rg-fedtrust-credit"
  location = var.azure_region
  tags = {
    Project     = "FedTrust-Credit"
    Course      = "BITE412L-Cloud-Computing"
    Environment = "Production-Evaluation"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "bank3_vnet" {
  name                = "vnet-fedtrust-bank3"
  address_space       = ["10.3.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

# Subnet
resource "azurerm_subnet" "bank3_subnet" {
  name                 = "snet-bank3"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.bank3_vnet.name
  address_prefixes     = ["10.3.1.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "bank3_nsg" {
  name                = "nsg-fedtrust-bank3"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Public IP for Bank Client 3
resource "azurerm_public_ip" "bank3_pip" {
  name                = "pip-fedtrust-bank3"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

# Network Interface
resource "azurerm_network_interface" "bank3_nic" {
  name                = "nic-fedtrust-bank3"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "ipconfig-bank3"
    subnet_id                     = azurerm_subnet.bank3_subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.bank3_pip.id
  }
}

# Associate NSG with NIC
resource "azurerm_network_interface_security_group_association" "bank3_assoc" {
  network_interface_id      = azurerm_network_interface.bank3_nic.id
  network_security_group_id = azurerm_network_security_group.bank3_nsg.id
}

# Linux Virtual Machine: Bank Client 3
resource "azurerm_linux_virtual_machine" "bank_client_3" {
  name                  = "vm-fedtrust-bank3"
  resource_group_name   = azurerm_resource_group.rg.name
  location              = azurerm_resource_group.rg.location
  size                  = var.azure_vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.bank3_nic.id]

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = 30
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  tags = {
    Name = "fedtrust-bank-client-3"
    Role = "Bank-Client-3-Grades-E-G-Azure"
  }
}

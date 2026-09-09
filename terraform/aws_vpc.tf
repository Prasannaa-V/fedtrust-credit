# ==============================================================================
# AWS Network Isolation — Dual Bank VPCs & Central Aggregator VPC
# Fulfills Section 3.5: Per-Client Boundary Network Isolation
# ==============================================================================

# ── 1. Aggregator VPC ─────────────────────────────────────────────────────────
resource "aws_vpc" "aggregator_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "fedtrust-aggregator-vpc" }
}

resource "aws_internet_gateway" "aggregator_igw" {
  vpc_id = aws_vpc.aggregator_vpc.id
  tags   = { Name = "fedtrust-aggregator-igw" }
}

resource "aws_subnet" "aggregator_subnet" {
  vpc_id                  = aws_vpc.aggregator_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "${var.aws_region}a"
  tags                    = { Name = "fedtrust-aggregator-subnet" }
}

resource "aws_route_table" "aggregator_rt" {
  vpc_id = aws_vpc.aggregator_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.aggregator_igw.id
  }
  tags = { Name = "fedtrust-aggregator-rt" }
}

resource "aws_route_table_association" "aggregator_rta" {
  subnet_id      = aws_subnet.aggregator_subnet.id
  route_table_id = aws_route_table.aggregator_rt.id
}

resource "aws_security_group" "aggregator_sg" {
  name        = "fedtrust-aggregator-sg"
  description = "Allow inbound HTTP (8000) for Web Dashboard and gRPC (8080) for Flower FL"
  vpc_id      = aws_vpc.aggregator_vpc.id

  # SSH Management
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # FastAPI Web Dashboard & REST API
  ingress {
    description = "FastAPI Service Dashboard"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Flower gRPC Aggregation Port
  ingress {
    description = "Flower FL Aggregator Server"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "fedtrust-aggregator-sg" }
}

# ── 2. Bank 1 Isolated VPC ───────────────────────────────────────────────────
resource "aws_vpc" "bank1_vpc" {
  cidr_block           = "10.1.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "fedtrust-bank1-vpc" }
}

resource "aws_internet_gateway" "bank1_igw" {
  vpc_id = aws_vpc.bank1_vpc.id
  tags   = { Name = "fedtrust-bank1-igw" }
}

resource "aws_subnet" "bank1_subnet" {
  vpc_id                  = aws_vpc.bank1_vpc.id
  cidr_block              = "10.1.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "${var.aws_region}a"
  tags                    = { Name = "fedtrust-bank1-subnet" }
}

resource "aws_route_table" "bank1_rt" {
  vpc_id = aws_vpc.bank1_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.bank1_igw.id
  }
  tags = { Name = "fedtrust-bank1-rt" }
}

resource "aws_route_table_association" "bank1_rta" {
  subnet_id      = aws_subnet.bank1_subnet.id
  route_table_id = aws_route_table.bank1_rt.id
}

resource "aws_security_group" "bank1_sg" {
  name        = "fedtrust-bank1-sg"
  description = "Isolated Bank 1 security group"
  vpc_id      = aws_vpc.bank1_vpc.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "fedtrust-bank1-sg" }
}

# ── 3. Bank 2 Isolated VPC ───────────────────────────────────────────────────
resource "aws_vpc" "bank2_vpc" {
  cidr_block           = "10.2.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "fedtrust-bank2-vpc" }
}

resource "aws_internet_gateway" "bank2_igw" {
  vpc_id = aws_vpc.bank2_vpc.id
  tags   = { Name = "fedtrust-bank2-igw" }
}

resource "aws_subnet" "bank2_subnet" {
  vpc_id                  = aws_vpc.bank2_vpc.id
  cidr_block              = "10.2.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "${var.aws_region}b"
  tags                    = { Name = "fedtrust-bank2-subnet" }
}

resource "aws_route_table" "bank2_rt" {
  vpc_id = aws_vpc.bank2_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.bank2_igw.id
  }
  tags = { Name = "fedtrust-bank2-rt" }
}

resource "aws_route_table_association" "bank2_rta" {
  subnet_id      = aws_subnet.bank2_subnet.id
  route_table_id = aws_route_table.bank2_rt.id
}

resource "aws_security_group" "bank2_sg" {
  name        = "fedtrust-bank2-sg"
  description = "Isolated Bank 2 security group"
  vpc_id      = aws_vpc.bank2_vpc.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "fedtrust-bank2-sg" }
}

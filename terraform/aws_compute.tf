# ==============================================================================
# AWS Compute — Aggregator Node & Isolated Client EC2 Instances
# Fulfills Section 4.1: Aggregator on dedicated EC2 + Client 1 & 2 in separate VPCs
# ==============================================================================

# Latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# SSH Key Pair
resource "aws_key_pair" "deployer" {
  key_name   = "fedtrust-deployer-key"
  public_key = var.ssh_public_key
}

# ── 1. Aggregator Server Instance ─────────────────────────────────────────────
resource "aws_instance" "aggregator" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.aws_instance_type
  subnet_id                   = aws_subnet.aggregator_subnet.id
  vpc_security_group_ids      = [aws_security_group.aggregator_sg.id]
  key_name                    = aws_key_pair.deployer.key_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.aggregator_profile.name

  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = <<-EOF
              #!/bin/bash
              set -e
              apt-get update -y
              apt-get install -y docker.io docker-compose git curl

              systemctl start docker
              systemctl enable docker
              usermod -aG docker ubuntu

              # Clone & launch FedTrust-Credit service
              git clone https://github.com/Prasannaa-V/fedtrust-credit.git /home/ubuntu/fedtrust-credit
              cd /home/ubuntu/fedtrust-credit
              docker-compose up -d aggregator
              EOF

  tags = {
    Name = "fedtrust-aggregator-server"
    Role = "Federated-Aggregator-and-API"
  }
}

# ── 2. Bank Client 1 Instance (AWS VPC 1) ─────────────────────────────────────
resource "aws_instance" "bank_client_1" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.aws_instance_type
  subnet_id                   = aws_subnet.bank1_subnet.id
  vpc_security_group_ids      = [aws_security_group.bank1_sg.id]
  key_name                    = aws_key_pair.deployer.key_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.bank1_profile.name

  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "fedtrust-bank-client-1"
    Role = "Bank-Client-1-Grades-A-B"
  }
}

# ── 3. Bank Client 2 Instance (AWS VPC 2) ─────────────────────────────────────
resource "aws_instance" "bank_client_2" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.aws_instance_type
  subnet_id                   = aws_subnet.bank2_subnet.id
  vpc_security_group_ids      = [aws_security_group.bank2_sg.id]
  key_name                    = aws_key_pair.deployer.key_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.bank2_profile.name

  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "fedtrust-bank-client-2"
    Role = "Bank-Client-2-Grades-C-D"
  }
}

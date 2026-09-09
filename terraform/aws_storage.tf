# ==============================================================================
# AWS Storage & IAM — Encrypted S3 Buckets & Least-Privilege Policies
# Fulfills Section 3.5: Per-Client Encrypted Storage & AWS IAM Policies
# ==============================================================================

# Random suffix for globally unique S3 bucket names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ── S3 Bucket: Bank 1 Private Data ────────────────────────────────────────────
resource "aws_s3_bucket" "bank1_bucket" {
  bucket        = "fedtrust-bank1-data-${random_id.bucket_suffix.hex}"
  force_destroy = true
  tags          = { Name = "fedtrust-bank1-data", Client = "bank_1" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bank1_sse" {
  bucket = aws_s3_bucket.bank1_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "bank1_block" {
  bucket                  = aws_s3_bucket.bank1_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── S3 Bucket: Bank 2 Private Data ────────────────────────────────────────────
resource "aws_s3_bucket" "bank2_bucket" {
  bucket        = "fedtrust-bank2-data-${random_id.bucket_suffix.hex}"
  force_destroy = true
  tags          = { Name = "fedtrust-bank2-data", Client = "bank_2" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bank2_sse" {
  bucket = aws_s3_bucket.bank2_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "bank2_block" {
  bucket                  = aws_s3_bucket.bank2_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM Roles & Policies ──────────────────────────────────────────────────────
data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# Aggregator Profile (CloudWatch metrics & logging)
resource "aws_iam_role" "aggregator_role" {
  name               = "fedtrust-aggregator-role-${random_id.bucket_suffix.hex}"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "aggregator_cw" {
  role       = aws_iam_role.aggregator_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "aggregator_profile" {
  name = "fedtrust-aggregator-profile-${random_id.bucket_suffix.hex}"
  role = aws_iam_role.aggregator_role.name
}

# Bank 1 Profile (isolated access to Bank 1 S3 bucket only)
resource "aws_iam_role" "bank1_role" {
  name               = "fedtrust-bank1-role-${random_id.bucket_suffix.hex}"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_policy" "bank1_s3_policy" {
  name        = "fedtrust-bank1-s3-access-${random_id.bucket_suffix.hex}"
  description = "Least privilege S3 access for Bank 1"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.bank1_bucket.arn,
          "${aws_s3_bucket.bank1_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "bank1_s3_attach" {
  role       = aws_iam_role.bank1_role.name
  policy_arn = aws_iam_policy.bank1_s3_policy.arn
}

resource "aws_iam_instance_profile" "bank1_profile" {
  name = "fedtrust-bank1-profile-${random_id.bucket_suffix.hex}"
  role = aws_iam_role.bank1_role.name
}

# Bank 2 Profile (isolated access to Bank 2 S3 bucket only)
resource "aws_iam_role" "bank2_role" {
  name               = "fedtrust-bank2-role-${random_id.bucket_suffix.hex}"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_policy" "bank2_s3_policy" {
  name        = "fedtrust-bank2-s3-access-${random_id.bucket_suffix.hex}"
  description = "Least privilege S3 access for Bank 2"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.bank2_bucket.arn,
          "${aws_s3_bucket.bank2_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "bank2_s3_attach" {
  role       = aws_iam_role.bank2_role.name
  policy_arn = aws_iam_policy.bank2_s3_policy.arn
}

resource "aws_iam_instance_profile" "bank2_profile" {
  name = "fedtrust-bank2-profile-${random_id.bucket_suffix.hex}"
  role = aws_iam_role.bank2_role.name
}

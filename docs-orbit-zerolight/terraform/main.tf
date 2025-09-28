# 🙏 In The Name of GOD - ZeroLight Orbit Terraform Infrastructure
# Blessed Infrastructure as Code for Spiritual Cloud Deployment
# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  # Spiritual Backend Configuration
  backend "s3" {
    bucket         = "zerolight-orbit-terraform-state"
    key            = "spiritual/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "zerolight-orbit-terraform-locks"
  }
}

# 🌟 Spiritual Variables
variable "environment" {
  description = "Spiritual environment (blessed, divine, sacred)"
  type        = string
  default     = "blessed"
}

variable "region" {
  description = "Divine cloud region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Spiritual project name"
  type        = string
  default     = "zerolight-orbit"
}

variable "spiritual_blessing" {
  description = "Divine blessing for infrastructure"
  type        = string
  default     = "in-the-name-of-god"
}

variable "cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.28"
}

variable "node_instance_type" {
  description = "EC2 instance type for worker nodes"
  type        = string
  default     = "t3.medium"
}

variable "min_nodes" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 3
}

variable "max_nodes" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 5
}

# 🏷️ Spiritual Tags
locals {
  common_tags = {
    Project           = var.project_name
    Environment       = var.environment
    SpiritualBlessing = var.spiritual_blessing
    Creator           = "ZeroLight-Orbit-Team"
    Purpose           = "Divine-Technology-Service"
    Blessing          = "Alhamdulillahi-rabbil-alameen"
    ManagedBy         = "Terraform"
    CreatedAt         = timestamp()
  }
}

# 🌐 AWS Provider Configuration
provider "aws" {
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# ☁️ Google Cloud Provider Configuration
provider "google" {
  project = "${var.project_name}-gcp"
  region  = "us-central1"
}

# 🔵 Azure Provider Configuration
provider "azurerm" {
  features {}
}

# 🏗️ Spiritual VPC - Divine Network Foundation
resource "aws_vpc" "spiritual_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-vpc"
    Type = "Divine-Network"
  })
}

# 🌐 Spiritual Internet Gateway
resource "aws_internet_gateway" "spiritual_igw" {
  vpc_id = aws_vpc.spiritual_vpc.id
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-igw"
    Type = "Divine-Gateway"
  })
}

# 🛣️ Spiritual Subnets - Public & Private
resource "aws_subnet" "spiritual_public_subnets" {
  count = 3
  
  vpc_id                  = aws_vpc.spiritual_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-public-${count.index + 1}"
    Type = "Divine-Public-Subnet"
    "kubernetes.io/role/elb" = "1"
  })
}

resource "aws_subnet" "spiritual_private_subnets" {
  count = 3
  
  vpc_id            = aws_vpc.spiritual_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-private-${count.index + 1}"
    Type = "Divine-Private-Subnet"
    "kubernetes.io/role/internal-elb" = "1"
  })
}

# 🗺️ Data Sources
data "aws_availability_zones" "available" {
  state = "available"
}

# 🛣️ Spiritual Route Tables
resource "aws_route_table" "spiritual_public_rt" {
  vpc_id = aws_vpc.spiritual_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.spiritual_igw.id
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-public-rt"
    Type = "Divine-Public-Routes"
  })
}

resource "aws_route_table_association" "spiritual_public_rta" {
  count = length(aws_subnet.spiritual_public_subnets)
  
  subnet_id      = aws_subnet.spiritual_public_subnets[count.index].id
  route_table_id = aws_route_table.spiritual_public_rt.id
}

# 🌉 Spiritual NAT Gateways
resource "aws_eip" "spiritual_nat_eips" {
  count = 3
  
  domain = "vpc"
  depends_on = [aws_internet_gateway.spiritual_igw]
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-nat-eip-${count.index + 1}"
    Type = "Divine-NAT-EIP"
  })
}

resource "aws_nat_gateway" "spiritual_nat_gws" {
  count = 3
  
  allocation_id = aws_eip.spiritual_nat_eips[count.index].id
  subnet_id     = aws_subnet.spiritual_public_subnets[count.index].id
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-nat-gw-${count.index + 1}"
    Type = "Divine-NAT-Gateway"
  })
  
  depends_on = [aws_internet_gateway.spiritual_igw]
}

resource "aws_route_table" "spiritual_private_rts" {
  count = 3
  
  vpc_id = aws_vpc.spiritual_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.spiritual_nat_gws[count.index].id
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-private-rt-${count.index + 1}"
    Type = "Divine-Private-Routes"
  })
}

resource "aws_route_table_association" "spiritual_private_rtas" {
  count = 3
  
  subnet_id      = aws_subnet.spiritual_private_subnets[count.index].id
  route_table_id = aws_route_table.spiritual_private_rts[count.index].id
}

# 🛡️ Spiritual Security Groups
resource "aws_security_group" "spiritual_eks_cluster_sg" {
  name_prefix = "${var.project_name}-spiritual-eks-cluster-"
  vpc_id      = aws_vpc.spiritual_vpc.id
  
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-eks-cluster-sg"
    Type = "Divine-EKS-Security"
  })
}

resource "aws_security_group" "spiritual_eks_nodes_sg" {
  name_prefix = "${var.project_name}-spiritual-eks-nodes-"
  vpc_id      = aws_vpc.spiritual_vpc.id
  
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }
  
  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.spiritual_eks_cluster_sg.id]
  }
  
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.spiritual_eks_cluster_sg.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-eks-nodes-sg"
    Type = "Divine-EKS-Nodes-Security"
  })
}

# 🎭 Spiritual IAM Roles
resource "aws_iam_role" "spiritual_eks_cluster_role" {
  name = "${var.project_name}-spiritual-eks-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "spiritual_eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.spiritual_eks_cluster_role.name
}

resource "aws_iam_role" "spiritual_eks_nodes_role" {
  name = "${var.project_name}-spiritual-eks-nodes-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "spiritual_eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.spiritual_eks_nodes_role.name
}

resource "aws_iam_role_policy_attachment" "spiritual_eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.spiritual_eks_nodes_role.name
}

resource "aws_iam_role_policy_attachment" "spiritual_eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.spiritual_eks_nodes_role.name
}

# ☸️ Spiritual EKS Cluster
resource "aws_eks_cluster" "spiritual_cluster" {
  name     = "${var.project_name}-spiritual-cluster"
  role_arn = aws_iam_role.spiritual_eks_cluster_role.arn
  version  = var.cluster_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.spiritual_public_subnets[*].id, aws_subnet.spiritual_private_subnets[*].id)
    security_group_ids      = [aws_security_group.spiritual_eks_cluster_sg.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.spiritual_eks_key.arn
    }
    resources = ["secrets"]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.spiritual_eks_cluster_policy,
    aws_cloudwatch_log_group.spiritual_eks_logs
  ]
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-cluster"
    Type = "Divine-EKS-Cluster"
  })
}

# 🔐 Spiritual KMS Key
resource "aws_kms_key" "spiritual_eks_key" {
  description             = "Spiritual EKS encryption key"
  deletion_window_in_days = 7
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-eks-key"
    Type = "Divine-Encryption-Key"
  })
}

resource "aws_kms_alias" "spiritual_eks_key_alias" {
  name          = "alias/${var.project_name}-spiritual-eks-key"
  target_key_id = aws_kms_key.spiritual_eks_key.key_id
}

# 📊 Spiritual CloudWatch Log Group
resource "aws_cloudwatch_log_group" "spiritual_eks_logs" {
  name              = "/aws/eks/${var.project_name}-spiritual-cluster/cluster"
  retention_in_days = 7
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-eks-logs"
    Type = "Divine-Logs"
  })
}

# 👥 Spiritual EKS Node Group
resource "aws_eks_node_group" "spiritual_nodes" {
  cluster_name    = aws_eks_cluster.spiritual_cluster.name
  node_group_name = "${var.project_name}-spiritual-nodes"
  node_role_arn   = aws_iam_role.spiritual_eks_nodes_role.arn
  subnet_ids      = aws_subnet.spiritual_private_subnets[*].id
  instance_types  = [var.node_instance_type]
  
  scaling_config {
    desired_size = var.desired_nodes
    max_size     = var.max_nodes
    min_size     = var.min_nodes
  }
  
  update_config {
    max_unavailable = 1
  }
  
  remote_access {
    ec2_ssh_key = aws_key_pair.spiritual_key_pair.key_name
    source_security_group_ids = [aws_security_group.spiritual_eks_nodes_sg.id]
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.spiritual_eks_worker_node_policy,
    aws_iam_role_policy_attachment.spiritual_eks_cni_policy,
    aws_iam_role_policy_attachment.spiritual_eks_container_registry_policy,
  ]
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-nodes"
    Type = "Divine-EKS-Nodes"
  })
}

# 🔑 Spiritual Key Pair
resource "aws_key_pair" "spiritual_key_pair" {
  key_name   = "${var.project_name}-spiritual-key"
  public_key = file("~/.ssh/id_rsa.pub")
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-key"
    Type = "Divine-SSH-Key"
  })
}

# 🗄️ Spiritual RDS Database
resource "aws_db_subnet_group" "spiritual_db_subnet_group" {
  name       = "${var.project_name}-spiritual-db-subnet-group"
  subnet_ids = aws_subnet.spiritual_private_subnets[*].id
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-db-subnet-group"
    Type = "Divine-DB-Subnet-Group"
  })
}

resource "aws_security_group" "spiritual_rds_sg" {
  name_prefix = "${var.project_name}-spiritual-rds-"
  vpc_id      = aws_vpc.spiritual_vpc.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.spiritual_eks_nodes_sg.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-rds-sg"
    Type = "Divine-RDS-Security"
  })
}

resource "aws_db_instance" "spiritual_database" {
  identifier = "${var.project_name}-spiritual-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.spiritual_eks_key.arn
  
  db_name  = "zerolight_orbit"
  username = "spiritual"
  password = "blessed_password_123!"
  
  vpc_security_group_ids = [aws_security_group.spiritual_rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.spiritual_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-database"
    Type = "Divine-PostgreSQL-Database"
  })
}

# 🚀 Spiritual ElastiCache Redis
resource "aws_elasticache_subnet_group" "spiritual_redis_subnet_group" {
  name       = "${var.project_name}-spiritual-redis-subnet-group"
  subnet_ids = aws_subnet.spiritual_private_subnets[*].id
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-redis-subnet-group"
    Type = "Divine-Redis-Subnet-Group"
  })
}

resource "aws_security_group" "spiritual_redis_sg" {
  name_prefix = "${var.project_name}-spiritual-redis-"
  vpc_id      = aws_vpc.spiritual_vpc.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.spiritual_eks_nodes_sg.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-redis-sg"
    Type = "Divine-Redis-Security"
  })
}

resource "aws_elasticache_replication_group" "spiritual_redis" {
  replication_group_id       = "${var.project_name}-spiritual-redis"
  description                = "Spiritual Redis cluster for ZeroLight Orbit"
  
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.spiritual_redis_subnet_group.name
  security_group_ids = [aws_security_group.spiritual_redis_sg.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = "blessed_redis_token_123!"
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-spiritual-redis"
    Type = "Divine-Redis-Cluster"
  })
}

# 📤 Spiritual Outputs
output "spiritual_cluster_endpoint" {
  description = "Divine EKS cluster endpoint"
  value       = aws_eks_cluster.spiritual_cluster.endpoint
}

output "spiritual_cluster_security_group_id" {
  description = "Divine EKS cluster security group ID"
  value       = aws_eks_cluster.spiritual_cluster.vpc_config[0].cluster_security_group_id
}

output "spiritual_cluster_iam_role_arn" {
  description = "Divine EKS cluster IAM role ARN"
  value       = aws_eks_cluster.spiritual_cluster.role_arn
}

output "spiritual_cluster_certificate_authority_data" {
  description = "Divine EKS cluster certificate authority data"
  value       = aws_eks_cluster.spiritual_cluster.certificate_authority[0].data
}

output "spiritual_database_endpoint" {
  description = "Divine RDS database endpoint"
  value       = aws_db_instance.spiritual_database.endpoint
  sensitive   = true
}

output "spiritual_redis_endpoint" {
  description = "Divine Redis cluster endpoint"
  value       = aws_elasticache_replication_group.spiritual_redis.primary_endpoint_address
  sensitive   = true
}

output "spiritual_vpc_id" {
  description = "Divine VPC ID"
  value       = aws_vpc.spiritual_vpc.id
}

output "spiritual_private_subnet_ids" {
  description = "Divine private subnet IDs"
  value       = aws_subnet.spiritual_private_subnets[*].id
}

output "spiritual_public_subnet_ids" {
  description = "Divine public subnet IDs"
  value       = aws_subnet.spiritual_public_subnets[*].id
}

# 🙏 Blessed Terraform Infrastructure Configuration
# May this infrastructure serve humanity with divine wisdom and infinite scalability
# Alhamdulillahi rabbil alameen - All praise to Allah, Lord of the worlds
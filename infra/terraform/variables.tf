variable "region" {
  description = "AWS region."
  type        = string
  default     = "eu-west-2"
}

variable "name_prefix" {
  description = "Prefix for every resource name."
  type        = string
  default     = "araa"
}

variable "image_uri" {
  description = <<-EOT
    ECR image URI for the ingestion Lambda, including the tag or digest.
    Built from the repository Dockerfile. Pin by digest in anything real:
    a mutable tag means the scheduled job silently changes under you.
  EOT
  type        = string
}

variable "schedule_expression" {
  description = <<-EOT
    EventBridge schedule for re-ingestion. Kept in sync with ARAA_INGEST_CRON
    so the local APScheduler worker and the Lambda run the same cadence.
    Note EventBridge cron has six fields and a mandatory '?' in one of the day
    slots; it is not the same grammar as the five-field crontab used locally.
  EOT
  type        = string
  default     = "cron(15 3 * * ? *)"
}

variable "corpus_bucket_name" {
  description = "S3 bucket for the corpus, the index and the run logs. Empty means generate one."
  type        = string
  default     = ""
}

variable "lambda_memory_mb" {
  description = <<-EOT
    Embedding 2k+ chunks with bge-small is memory- and CPU-bound. Lambda
    allocates vCPU proportionally to memory, so this is really a CPU dial:
    below ~3GB the embedding pass gets slow enough to threaten the timeout.
  EOT
  type        = number
  default     = 4096

  validation {
    condition     = var.lambda_memory_mb >= 2048 && var.lambda_memory_mb <= 10240
    error_message = "Embedding needs at least 2048 MB; Lambda's ceiling is 10240 MB."
  }
}

variable "lambda_timeout_s" {
  description = "Lambda timeout. Ingesting 200 papers took ~4 minutes on a laptop CPU."
  type        = number
  default     = 900

  validation {
    condition     = var.lambda_timeout_s > 0 && var.lambda_timeout_s <= 900
    error_message = "Lambda's maximum timeout is 900 seconds."
  }
}

variable "ephemeral_storage_mb" {
  description = "/tmp size. PDFs plus the Chroma index need more than the 512 MB default."
  type        = number
  default     = 4096
}

variable "corpus_target_docs" {
  description = "Documents to fetch per scheduled run. Matches araa.config."
  type        = number
  default     = 12
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 30
}

variable "alarm_email" {
  description = "Optional email for ingestion-failure alarms. Empty disables the subscription."
  type        = string
  default     = ""
}

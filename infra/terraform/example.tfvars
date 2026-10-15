# Copy to terraform.tfvars and edit before applying.
#
# Nothing here has been applied. See main.tf and docs/architecture.md.

region      = "eu-west-2"
name_prefix = "araa"

# Build and push the repository Dockerfile to ECR, then pin by digest.
# A mutable tag means the scheduled job changes underneath you without a plan.
image_uri = "111122223333.dkr.ecr.eu-west-2.amazonaws.com/araa:sha256:REPLACE_ME"

# EventBridge cron: six fields, and one of the day slots must be '?'.
# This is NOT the five-field crontab used by ARAA_INGEST_CRON locally.
schedule_expression = "cron(15 3 * * ? *)"

corpus_target_docs = 200
lambda_memory_mb   = 4096
lambda_timeout_s   = 900

# alarm_email = "you@example.com"

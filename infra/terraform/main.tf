# Scheduled ingestion on AWS.
#
# NOT APPLIED. This configuration is written, `terraform validate`d and
# `terraform fmt`ed in CI, and has never been run against an AWS account.
# Saying so is the point: an unapplied Terraform config that claims to be
# deployed infrastructure is a lie, and an unapplied config that says it is
# unapplied is a design artefact. See docs/architecture.md.
#
# What actually runs today is the APScheduler worker in
# `src/araa/ingestion/schedule.py` (`docker compose --profile scheduler up`).
# Both call the same `scheduled_ingest()` entry point, so this is a second
# trigger for one implementation rather than a second implementation.

locals {
  name   = var.name_prefix
  bucket = var.corpus_bucket_name != "" ? var.corpus_bucket_name : "${var.name_prefix}-corpus-${random_id.suffix.hex}"
}

resource "random_id" "suffix" {
  byte_length = 4
}

# --------------------------------------------------------------- storage
resource "aws_s3_bucket" "corpus" {
  bucket = local.bucket
}

resource "aws_s3_bucket_public_access_block" "corpus" {
  bucket                  = aws_s3_bucket.corpus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "corpus" {
  bucket = aws_s3_bucket.corpus.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "corpus" {
  bucket = aws_s3_bucket.corpus.id

  # The index is rebuilt in place on every run. Versioning is what lets you
  # roll back to the previous index when a re-embed makes retrieval worse -
  # which is not hypothetical, it is the reason the digest reports deltas.
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "corpus" {
  bucket = aws_s3_bucket.corpus.id

  rule {
    id     = "expire-old-index-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# ------------------------------------------------------------------ iam
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingest" {
  name               = "${local.name}-ingest"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "ingest" {
  # Scoped to this bucket only. The ingestion job reads third-party PDFs, so
  # a wildcard here would mean a malicious document had a wildcard too.
  statement {
    sid       = "CorpusObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.corpus.arn}/*"]
  }

  statement {
    sid       = "CorpusList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.corpus.arn]
  }

  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.ingest.arn}:*"]
  }
}

resource "aws_iam_role_policy" "ingest" {
  name   = "${local.name}-ingest"
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

# --------------------------------------------------------------- compute
resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/aws/lambda/${local.name}-ingest"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "ingest" {
  function_name = "${local.name}-ingest"
  role          = aws_iam_role.ingest.arn
  package_type  = "Image"
  image_uri     = var.image_uri

  # Container image, because the embedding and reranker weights are ~500 MB
  # baked into the image. A zip package caps at 250 MB unzipped.
  image_config {
    command = ["infra.lambda.handler.handler"]
  }

  memory_size = var.lambda_memory_mb
  timeout     = var.lambda_timeout_s

  ephemeral_storage {
    size = var.ephemeral_storage_mb
  }

  environment {
    variables = {
      ARAA_CORPUS_BUCKET      = aws_s3_bucket.corpus.id
      ARAA_CORPUS_TARGET_DOCS = tostring(var.corpus_target_docs)
      ARAA_DATA_DIR           = "/tmp/data"
      ARAA_CORPUS_DIR         = "/tmp/data/corpus"
      ARAA_CHROMA_DIR         = "/tmp/data/chroma"
      ARAA_INDEX_DIR          = "/tmp/data/index"
      ARAA_RUNS_DIR           = "/tmp/data/runs"
      HF_HOME                 = "/models"
      ANONYMIZED_TELEMETRY    = "False"
      # Ingestion is embeddings only - it never calls a generative model, so
      # no LLM credentials are needed here at all.
      ARAA_LLM_BACKEND = "fixture"
    }
  }

  depends_on = [aws_cloudwatch_log_group.ingest]
}

# -------------------------------------------------------------- schedule
resource "aws_cloudwatch_event_rule" "ingest" {
  name                = "${local.name}-ingest-schedule"
  description         = "Re-fetch the corpus and rebuild the index."
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "ingest" {
  rule      = aws_cloudwatch_event_rule.ingest.name
  target_id = "${local.name}-ingest"
  arn       = aws_lambda_function.ingest.arn
}

resource "aws_lambda_permission" "events" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ingest.arn
}

# ------------------------------------------------------------- alarming
# A scheduled job that silently stops running is indistinguishable from one
# that runs and finds nothing new, until an answer cites a document that no
# longer exists. Alarm on both failure and on the absence of invocations.
resource "aws_sns_topic" "alerts" {
  name = "${local.name}-ingest-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

resource "aws_cloudwatch_metric_alarm" "ingest_errors" {
  alarm_name          = "${local.name}-ingest-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 86400
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Scheduled ingestion failed."
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = aws_lambda_function.ingest.function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "ingest_missing" {
  alarm_name          = "${local.name}-ingest-not-running"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocations"
  namespace           = "AWS/Lambda"
  period              = 172800 # two days: a daily job that missed two windows
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Scheduled ingestion has not run. Silence is the failure mode that hides."
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "breaching"

  dimensions = {
    FunctionName = aws_lambda_function.ingest.function_name
  }
}

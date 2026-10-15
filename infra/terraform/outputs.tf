output "corpus_bucket" {
  description = "S3 bucket holding the corpus, the index and the run logs."
  value       = aws_s3_bucket.corpus.id
}

output "lambda_function_name" {
  description = "Scheduled ingestion function."
  value       = aws_lambda_function.ingest.function_name
}

output "schedule_expression" {
  description = "EventBridge schedule actually in effect."
  value       = aws_cloudwatch_event_rule.ingest.schedule_expression
}

output "log_group" {
  description = "CloudWatch log group for ingestion runs."
  value       = aws_cloudwatch_log_group.ingest.name
}

output "alerts_topic_arn" {
  description = "SNS topic for ingestion failure and staleness alarms."
  value       = aws_sns_topic.alerts.arn
}

output "not_applied_notice" {
  description = "Read this before believing anything above."
  value       = <<-EOT
    This configuration has never been applied to an AWS account. It is
    validated and formatted in CI only. The scheduled ingestion that actually
    runs is the APScheduler worker in src/araa/ingestion/schedule.py, started
    with `docker compose --profile scheduler up`.
  EOT
}

output "trigger_id" {
  value       = google_cloudbuild_trigger.benchmark-trigger.id
  description = "The ID of the created Cloud Build trigger."
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloudbuild_trigger" "benchmark-trigger" {
  name        = "aletheia-benchmark-trigger"
  description = "Trigger to run tests and deploy on push to main"

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }

  filename = "cloudbuild.yaml"
}

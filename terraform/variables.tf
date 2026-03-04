variable "project_id" {
  description = "The ID of the Google Cloud project."
  type        = string
}

variable "region" {
  description = "The region for resources."
  type        = string
  default     = "us-central1"
}

variable "github_owner" {
  description = "The GitHub repository owner."
  type        = string
}

variable "github_repo" {
  description = "The GitHub repository name."
  type        = string
}

#!/bin/bash
gcloud builds submit --config cloudbuild.yaml .
gcloud run deploy aletheia-ui --source . --port 8501 --allow-unauthenticated
gcloud run jobs create aletheia-benchmark --source . --command "./run.sh" --args "benchmark"
gcloud run jobs execute aletheia-benchmark

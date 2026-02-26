# Vertex AI Evaluation Workflow for Aletheia

This document describes the workflow for evaluating and improving the Aletheia AI Detector using the **Vertex AI Gen AI Evaluation Service**.

## Overview

The goal is to use Google Cloud's enterprise-grade evaluation infrastructure to:
1.  **Generate Adversarial Data:** Use advanced LLMs (Gemini Pro) to generate "Humanized" AI text designed to evade detection.
2.  **Evaluate Detector Performance:** Run Aletheia against these adversarial samples.
3.  **Compare with "Ground Truth":** Use a powerful LLM-as-a-Judge (Gemini) to rate the "Human-Likeness" of the text.
4.  **Identify Weaknesses:** Find cases where the LLM Judge correctly identifies the text as AI (low human-likeness) but Aletheia fails (low AI score).

## Components

### 1. Adversarial Data Generation
*   **Script:** `scripts/generate_adversarial_data.py`
*   **Description:** Generates a dataset of AI text and then "humanizes" it using specific strategies (burstiness, colloquialisms).
*   **Output:** `data/adversarial_samples.json`

### 2. Custom Metrics (`ia_detector/vertex_metrics.py`)
*   **`aletheia_ai_score`**: A custom Python metric that wraps the local `EnsembleDetector`. It returns the AI Probability (0-100) for each sample.

### 3. Evaluation Script (`scripts/run_vertex_eval.py`)
*   **Description:** Orchestrates the evaluation using `vertexai.EvaluationTask`.
*   **Metrics Used:**
    *   **Coherence/Fluency/Safety:** Built-in Vertex AI metrics to ensure the adversarial text is high quality.
    *   **`human_likeness`:** A custom **Pointwise LLM Metric**. The judge rates the text on a 1-5 scale based on fluency, colloquialisms, and imperfections.
    *   **`aletheia_ai_score`:** The actual detector score we want to test.

## Running the Workflow

### Prerequisites
*   Google Cloud Project with Vertex AI API enabled.
*   `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` environment variables set.
*   `GEMINI_API_KEY` (optional, for local generation).

### Step 1: Generate Data
```bash
python scripts/generate_adversarial_data.py
```
This will create `data/adversarial_samples.json` with new challenging examples.

### Step 2: Run Evaluation
```bash
python scripts/run_vertex_eval.py
```
This script will:
*   Upload the dataset to Vertex AI (in-memory or staging).
*   Run the LLM Judge to rate "Human-Likeness".
*   Run the Aletheia Detector locally on the samples.
*   Correlate the results.

### Step 3: Analyze Results
The results are saved to `docs/benchmarks/data/vertex_eval_results.csv`.

**Improvement Loop:**
1.  Look for samples where **Judge AI Probability is High** (Low Human-Likeness) but **Aletheia Score is Low** (Classified as Human).
2.  These are **False Negatives**.
3.  Analyze the text features of these samples (e.g., did they use specific "filler" words? was the burstiness artificially high?).
4.  Tune the `EnsembleDetector` weights or add new features in `ia_detector/features.py`.
5.  Re-run the evaluation to verify improvement.

## Metric Interpretation

*   **`human_likeness` (1-5):**
    *   5: Perfectly Human.
    *   1: Obviously AI.
*   **`aletheia_ai_score` (0-100):**
    *   0: Human.
    *   100: AI.
*   **Correlation:** We want a **strong negative correlation** between `human_likeness` and `aletheia_ai_score`. (High Human-Likeness -> Low AI Score).

## Troubleshooting
*   **Quota Issues:** If you hit 429 errors, reduce the dataset size in `generate_adversarial_data.py` or request a quota increase for Vertex AI.
*   **Auth Errors:** Ensure `gcloud auth application-default login` is run or Service Account keys are set.

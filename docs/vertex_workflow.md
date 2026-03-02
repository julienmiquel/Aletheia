# Aletheia Optimization Workshop Context

> This file provides context for AI assistants and developers working on improving the Aletheia Detector. For setup instructions, see [README.md](../README.md).

---

## Workshop Overview

This is the **Aletheia Optimization & Evaluation Workflow** guide.

**Objective:** Teach developers to move from heuristic-based statistical detection to rigorous, data-driven context engineering, using a production-grade Vertex AI evaluation framework to measure and validate detector improvements against adversarial attacks.

**Key Concepts:**
- Context Engineering: Systematic tuning of detection sensors (Perplexity, Burstiness, GLTR, Semantic Consistency).
- Evaluation Framework: 3-step process (Generate Adversarial Data → Run Evaluation → Analyze Correlation).
- Hill Climb Methodology: Iterative optimization from M0 (baseline heuristic) to M5 (fully optimized, ML-backed ensemble).

---

## Repository Structure

```
aletheia-detector/
├── README.md                  # Project overview and installation
├── docs/
│   ├── vertex_workflow.md     # This file (evaluation workflow)
│   └── benchmarks/data/       # Evaluation results output directory
├── ia_detector/               # Core Aletheia package
│   ├── ensemble.py            # The main detector orchestrator
│   └── vertex_metrics.py      # Custom Vertex AI metric wrapper
├── data/                      # Datasets
│   └── adversarial_samples.json # Generated adversarial test cases
└── scripts/                   # Evaluation and data generation scripts
    ├── generate_adversarial_data.py # Red team generator
    └── run_vertex_eval.py     # Vertex AI evaluation pipeline
```

---

## What Participants Are Doing

1. **Running baseline evaluations** on the `main` branch using standard heuristics.
2. **Generating adversarial data** (`scripts/generate_adversarial_data.py`) to simulate sophisticated evasion attempts.
3. **Checking out optimization branches** (e.g., tuning thresholds, adding new features).
4. **Re-running evaluations** (`scripts/run_vertex_eval.py`) to compare before/after metrics.
5. **Understanding trade-offs** between false positives (flagging human text) and false negatives (missing adversarial AI text).

---

## Common Tasks You Should Help With

### Running Evaluations
```bash
# 1. Generate challenging adversarial data
python scripts/generate_adversarial_data.py

# 2. Run the Vertex AI evaluation pipeline
python scripts/run_vertex_eval.py
```

### Creating Custom Metrics
Metrics are defined in `scripts/run_vertex_eval.py`. Help participants write custom LLM-as-Judge metrics (like `human_likeness`) with clear scoring criteria using the modern **`vertexai.types.LLMMetric`** and **`vertexai.types.MetricPromptBuilder`**.

**Important:** Do NOT use the deprecated `vertexai.evaluation.EvalTask` or `PointwiseMetric`. Always use the unified `vertexai.Client`.

### Debugging Evaluation Issues
- Quota errors (429) → Reduce batch size or configure retries.
- "Project not set" → Ensure `GOOGLE_CLOUD_PROJECT` environment variable is defined.
- Correlation errors → Ensure metric scores are normalized correctly before comparing (e.g., converting 1-5 Likeness to 0-100 AI Probability).

### Understanding Results
- `docs/benchmarks/data/vertex_eval_results.csv` → Raw evaluation data.
- Look for **Disagreements**: Cases where the LLM Judge determines text is AI (low human-likeness) but Aletheia scores it as Human.
- Low correlation → Map to optimization patterns (e.g., the detector relies too heavily on simple burstiness, which adversarial models now fake).

---

## Optimization Branches

| Branch | Optimization | Component | Focus |
|--------|--------------|-------|-------|
| `main` | Baseline | Ensemble | Establish baseline correlation |
| `optimizations/01-semantic-tuning` | Semantic Prompt Hardening | `SemanticConsistencyAnalyzer` | Fix hallucinated logic detection |
| `optimizations/02-burstiness-thresholds` | Threshold Tuning | `BurstinessAnalyzer` | Reduce false positives on short text |
| `optimizations/03-surrogate-integration` | Model Integration | `SurrogatePPLDetector` | Catch advanced style mimicry |

---

## Critical Reminders

1. **Always use the unified `vertexai.Client`**, ensuring ADC or API keys are properly configured.
2. **Generate fresh data** if evaluating new adversarial strategies.
3. **Run `uv sync` or `pip install`** after switching branches if dependencies change.
4. **Check correlation** as the primary success metric, not just raw accuracy.

---

## Creating Optimization Logs (Comparing Results)

When participants run evaluations on baseline vs optimized code, help them create an **OPTIMIZATION_LOG.md** that compares results. This is the key deliverable for understanding what changed.

### Location
Save optimization logs to: `docs/benchmarks/OPTIMIZATION_LOG.md`

### How to Generate

**Option A: The AI Assist Method (Recommended)**

Use this prompt to have your AI assistant generate the log for you.

```text
Role: You are a Senior AI Forensics Architect and QA Analyst.
Objective: Update the OPTIMIZATION_LOG.md file to prove whether the applied detection strategy worked against adversarial attacks.

Inputs:
1. Strategy Applied: [e.g., "Iteration 1: Burstiness Threshold Tuning"]
2. New Evaluation Data: [Paste correlation scores and disagreement counts from run_vertex_eval.py output]
3. Qualitative Analysis: [Describe the types of text the detector started catching or missing]
4. Current Log: [Paste current OPTIMIZATION_LOG.md content]

Instructions:
1. Update Metrics Table: Calculate deltas between previous iteration and this one. Use 🟢/🔴/⚪.
2. Append Iteration History:
   - Create a new section for this Iteration.
   - Optimization Focus: (e.g., Statistical Tuning, Semantic Deep Dive).
   - Analysis of Variance: Did the correlation with the Ground Truth Judge improve? Quote specific metrics.
   - Evidence: Extract 1-2 specific text examples that the detector previously missed but now catches.
   - Conclusion: One sentence summary of the strategic pivot for the next step.
```

**Option B: The Manual Method**

**Step 1: Extract metrics from evaluation runs**

Look at the console output of `python scripts/run_vertex_eval.py`. Pay attention to the correlation score and the number of "significant disagreements".

```python
# Pseudo-code for tracking metrics
baseline_correlation = -0.65
optimization_correlation = -0.82

baseline_disagreements = 12
optimization_disagreements = 4
```

**Step 2: Create the OPTIMIZATION_LOG.md with this structure:**

```markdown
# Optimization Log: Aletheia Detector

**Branch:** `optimizations/XX-name`
**Optimization:** [Name] (Focus: Statistical/Semantic/Ensemble)
**Date:** YYYY-MM-DD

## 1. Metrics Comparison Table

| Metric | Baseline | Optimization | Delta |
|--------|----------|--------------|-------|
| Correlation (Aletheia vs Judge) | -0.65 | -0.82 | -0.17 🟢 |
| Significant Disagreements | 12 | 4 | -8 🟢 |

## 2. Iteration History

### Baseline (M0)
- Detector relies heavily on basic Perplexity.
- Frequently fooled by prompted burstiness.

### Optimization XX
- What was implemented: Integrated Surrogate PPL.
- What improved: Correlation increased significantly. The detector now catches texts that artificially inflate burstiness but maintain predictable token paths.
- What trade-offs occurred: Analysis latency increased by 15%.

## 3. Conclusions
- What worked: Adding neural stylistic checks improved resilience.
- What needs attention: Latency on short texts is too high.
- Recommended next optimization: Implement caching for surrogate model results.
```

### Emoji Legend for Deltas
- 🟢 = Improvement (stronger negative correlation OR fewer disagreements)
- 🔴 = Regression (weaker correlation OR more disagreements)
- ⚪ = Neutral/unchanged

---

## Key Files to Reference

- `README.md` - Framework overview
- `scripts/run_vertex_eval.py` - The core evaluation pipeline
- `ia_detector/vertex_metrics.py` - How the detector integrates with Vertex AI
- `docs/benchmarks/OPTIMIZATION_LOG.md` - Optimization comparison reports

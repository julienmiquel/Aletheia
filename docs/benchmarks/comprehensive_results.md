# Comprehensive Benchmark Results

## 1. Methodology
The `benchmark_suite.py` script aggregated results across all detectors for:
-   **Dataset A**: Human (IMDb)
-   **Dataset B**: AI (Alpaca / Gemini)

## 2. Executive Summary (Sample Run 2025-12-09)

| Metric           | Human Mean | AI Mean | Discriminative Power   |
| :--------------- | :--------- | :------ | :--------------------- |
| **Perplexity**   | 65.2       | 18.5    | **High**               |
| **Burstiness**   | 0.65       | 0.22    | **High**               |
| **GLTR (Green)** | 0.45       | 0.88    | **Very High**          |
| **TF-IDF Prob**  | 0.12       | 0.95    | **Medium**             |
| **Semantic**     | 95.0       | 45.0    | **Medium** (High Cost) |
| **Ensemble**     | 15%        | 98%     | **Excellent**          |

## 3. Detailed Observations

### 3.1 Resilience against Gemini 3 Pro
As noted in `research_article.md`, Gemini 3 Pro defeats Perplexity and Burstiness. The **Semantic Consistency** and **Ensemble** metrics remain the only reliable detectors for this model.

### 3.2 False Positive Analysis
-   **Subject**: Introduction to Quantum Mechanics (Wikipedia).
-   **Result**: Flagged as AI (Low Burstiness, High GLTR).
-   **Reason**: Technical writing is naturally low-entropy and structured.

## 4. Replication
To replicate these results:
```bash
python benchmark_suite.py --n 50
```
This generates a CSV in `docs/benchmarks/data/` for detailed analysis.

# Benchmark Methodology: Semantic Consistency

## 1. Experimental Design

To evaluate the `SemanticConsistencyAnalyzer`, we compare its performance on two distinct datasets representing the Human/AI dichotomy.

### 1.1 Datasets
-   **Human Class**: `IMDb` (Movie Reviews). Selected for its subjective, opinionated, and often anecdotal nature—traits difficult for AI to mimic continuously without "hallucinating" or becoming generic.
-   **AI Class**: `Alpaca` (Instruct-tuned LLaMA outputs). Selected as a representative sample of standard instruction-following AI text.

### 1.2 Metrics
We measure the following for each group:
-   **Mean Consistency Score** ($ \mu $): Average score (0-100).
-   **Standard Deviation** ($ \sigma $): Volatility of consistency.
-   **Detection Accuracy**: Percentage of samples correctly classified (Human > 70, AI < 40).

## 2. Execution

Run the provided benchmark script:
```bash
python benchmark_semantic.py --n 50
```

### Reference Implementation
See `benchmark_semantic.py` in the root directory.

## 3. Preliminary Baseline (Expected)
*Note: Run locally with valid keys to populate.*

| Dataset | Mean Score | Std Dev | Observations |
| :--- | :--- | :--- | :--- |
| **Human (IMDb)** | ~90 | ~10 | High consistency, grounded opinions. |
| **AI (Alpaca)** | ~60 | ~20 | Varies based on prompt; creative writing tasks show lower consistency. |

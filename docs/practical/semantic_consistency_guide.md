# Practical Guide: Semantic Consistency Analysis in `IA_detector`

## 1. Overview

The `SemanticConsistencyAnalyzer` is a Python module designed to evaluate the logical coherence of a text segment. Unlike statistical detectors (Burstiness, PPL), it uses a Large Language Model (Gemini 3 Pro) to "read" the text and identify internal contradictions or "hallucinations."

## 2. API Reference

### 2.1 Class Initialization
```python
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer

# Initialize with default model (Gemini 3 Pro)
analyzer = SemanticConsistencyAnalyzer()
```

**Environment Variables**:
-   `GEMINI_API_KEY`: (Recommended) Your AI Studio API Key.
-   `GOOGLE_CLOUD_PROJECT`: (Optional) GCP Project ID for Vertex AI.
-   `GOOGLE_CLOUD_LOCATION`: (Optional) Region (default: `us-central1`).

### 2.2 Analysis Method
```python
result = analyzer.analyze(text_string)
```

**Input**:
-   `text_string` (str): The text to analyze. Truncated to 4000 characters by default to fit context limits efficiently.

**Output**### 2. Quantitative Coherence Metric
**What it is:** A mathematically calculated score (0-100) representing the "Semantic Flow" or "Lexical Cohesion" of the text.
**How it works:**
1.  Segments text into sentences using SpaCy.
2.  Computes TF-IDF vectors for each sentence.
3.  Calculates the **Cosine Similarity** between adjacent sentences ($S_i$ vs $S_{i+1}$).
4.  Returns the average similarity.
**Interpretation:**
*   **High (>30)**: Sentences share many keywords/topics. Highly cohesive (sometimes repetitive).
*   **Low (<10)**: Sentences are disjointed or jump topics rapidly.
*   *Note:* AI often has higher coherence than creative human writing due to statistical "safety".

### 3. LLM Consistency Check
**What it is:** (dict):
```json
{
    "consistency_score": 95.0,
    "reasoning": "The text maintains a consistent timeline and logical flow. No contradictions found."
}
```

-   **consistency_score** ($0-100$):
    -   $0-40$: High probability of AI hallucination/contradiction.
    -   $41-70$: Ambiguous or generic text.
    -   $71-100$: High consistency (Likely Human or Grounded AI).

## 3. Integration Example

This module is designed to be part of the `EnsembleDetector`.

```python
from ia_detector.ensemble import EnsembleDetector
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer

detector = EnsembleDetector()
consistency_checker = SemanticConsistencyAnalyzer()

text = "..."

# Standard statistical checks
stats = detector.predict(text)

# Semantic check (Expensive, use optionally)
semantic = consistency_checker.analyze(text)

print(f"Statistical Prob: {stats['combined_score']}%")
print(f"Semantic Consistency: {semantic['consistency_score']}%")

if stats['combined_score'] > 50 and semantic['consistency_score'] < 40:
    print("Verdict: DEFINITELY AI (Statistical detection + Hallucination confirmed)")
```

## 4. Best Practices
-   **Cost Management**: This analyzer makes an LLM call per analysis. Cache results using `ResultCache` (see `ia_detector/cache.py`) for production use.
-   **Context Limit**: Analysis is most effective on texts long enough to contain a narrative ($>200$ words) but short enough to fit in the prompt context.

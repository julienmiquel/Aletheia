# Practical Guide: GLTR Analyzer

## 1. Overview

The `GLTRAnalyzer` uses a BERT-based model (`gpt2` or `bert-base-cased`) to rank the tokens of an input text. It calculates the percentage of tokens that fall into the Top-10 / Top-100 probability buckets.

## 2. API Reference

### 2.1 Initialization
```python
from ia_detector.gltr import GLTRAnalyzer

# Loads GPT-2 Small by default
gltr = GLTRAnalyzer()
```

### 2.2 Analysis
```python
results = gltr.analyze(text)
stats = gltr.get_fraction_clean(results)
```

**Output** (`stats` dict):
```json
{
    "Green": 0.75,   # % in Top 10
    "Yellow": 0.15,  # % in Top 100
    "Red": 0.08,     # % in Top 1000
    "Purple": 0.02   # % > 1000
}
```

## 3. Use Cases
-   **Visual Debugging**: The most powerful feature of GLTR is visual. While this API returns stats, a frontend can use the token-level ranks in `results` to highlight text (Green/Yellow/Red).
-   **Ensemble Signal**: The `Green` fraction is a strong feature for the `EnsembleDetector`.

## 4. Performance Note
GLTR is **computationally expensive**. It requires a forward pass of a Transformer model for *every token*.
-   **Speed**: ~10 tokens/sec on CPU.
-   **Optimization**: Use the `ResultCache` to store GLTR calculations for repeated texts.

## 5. Thresholds
-   **Green > 0.60**: Likely AI.
-   **Green < 0.45**: Likely Human.

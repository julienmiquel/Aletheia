# Practical Guide: Burstiness Analyzer

## 1. Overview

The `BurstinessAnalyzer` evaluates the structural variance of a text. It uses the `spacy` NLP library to parse sentences accurately rather than simple period-splitting.

## 2. API Reference

### 2.1 Initialization
```python
from ia_detector.burstiness import BurstinessAnalyzer

analyzer = BurstinessAnalyzer()
# Spacy 'en_core_web_sm' model is loaded automatically
```

### 2.2 Analysis
```python
result = analyzer.analyze(text)
```

**Output** (dict):
```json
{
    "burstiness_coefficient": 0.54, 
    "avg_sentence_length": 18.2, 
    "std_dev_sentence_length": 9.8,
    "num_sentences": 12
}
```

## 3. Interpretation
-   **Burstiness < 0.2**: Likely AI (Very monotonous).
-   **Burstiness 0.2 - 0.4**: Ambiguous.
-   **Burstiness > 0.4**: Likely Human (High structural variance).

## 4. Dependencies
Requires `spacy` and the English language model:
```bash
python -m spacy download en_core_web_sm
```

## 5. Weaknesses
-   **Short Texts**: Variance calculations on texts with $< 5$ sentences are statistically insignificant and noisy. The analyzer generates a warning for such inputs.
-   **Poetry/Lyrics**: Highly structured human text can have low burstiness.

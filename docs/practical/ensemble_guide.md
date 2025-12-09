# Practical Guide: Ensemble Detector

## 1. Overview

The `EnsembleDetector` is the main entry point for the `IA_detector` library. It aggregates signals from all other detectors to provide a final verdict.

## 2. API Reference

### 2.1 Initialization
```python
from ia_detector.ensemble import EnsembleDetector

# Initializes ALL sub-models (can take memory)
detector = EnsembleDetector()
```

### 2.2 Prediction
```python
result = detector.predict(text)
```

**Output** (dict):
```json
{
    "combined_score": 92.5,  # 0-100% Likelihood of AI
    "verdict": "AI",         # "AI", "Human", "Uncertain"
    "confidence": "High",    # "Low", "Medium", "High"
    "metrics": {
        "perplexity": 12.4,
        "burstiness": 0.15,
        "gltr_green": 0.82,
        "tfidf_prob": 0.99
    }
}
```

## 3. Configuration

### 3.1 Weight Tuning
You can adjust the heuristic weights in `ia_detector/ensemble.py` if you notice specific false positives.

```python
# ia_detector/ensemble.py

# Current Weights
WEIGHTS = {
    'perplexity': 2.0,
    'burstiness': 1.5,
    'gltr': 1.5,
    'tfidf': 2.5  # Highest weight due to specificity
}
```

## 4. Training the Meta-Classifier
To move beyond heuristics to a trained Random Forest:
1.  Harvest a large dataset (Human/AI).
2.  Run `train_ensemble.py`.
3.  This generates `ensemble_model.pkl`.
4.  The `EnsembleDetector` will automatically load this pickle if it exists, overriding the heuristic logic.

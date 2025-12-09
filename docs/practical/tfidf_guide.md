# Practical Guide: TF-IDF Stylometric Detector

## 1. Overview

The `TfidfDetector` is a traditional Machine Learning component that classifies text based on word usage patterns. It detects the "accent" of the AI.

## 2. API Reference

### 2.1 Initialization
The model requires a pre-trained pickle file.
```python
from ia_detector.features import TfidfDetector

# Automatically loads 'tfidf_model.pkl' if present
detector = TfidfDetector()
```

### 2.2 Training (Required first time)
If `tfidf_model.pkl` is missing, you must train it:
```python
human_texts = [...] # List of strings
ai_texts = [...]    # List of strings

detector.train(human_texts, ai_texts)
detector.save() # Saves to disk
```

### 2.3 Prediction
```python
result = detector.predict(text)
```

**Output**:
```json
{
    "ai_probability": 0.85, # 0.0 - 1.0
    "top_features": ["delve", "crucial", "harness"] # Words that triggered the decision
}
```

## 3. Maintenance

### 3.1 Model Drift
As new LLMs (e.g., Gemini 3) are released, their style changes. The TF-IDF model must be **retrained periodically** with fresh samples to remain effective.
-   **Old Style**: "In conclusion, it is important..."
-   **New Style**: (More conversational, but may overuse specific metaphors).

### 3.2 Retraining Script
Use the provided `train_ensemble.py` or `fetch_and_test.py` to regenerate the model using updated datasets.

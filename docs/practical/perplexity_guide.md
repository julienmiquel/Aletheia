# Practical Guide: Perplexity Calculator

## 1. Overview

The `PerplexityCalculator` computes the perplexity of a text using a fixed sliding-window approach with a pre-trained GPT-2 model. It serves as the "basement-level" statistical detector.

## 2. API Reference

### 2.1 Initialization
```python
from ia_detector.perplexity import PerplexityCalculator

# Initializes GPT-2 (Medium) by default
ppl_calc = PerplexityCalculator()
```

### 2.2 Calculation
```python
score = ppl_calc.calculate(text)
```
-   **Input**: `text` (str).
-   **Output**: `float` (The PPL value).

## 3. Interpretation
-   **PPL < 30**: High probability of Machine Generation (Text is "too predictable").
-   **PPL 30 - 60**: Ambiguous / Simple Human or Scientific writing.
-   **PPL > 80**: High probability of Human Generation (Text is "complex/surprising").

## 4. Implementation Details
The calculator uses a sliding window (stride=512) to handle texts longer than the model's context.

```python
# Simplified Logic
encodings = tokenizer(text, return_tensors="pt")
for i in range(0, encodings.input_ids.size(1), stride):
    # ... compute likelihood ...
```

## 5. Known Issues
-   **False Positives**: Legal texts, technical manuals, and repetitive lists often have naturally low PPL despite being human-written.
-   **False Negatives**: Creative AI writing (high temperature) can artificially inflate PPL.

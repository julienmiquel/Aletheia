# Theoretical Framework: TF-IDF Stylometry

## 1. Concept: Stylistic Signatures

While LLMs are trained on massive corpora, their fine-tuning (RLHF) imprints specific stylistic biases. They tend to overuse "academic" transition words and avoid colloquialisms to sound helpful and harmless.

**TF-IDF (Term Frequency - Inverse Document Frequency)** allows us to detect these signatures by weighing terms that are frequent in AI output but rare in general human discourse.

## 2. Mathematical Formulation

For a term $t$ in a document $d$:

$$ TF(t, d) = \frac{\text{count}(t, d)}{\text{total terms in } d} $$

$$ IDF(t) = \log \frac{N}{| \{d \in D : t \in d\} |} $$

$$ \text{TF-IDF}(t, d) = TF(t, d) \cdot IDF(t) $$

### 2.1 The AI Signature
Empirical analysis reveals that RLHF models have high TF-IDF scores for:
-   **Connectives**: "However," "Furthermore," "In conclusion," "Crucially."
-   **Hedges**: "It is important to note," "While x, y..."
-   **Generic Adjectives**: "Vibrant," "Bustling," "tapestry."

### 2.2 Classification
We feed the TF-IDF vectors (N-grams where $N=1,2$) into a Logistic Regression classifier trained on a balanced dataset of Human (IMDb/WikiText) vs. AI (Alpaca/Gemini) text.

$$ P(AI | d) = \sigma(W^T \cdot \vec{x}_{tfidf} + b) $$

## 3. Advantages over Deep Learning
-   **Transparency**: We can inspect the weights $W$ to see exactly *which words* trigger the detector.
-   **Speed**: Vectorization is $O(N)$ and inference is extremely fast compared to Transformers.

---
**References**
[1]  Jurafsky & Martin, "Speech and Language Processing," Chapter 6.

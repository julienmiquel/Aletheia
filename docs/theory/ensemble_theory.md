# Theoretical Framework: Ensemble Detection and Sensor Fusion

## 1. The Condorcet Jury Theorem

The core philosophy of the Ensemble Detector relies on the **Condorcet Jury Theorem**: if each individual independent classifier (juror) has a probability $p > 0.5$ of being correct, the probability that the majority vote is correct increases with the number of jurors, approaching 1.

Our "jurors" are:
1.  **Perplexity**: Statistical (Predictability).
2.  **Burstiness**: Structural (Variance).
3.  **GLTR**: Visual (Rank Distribution).
4.  **TF-IDF**: Stylistic (Vocabulary).
5.  **LLM Judge**: Semantic (Hallucination/Nuance).

## 2. Sensor Fusion Architecture

We employ a **Stacked Generalization** (Stacking) approach.

```mermaid
graph TD
    Input[Input Text] --> PPL[Perplexity <br/>(Statistical)]
    Input --> Burst[Burstiness <br/>(Structural)]
    Input --> GLTR[GLTR <br/>(Visual)]
    Input --> TFIDF[TF-IDF <br/>(Stylistic)]
    Input --> Semantic[Semantic <br/>(Cognitive)]
    
    PPL --> Fusion[Ensemble / Meta-Classifier]
    Burst --> Fusion
    GLTR --> Fusion
    TFIDF --> Fusion
    Semantic --> Fusion
    
    Fusion --> Verdict{Final Verdict}
    Verdict -->|Score > 50%| AI[AI-Generated]
    Verdict -->|Score < 50%| Human[Human-Written]
    
    style Fusion fill:#f9f,stroke:#333,stroke-width:2px
    style AI fill:#ffdede,stroke:#f00
    style Human fill:#deffde,stroke:#0f0
```

### 2.1 Feature Vector Extraction
For a given input text $x$, we extract a feature vector $V$:
$$ V(x) = [ PPL(x), Burst(x), F_{green}(x), P_{tfidf}(x), P_{judge}(x) ] $$

### 2.2 The Meta-Classifier
A meta-classifier $M$ (Random Forest or Logistic Regression) maps $V(x)$ to a final probability $y$.

$$ y = M(V(x)) $$

This allows the system to learn non-linear relationships. For example:
-   If $PPL$ is low (indicating AI) but $Burst$ is high (indicating Human), the meta-classifier learns which signal is dominant for the specific context (e.g., highly bursty AI models like Gemini 3).

## 3. Heuristic Fallback

In the absence of a trained meta-classifier, we use a **Weighted Linear Combination**:

$$ Score = \frac{\sum w_i \cdot NormalizedScore(Feature_i)}{\sum w_i} $$

Weights $w_i$ are determined empirically based on the False Positive Rate of each individual metric.

---
**References**
[1] Wolpert, D. H. (1992). "Stacked generalization." Neural networks.

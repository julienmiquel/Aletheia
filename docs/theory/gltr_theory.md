# Theoretical Framework: Giant Language Model Test Room (GLTR)

## 1. Concept: Token Rank Distribution

GLTR (Giant Language Model Test Room) is based on the forensic hypothesis that AI models, which are statistical engines trained to minimize cross-entropy, disproportionately select high-probability tokens from the "head" of the distribution [1].

### 1.1 The Top-K Visualizer
For any given context $C$, a language model predicts a probability distribution over the vocabulary $V$. We rank all tokens $w \in V$ by $P(w|C)$.

-   **Green Zone (Top 10)**: The most likely words.
-   **Yellow Zone (Top 100)**: Likely but slightly more specific.
-   **Red Zone (Top 1000)**: Creative or unusual choices.
-   **Purple Zone (> 1000)**: Extremely rare or contextually surprising words.

### 1.2 The Forensic Signature
-   **AI Text**: Heavily dominated by Green tokens. The "path of least resistance" minimizes surprise.
-   **Human Text**: A "peppered" distribution. Humans frequently dip into the Red and Purple zones to express novelty, specific names, or complex ideas.

## 2. Mathematical Metric: Green Fraction
We define the **Green Fraction** ($F_g$) as:

$$ F_g = \frac{\text{Count}(rank \le 10)}{N_{tokens}} $$

An $F_g > 0.65$ is highly correlated with machine generation.

## 3. Vulnerabilities

Newer Sampling methods (Temperature setting > 1.0, Nucleus Sampling with high $p$) allow models to select from Yellow/Red zones occasionally. However, doing so without losing coherence is difficult for models, often resulting in "hallucinations" or non-sequiturs, which are then detected by the Semantic Consistency Analyzer (see `docs/theory/semantic_consistency_framework.md`).

---
**References**
[1] Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical Detection of Fake Text. ACL.

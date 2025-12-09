# Theoretical Framework: Burstiness and Structural Variance

## 1. Concept

While Perplexity measures the predictability of word choice, **Burstiness** measures the predictability of sentence structure and length. It relies on the observation that human writing is naturally non-uniform [1].

### 1.1 The "Staccato" of Thought
Human cognition is dynamic. We mix short, punchy declarative sentences with long, meandering explanatory clauses.
*   "Ideally, truth is a simple thing. But getting there? That's the hard part, involving nuanced layers of investigation, skepticism, and proof."
    *   Sentence 1: Short.
    *   Sentence 2: Medium.
    *   Sentence 3: Long.

### 1.2 Mathematical Formulation
We quantify burstiness using the **Coefficient of Variation (CV)** of sentence lengths ($L$) and lexical diversity.

$$ \sigma_{L} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (l_i - \mu_{L})^2} $$
$$ Burstiness = \frac{\sigma_{L}}{\mu_{L}} $$

Where:
-   $l_i$: Length of sentence $i$.
-   $\mu_{L}$: Mean sentence length.
-   $\sigma_{L}$: Standard Deviation of sentence length.

High Burstiness (High CV) indicates significant variance, typical of human authors. Low Burstiness (Low CV, near 0) indicates a "flat," highly consistent robotic cadence.

## 2. LLM Patterning

LLMs, despite advancements, often default to a "Golden Mean" of sentence length (typically 15-20 words) to maximize safety and readability constraints reinforced during RLHF. This results in a monotonous rhythm:
*   "The quick brown fox jumps. The lazy dog sleeps there. The day is very sunny." (Low Burstiness)

## 3. Limitations

Advanced prompting ("Write with varying sentence lengths") can artificially induce burstiness. However, semantic burstiness (variance in idea density) is harder to fake.

---
**References**
[1]  Gao et al., "Comparing Human and Machine Text Generation," arXiv 2023.

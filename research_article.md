# Research Article: Analysis and Improvement of AI Text Detection

## 1. Abstract
This study analyzes the effectiveness of a multi-metric AI text detector against modern Large Language Models (LLMs), including Google's Gemini series. We evaluate the detector's initial performance, identify weaknesses against advanced models like Gemini 3 Pro, and demonstrate a significant improvement process involving data augmentation and re-training of specific feature detectors.

## 2. Methodology
The detector integrates four distinct analytical approaches:

### 2.1 Perplexity (Statistical)
**Theory**: Measures how predictable a text is to a language model (GPT-2). 
**Implementation**: A sliding window approach calculates the perplexity. Lower perplexity indicates higher predictability, characteristic of AI text.
**Observation**: AI text typically scores < 30, while human text varies widely (20-100+).

### 2.2 Burstiness (Structural)
**Theory**: Human writing exhibits "bursts" of complex and simple sentences. AI writing tends to be more consistent and "flat".
**Implementation**: We analyze the variation in sentence length and lexical diversity.
**Observation**: Human text has high burstiness (> 0.4); AI text has low burstiness (< 0.2).

### 2.3 GLTR (Visual Forensics)
**Theory**: The Giant Language Model Test Room (GLTR) analyzes the rank of each token.
**Implementation**: We calculate the fraction of words falling into the "Green" (Top-10 predicted) bucket.
**Observation**: AI text is dominated by Green tokens (> 60%), while human text contains more unpredictable (Red/Purple) choices.

### 2.4 TF-IDF (Stylistic)
**Theory**: Support Vector Machines or Logistic Regression on N-gram frequencies can detect specific stylistic signatures (e.g., overuse of "delve", "crucial", etc.).
**Implementation**: A baseline Logistic Regression model trained on IMDb (Human) and Alpaca (AI) datasets.

### 2.5 Ensemble Learning (Meta-Classifier)
**Theory**: Individual metrics (PPL, Burstiness, etc.) have strengths and weaknesses. A meta-classifier can learn non-linear combinations of these features to improve overall accuracy.
**Implementation**: We extract a 5-dimensional feature vector for each text: 
`[Perplexity, Burstiness Coeff, Lexical Entropy, GLTR Green Fraction, TF-IDF Probability]`
and train a **Random Forest Classifier** to predict the final verdict.

### 2.6 LLM as a Judge (Semantic)
**Theory**: Modern LLMs (e.g., Gemini) have advanced semantic understanding and can detect "hallucination-like" patterns or lack of personal nuance that statistical methods miss.
**Implementation**: We prompt `gemini-2.0-flash-exp` to analyze the text for specific AI artifacts (repetitive structure, lack of anecdotes) and provide a probability score.

## 3. Initial Results (Baseline)
Before improvement, the detector was benchmarked against:
- **Human (IMDb)**: 90% Detection Accuracy.
- **AI (Alpaca/LLaMA)**: 100% Detection Accuracy.
- **AI (Gemini 2.0 Flash)**: 100% Detection Accuracy.
- **AI (Gemini 3 Pro Preview)**: **50% Detection Accuracy**.

**Analysis**: Gemini 3 Pro Preview generated text with higher burstiness and lower predictability, effectively mimicking human nuances and bypassing the heuristic thresholds of the statistical metrics.

## 4. Improvement Process
To address the weakness against Gemini 3 Pro, we implemented a data-driven enhancement strategy:

1.  **Data Harvesting (Enriched)**: We harvested **~900 diverse samples** from `gemini-3-pro-preview` using prompt engineering that covers technical, creative, adversarial, and professional writing styles.
2.  **Dataset Balancing**: We combined IMDb reviews and WikiText to ensure a robust human baseline.
3.  **Ensemble Training**: Instead of heuristic thresholds, we trained a Random Forest model on the extracted feature vectors of the balanced dataset.

## 5. Final Results (Post-Optimization)
After retraining the TF-IDF model with 100 new Gemini samples and 100 WikiText samples:

### Performance on Gemini 3 Pro Preview
- **Detection Rate**: ~50% (Mixed results with Ensemble).
- **LLM Judge**: **High Accuracy (85%+)** on generic AI text, identifying semantic patterns like "lack of anecdotes" and "balanced tone".
- **Average Confidence**: ~48-56% (Ensemble), 85% (Judge).
- **Observation**: While statistical metrics struggle with Gemini 3 Pro's high burstiness, the **LLM Judge** provides a critical semantic layer that successfully flags "human-like" but generic content.

### Engineering & Optimization
To support these advanced methods, we implemented:
- **Result Caching**: SQLite-based caching for expensive metrics (GLTR, PPL), reducing re-run time by 90%.
- **Dynamic Thresholding**: Algorithm adapts to text length, dampening confidence for short (< 400 char) inputs to avoid false positives.

### Performance on Gemini 2.5 Flash / Pro
- **Detection Rate**: **100%**.
- **Confidence**: High (> 99%).
- **Observation**: Surprisingly, the intermediate Gemini 2.5 models are significantly easier to detect than their predecessor (2.0) or successor (3.0), likely due to a more standardized, less bursty generation style.

### Performance on Gemini 2.0 Flash
- **Detection Rate**: **~25%** (High Evasion).
- **Observation**: Gemini 2.0 Flash generates long, highly coherent, and structurally varied text that effectively mimics human statistical patterns, making it the most elusive model in our suite.

### Performance on Legacy AI (Alpaca)
- **Detection Rate**: 100%.
- **Confidence**: High (> 80%).

## 6. Conclusion
The "Data Augmentation & Retraining" strategy successfully hardened the detector against standard AI models but revealed a significant challenge with next-generation models like **Gemini 3 Pro**.

**Key Takeaways**:
1.  **Diminishing Returns**: Adding more TF-IDF training data yields diminishing returns against models that effectively emulate human variability.
2.  **Next-Gen Evasion**: Models optimized for "human-like" writing style (high burstiness) effectively break current statistical detection paradigms.
3.  **Future Directions**:
    - **Adversarial Training**: Using GANs to generate hard-to-detect samples for training.
    - **Semantic Analysis**: Moving beyond statistical metrics to analyze logical consistency and hallucination patterns.

## 7. New Research Area: Watermarking & Semantic Forensics
To improve results against models like Gemini 3 Pro, we suggest two key research areas:
1.  **Robust Watermarking**: Investigating the presence of subtle, mathematically verifiable watermarks (e.g., SynthID) that persist through edits.
2.  **Semantic Coherence Analysis**: AI models often exhibit "hallucinations" or subtle logical inconsistencies over long contexts. Developing a "Semantic Consistency Scorer" could reveal AI artifacts that statistical metrics miss.

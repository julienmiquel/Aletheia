# Deconstructing the Machine: How We Built an AI Detector to Catch Gemini 3 Pro

*By Julien Miquel*

![AI Detector Cover Image](docs/ai_detector_cover.png)

## 1. Abstract
The arms race between AI text generation and detection is escalating. This study analyzes the effectiveness of a multi-metric AI text detector against modern Large Language Models (LLMs), including Google's Gemini series. We evaluate the detector's initial performance, identify weaknesses against advanced models like Gemini 3 Pro, and demonstrate a significant improvement process involving data augmentation and re-training of specific feature detectors.

Crucially, **we provide the code**. This article breaks down the implementation of Perplexity, Burstiness, GLTR, and Semantic Analysis to help you build your own forensic toolkit.

---

## 2. Methodology & Forensics
The detector integrates four distinct analytical approaches, mimicking a **Cost-Sensitive Cascade**: we start with cheap statistical metrics and move to expensive semantic analysis only when necessary.

### 2.1 Perplexity (Statistical)
**Theory**: Measures how predictable a text is to a language model (GPT-2).
*   **Observation**: AI text typically scores **< 30**, while human text varies widely (**20-100+**).
*   **Implementation**: A sliding window approach calculates the perplexity. Lower perplexity indicates higher predictability, characteristic of AI text.

### 2.2 Burstiness (Structural)
**Theory**: Human writing exhibits "bursts" of complex and simple sentences. AI writing tends to be more consistent and "flat".
*   **Observation**: Human text has high burstiness (**> 0.4**); AI text has low burstiness (**< 0.2**).
*   **Implementation**: We analyze the variation in sentence length and lexical diversity using statistical dispersion metrics.

### 2.3 GLTR (Visual Forensics)
**Theory**: The Giant Language Model Test Room (GLTR) analyzes the rank of each token.
*   **Observation**: AI text is dominated by "Green" (Top-10 predicted) tokens (**> 60%**). Human text contains frequent unpredictable (Red/Purple) choices.
*   **Implementation**: We calculate the fraction of words falling into the Top-10 prediction bucket of a standard model.

### 2.4 TF-IDF (Stylistic)
**Theory**: Support Vector Machines or Logistic Regression on N-gram frequencies can detect specific stylistic signatures (e.g., overuse of "delve", "crucial", "tapestry").
*   **Implementation**: A baseline Logistic Regression model trained on IMDb (Human) and Alpaca (AI) datasets using unigrams and bigrams.

### 2.5 Ensemble Learning (Meta-Classifier)
**Theory**: Individual metrics have weaknesses. A meta-classifier learns non-linear combinations of features to improve accuracy.
*   **Implementation**: We extract a 5-dimensional feature vector for each text:
    `[Perplexity, Burstiness Coeff, Lexical Entropy, GLTR Green Fraction, TF-IDF Probability]`
    and train a **Random Forest Classifier** to predict the final verdict.

---

## 3. Initial Results (Baseline)
Before our improvement process, the detector was benchmarked against standard datasets:

*   **Human (IMDb)**: 90% Detection Accuracy.
*   **AI (Alpaca/LLaMA)**: 100% Detection Accuracy.
*   **AI (Gemini 2.0 Flash)**: 100% Detection Accuracy.
*   **AI (Gemini 3 Pro Preview)**: **50% Detection Accuracy**.

**Analysis**: Gemini 3 Pro Preview generated text with significantly higher burstiness and lower predictability, effectively mimicking human nuances and bypassing the heuristic thresholds of the statistical metrics.

---

## 4. Code Implementation
Here is how we implemented the core metrics.

### Metric 1: Perplexity
We use a small model like `gpt2` to measure this. We slide a window across the text and calculate the negative log-likelihood of each token.

```python
def calculate_perplexity(self, text, stride=512):
    encodings = self.tokenizer(text, return_tensors='pt')
    seq_len = encodings.input_ids.size(1)
    
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    # Sliding window to handle long texts
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + self.max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
        target_ids = input_ids.clone()
        
        target_ids[:, :-trg_len] = -100 
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            num_valid_tokens = (target_ids != -100).sum().item()
            
            nll_sum += neg_log_likelihood * num_valid_tokens
            n_tokens += num_valid_tokens
            
        prev_end_loc = end_loc
        if end_loc == seq_len: break
    
    return torch.exp(nll_sum / n_tokens).item()
```

### Metric 2: Burstiness (The Rhythm)
We use `spaCy` to segment sentences and calculate the Coefficient of Variation (CV).

```python
def analyze_burstiness(self, text):
    doc = self.nlp(text)
    sentences = list(doc.sents)
    if not sentences: return 0
    
    lengths = [len(sent) for sent in sentences]
    mean_len = np.mean(lengths)
    std_dev = np.std(lengths)
    
    # Burstiness = CV (Sigma / Mu)
    return std_dev / mean_len if mean_len > 0 else 0
```

### Metric 3: GLTR (Visualizing Probability)
We analyze the rank of every token in the text against the model's predictions.

```python
def analyze_gltr(self, text):
    tokens = self.tokenizer.encode(text)
    input_ids = torch.tensor([tokens]).to(self.device)
    
    with torch.no_grad():
        outputs = self.model(input_ids)
        logits = outputs.logits[0, :-1, :] 
        
        results = []
        targets = tokens[1:]
        
        for i, target_id in enumerate(targets):
            probs = F.softmax(logits[i], dim=-1)
            target_prob = probs[target_id]
            rank = (probs > target_prob).sum().item() + 1
            
            if rank <= 10: bucket = "Green"
            elif rank <= 100: bucket = "Yellow"
            elif rank <= 1000: bucket = "Red"
            else: bucket = "Purple"
            
            results.append({"token": target_id, "bucket": bucket})
            
    green_count = sum(1 for r in results if r['bucket'] == "Green")
    return green_count / len(results)
```

### Metric 4: TF-IDF (Stylistic Fingerprinting)
We use a simple Logistic Regression pipeline to catch common AI-isms.

```python
def train_tfidf(self, human_texts, ai_texts):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42))
    ])
    
    corpus = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    
    pipeline.fit(corpus, labels)
```

---

## 5. Improvement Process
To address the weakness against Gemini 3 Pro, we implemented a data-driven enhancement strategy:
1.  **Data Harvesting**: We harvested **~900 diverse samples** from `gemini-3-pro-preview` using prompt engineering that covers technical, creative, adversarial, and professional writing styles.
2.  **Dataset Balancing**: We combined IMDb reviews and WikiText to ensure a robust human baseline.
3.  **Ensemble Training**: instead of heuristics, we trained the Random Forest model on the extracted feature vectors of the balanced dataset.

---

## 6. The Semantic Solution: LLM Judge
Statistical methods hit a wall with Gemini 3. To catch a smart AI, we need a smarter AI (Semantic Analysis).

**Theory**: Modern LLMs have advanced semantic understanding and can detect "hallucination-like" patterns, lack of personal nuance, or excessive "balance" that statistical methods miss.

```python
def evaluate_with_judge(self, text):
    prompt = f"""
    Analyze the text for:
    1. Absence of personal nuance or specific, verifiable anecdotes.
    2. Overuse of transition words.
    3. Generic, "safe" opinions.

    Text: {text[:4000]}
    
    Return JSON: {{"ai_probability": 0-100, "reasoning": "..."}}
    """
    # ... call Gemini API ...
    return score
```

---

## 7. Final Results (Post-Optimization)

After retraining and integrating the Semantic Judge, our results shifted dramatically.

### Performance on Gemini 3 Pro Preview
*   **Detection Rate**: **~50%** (Mixed results with Ensemble).
*   **LLM Judge**: **High Accuracy (85%+)**, identifying semantic patterns like "lack of anecdotes" and "balanced tone".
*   **Observation**: While statistical metrics struggle with Gemini 3 Pro's high burstiness, the **LLM Judge** provides a critical semantic layer.

### Performance on Gemini 2.0 Flash
*   **Detection Rate**: **~25%** (High Evasion).
*   **Observation**: Gemini 2.0 Flash is the most elusive model. It generates long, highly coherent, and structurally varied text that effectively mimics human statistical patterns.

### Performance on Gemini 2.5 Flash / Pro
*   **Detection Rate**: **100%**.
*   **Observation**: Surprisingly, the intermediate Gemini 2.5 models are significantly easier to detect than their predecessor (2.0) or successor (3.0), likely due to a more standardized generation style.

---

## 8. Conclusion
The "Data Augmentation & Retraining" strategy successfully hardened the detector against standard AI models but revealed a significant challenge with next-generation models like **Gemini 3 Pro**.

**Key Takeaways**:
1.  **Diminishing Returns**: Adding more TF-IDF training data yields diminishing returns against models that effectively emulate human variability.
2.  **Next-Gen Evasion**: Models optimized for "human-like" writing style (high burstiness) effectively break current statistical detection paradigms.
3.  **The Semantic Future**: Future detection must rely on **Semantic Consistency Scoring** and **Robust Watermarking** rather than simple statistical counters.

---

## 9. Code Availability
The complete source code, datasets, and benchmark scripts used in this study are publicly available for reproducibility and further research:
**GitHub Repository**: [https://github.com/julienmiquel/Aletheia](https://github.com/julienmiquel/Aletheia)

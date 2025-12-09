# AI Detector Improvements TODO

## Data Quality & Diversity
- [ ] **Diverse Prompt Engineering**: Expand prompt set beyond basic 25. Include:
    - Technical writing (coding, docs)
    - Creative writing (poetry, scripts)
    - Professional communication (emails, reports)
    - Academic writing (essays, abstracts)
- [ ] **Adversarial Samples**: Generate text with prompts like "Write this in a human style", "Avoid AI patterns", or "Use slang and typos".
- [ ] **Expanded Human Dataset**: IMDb is biased towards reviews. Add:
    - Wikipedia articles (WikiText)
    - Project Gutenberg (Literature)
    - Reddit comments (Conversational)

## Algorithm Enhancements
- [ ] **Trainable Ensemble**: Replace the heuristic `get_combined_score` with a Logistic Regression or SVM classifier trained on the 4 metric outputs (PPL, Burst, GLTR, TFIDF).
- [ ] **Dynamic Thresholding**: Adjust detection thresholds based on input length (shorter text is harder).

## Engineering
- [ ] **Parallel Generation**: Speed up `generate_training_data.py` using `asyncio` or thread pools.
- [ ] **Caching**: Cache Perplexity/GLTR results to speed up repeated testing.

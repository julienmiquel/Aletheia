# Aletheia: Implementation Roadmap

## Completed Milestones
- [x] **Core Detectors**: Perplexity (GPT-2), Burstiness (SpaCy), GLTR (Rank Analysis), TF-IDF (Stylometry).
- [x] **Ensemble Fusion**: Stacked meta-classifier improving accuracy over heuristics.
- [x] **Engineering**: Parallel data generation, SQLite caching, centralized config.
- [x] **Streamlit UI**: Interactive dashboard with metric visualization.
- [x] **Semantic Analysis V1**: LLM-based logical consistency check + **Quantitative Coherence Metric** (Cosine Similarity).

## Phase 1: Robustness (Current Focus)
- [ ] **Adversarial Benchmark**: Create a dataset of "Human-edited AI" and "Paraphrased AI" to test resistance to evasion.
- [ ] **Dynamic Thresholding**: Implement thresholds that adapt to text length (currently static).
- [ ] **False Positive Minimization**: Calibrate ensemble to prioritize precision on human text (avoid accusing humans).

## Phase 2: Neural & Structural Features
- [ ] **Structural Syntax**: Implement Graph-based syntactic dependency features (as per SILTD).
- [ ] **Surrogate PPL**: Use T5/masked-language-models for "perturbation discrepancy" (DetectGPT approach) instead of raw PPL.
- [ ] **Vector-Based Style**: Train a classifier on sentence embedding trajectories (Narrative Flow).

## Phase 3: Deployment & Scale
- [ ] **API Service**: Containerize (Docker) and expose via FastAPI.
- [ ] **Chrome Extension**: Lightweight client using the API.
- [ ] **Scale**: Optimize for high-throughput batch processing.

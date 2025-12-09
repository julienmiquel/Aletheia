# AI Text Detector

A comprehensive tool to detect AI-generated text using multiple metrics: **Perplexity** (GPT-2), **Burstiness**, **GLTR** (Giant Language Model Test Room), and **TF-IDF** n-gram analysis.

## Features
- **Multi-Metric Analysis**: Combines structural (Burstiness), statistical (Perplexity, GLTR), and stylistic (TF-IDF) signals.
- **Combined Scoring**: Heuristic algorithm to produce a single "AI Likelihood" percentage.
- **CLI Tool**: Easy-to-use command line interface.
- **Verification Scripts**: Automated testing against Human (IMDb), Alpaca, and Gemini models.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. (Optional) Create a `.env` file for Gemini testing:
   ```
   GEMINI_API_KEY=your_key_here
   ```

## Usage

### Single Text Analysis
```bash
python main.py "Your text here"
# OR
python main.py path/to/file.txt
```

### Run Benchmarks
1. **Dataset Test (Human vs Alpaca)**:
   ```bash
   python fetch_and_test.py
   ```
2. **Gemini Live Test**:
   ```bash
   python verify_gemini_detection.py
   ```

## Benchmarking Results

We tested the detector against various sources. A score **> 50%** is classified as AI.

| Source | Model / Dataset | Avg. AI Likelihood | Accuracy | Verdict |
|--------|-----------------|--------------------|----------|---------|
| **Human** | IMDb Reviews | **~35-45%** | 90% | ✅ Human |
| **AI** | Alpaca (LLaMA) | **~75-80%** | 100% | ✅ AI |
| **AI** | Gemini 2.5 Flash | **~99%** | 100% | ✅ AI (Easily Detected) |
| **AI** | Gemini 2.5 Pro | **~76%** | 75% | ✅ AI |
| **AI** | Gemini 3 Pro Preview | **~77%** | 66% | ⚠️ Mixed Results |
| **AI** | Gemini 2.0 Flash | **~32%** | 25% | ❌ Highly Evasive |

> **Note**: *Gemini 2.0 Flash* creates long, highly coherent, and bursty text that effectively evades statistical detection. *Gemini 2.5 Flash*, conversely, is easily flagged.

## Detailed Scoring Logic
- **Ensemble (Meta-Classifier)**: A Random Forest model aggregates 5 features (Perplexity, Burstiness, Entropy, GLTR, TF-IDF) to produce the final "AI Likelihood".
- **Perplexity**: Measures predictability (Lower = AI).
- **Burstiness**: Measures structural variability (Lower = AI).
- **GLTR**: Measures token rank confidence (Higher Green = AI).
- **TF-IDF**: Detects stylistic fingerprints.
- **Burstiness**: Measures sentence structure variability. Low variability -> AI.
- **GLTR**: Measures fraction of tokens in the "Green" (Top-10) probability bucket. High green -> AI.
- **TF-IDF**: Detects specific n-gram patterns common in AI training data.

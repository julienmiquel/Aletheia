# Aletheia: Advanced AI Text Forensics & Detection Framework

> *Aletheia (Greek: ἀλήθεια) - The state of not being hidden; disclosure; truth.*

**Aletheia** is a PhD-level research framework designed to detect AI-generated text through **multi-modal sensor fusion**. Unlike simple statistical counters, it combines information theory, structural linguistics, and semantic consistency analysis to identify the "watermarks" left by Large Language Model optimization.

![Status](https://img.shields.io/badge/Status-Research_Preview-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔬 The Forensic Architecture

Aletheia operates on the **Condorcet Jury Theorem**: aggregating multiple independent "jurors" (detectors) to maximize verdict reliability.

| Detector        | Type                   | Theory                 | Target Artifact                             |
| :-------------- | :--------------------- | :--------------------- | :------------------------------------------ |
| **Statistical** | `PerplexityCalculator` | Information Entropy    | Optimization pressure / low surprise.       |
| **Structural**  | `BurstinessAnalyzer`   | Linguistic Variance    | "Flat" sentence rhythm (RLHF smoothness).   |
| **Visual**      | `GLTRAnalyzer`         | Rank Distribution      | Dominance of Top-K ("Green") tokens.        |
| **Stylistic**   | `TfidfDetector`        | N-Gram Frequency       | Generic/Academic "accents" (e.g., "delve"). |
| **Semantic**    | `SemanticConsistency`  | **Meta-Cognition**     | Context decay & hallucinations. **(New)**   |
| **Fusion**      | `EnsembleDetector`     | Stacked Generalization | Meta-classification of all signals.         |

➡️ **[Read the Theoretical Frameworks](docs/theory/)**

---

## 🚀 Installation

```bash
git clone https://github.com/julienmiquel/Aletheia.git
cd aletheia-detector
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

*(Optional)* Create a `.env` file for Semantic Analysis features:
```bash
GEMINI_API_KEY=your_key_here
```

---

## 💻 Usage

### 1. The Ensemble (Recommended)
The `EnsembleDetector` orchestrates all sensors automatically.

```python
from ia_detector.ensemble import EnsembleDetector

# Initialize the forensic suite
detector = EnsembleDetector()

text = "In conclusion, it is important to delve into the tapestry..."

# Run analysis (use_semantic=True enables the LLM Judge)
report = detector.predict(text, use_semantic=True)

print(f"AI Probability: {report['combined_score']}%")
print(f"Verdict: {report['verdict']}")
```

### 2. Streamlit Dashboard (UI)
The easiest way to use Aletheia:
```bash
streamlit run app.py
```

### 3. Command Line Interface
```bash
python main.py "Your suspicious text here"
```

### 4. Research & Training Scripts
The `scripts/` directory contains tools for data generation, training, and benchmarking:

*   **Benchmark Suite**: Evalute detectors on IMDb vs Alpaca/Gemini.
    ```bash
    python scripts/benchmark_suite.py --n 50
    ```
*   **Generate Data**: Create adversarial training samples using Gemini (Requires API Key).
    ```bash
    python scripts/generate_training_data.py
    ```
*   **Train Ensemble**: Retrain the meta-classifier on new data.
    ```bash
    python scripts/train_ensemble.py
    ```

---

## 📊 Benchmarks

We maintain a rigorous benchmark suite against state-of-the-art models (Gemini 3 Pro, GPT-4).

| Dataset          | Human Baseline | AI Baseline | Detection Rate           |
| :--------------- | :------------- | :---------- | :----------------------- |
| **IMDb (Human)** | ~35% Score     | N/A         | **90%** (True Negative)  |
| **Alpaca (AI)**  | N/A            | ~80% Score  | **100%** (True Positive) |
| **Gemini 3 Pro** | N/A            | ~65% Score  | **~75%** (Evolving)      |

➡️ **[View Full Benchmark Report](docs/benchmarks/comprehensive_results.md)**

---

## 📂 Documentation

-   **Theory**: Mathematical underpinnings of [Perplexity](docs/theory/perplexity_theory.md), [Burstiness](docs/theory/burstiness_theory.md), and [GLTR](docs/theory/gltr_theory.md).
-   **Practice**: API Guides for [Ensemble](docs/practical/ensemble_guide.md) and [Semantic Analysis](docs/practical/semantic_consistency_guide.md).

## 📜 Citation

If you use Aletheia in your research, please cite the repository:
```
@software{aletheia2025,
  author = {Miquel, Julien},
  title = {Aletheia: Multi-Modal AI Text Detection Framework},
  year = {2025},
  url = {https://github.com/julienmiquel/Aletheia}
}
```

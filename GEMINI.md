# Aletheia Developer Guide

This document provides instructions and best practices for developing **Aletheia**, the AI Text Forensics & Detection Framework.

## 1. Project Overview

Aletheia is a research-grade framework designed to detect AI-generated text using **multi-modal sensor fusion**. Unlike simple statistical counters, it combines information theory, structural linguistics, and semantic consistency analysis.

-   **Core Philosophy**: The "Condorcet Jury Theorem" — aggregating multiple weak learners (detectors) to create a strong verdict.
-   **Architecture**: Modular "Sensor" pattern with a meta-classifier (Ensemble).

## 2. System Architecture

The codebase is organized into a core library and a presentation layer.

-   **`ia_detector/`**: Core Analysis Package.
    -   **`ensemble.py`**: The Orchestrator. Manages all sub-detectors and aggregates results.
    -   **`perplexity.py`**: Calculates text predictability (Entropy).
    -   **`burstiness.py`**: Analyzes sentence length variation (Rhythm).
    -   **`gltr.py`**: Analyzes token rank distribution (Visual Forensics).
    -   **`llm_judge.py`**: Semantic consistency check using Google Gemini.
-   **`app.py`**: Streamlit-based User Interface.

## 3. Technologies

-   **Runtime**: Python 3.12+
-   **ML/NLP**: `transformers` (Hugging Face), `torch` (PyTorch), `spacy`
-   **GenAI**: `google-genai` (Gemini API)

---

## 4. Coding Guidelines

### General Principles
-   **Lazy Loading is Mandatory**: This project uses heavy models (GPT-2, BERT). **NEVER** load these in `__init__`. Always use properties with lazy initialization.
-   **Type Hinting**: Use standard Python type hints for all function signatures.
-   **Robustness**: Detectors must fail gracefully (return `None`) if a model fails to load.

### Python Style Guide
- Follow **PEP 8**.
- Use **snake_case** for functions/variables and **PascalCase** for classes.
- Use **UPPERCASE** for constants (e.g., `ENSEMBLE_MODEL_PATH`).

---

## 5. Gemini API Best Practices

This section outlines the standards for interacting with Gemini models, particularly for the `LLMJudge` and `SemanticConsistency` modules.

### Client Initialization
- **Reusability**: Initialize the `genai.Client` once and reuse it.
- **Vertex AI Priorities**: Prefer Vertex AI (`vertexai=True`) for production stability, sourcing project/location from `os.environ` or config.

### Structured Output (Critical)
To ensure reliable analysis, **always** request JSON output using `pydantic` schemas or explicit MIME types.

1.  **Define Schema**: Create a Pydantic model for the expected output.
2.  **Configure Generation**: Use `response_schema` and `response_mime_type="application/json"`.

### Example: The LLM Judge Implementation

```python
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import os

# 1. Define Output Schema
class ForensicVerdict(BaseModel):
    is_ai: bool = Field(description="True if the text is likely AI-generated")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the forensic findings")

# 2. Lazy Client Initialization
class LLMJudge:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location="us-central1"
            )
        return self._client

    def eval(self, text):
        # 3. Secure Prompting
        prompt = f"Analyze this text for AI artifacts: {text[:4000]}"
        
        # 4. Structured Generation
        response = self.client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ForensicVerdict,
                temperature=0.0
            )
        )
        return response.parsed # Returns a ForensicVerdict object
```

### Prompt Management
- **Separation**: Do not hardcode long prompts in logic methods. Store them in a dedicated `prompts` module or constant.
- **Injection**: Use f-strings for dynamic data injection, ensuring inputs are truncated to avoid context window overflows.

---

## 6. Setup and Installation

1.  **Environment**:
    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    ```
2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
3.  **Running the App**:
    ```bash
    streamlit run app.py
    ```

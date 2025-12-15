# Aletheia: Digital Forensics for the AI Era

**Video Link:** https://youtu.be/Y8EHyhkmRsk

**Statistical detection is dead. Welcome to the era of Digital Forensics.**

For years, we spotted AI by its "statistical fingerprints" — perfect grammar, predictable word choices, and flat sentence structures. But models like **Gemini 3 Pro** and **GPT-4** have learned to hide. They simulate human "burstiness," they use "imperfections," and they bypass traditional detectors with ease.

This video introduces **Aletheia**, a military-grade open-source forensic framework designed to catch the uncatchable.

To use the tool and follow along with the code:
👉 **GITHUB REPO:** https://github.com/julienmiquel/Aletheia

---

### 🚨 THE PROBLEM: Why Old Detectors Fail
Traditional AI detectors rely on simple metrics:
1.  **Perplexity**: Measuring how "surprised" a model is by the text.
2.  **Burstiness**: Checking for sentence length variation.
3.  **N-Grams**: Looking for overused words like "delve" or "tapestry."

Modern "Human-Parity" models (like Gemini 3) are optimized to defeat these metrics using Reinforcement Learning (RLHF). They act human. They write with rhythm. They pass the test.

---

### 🛡️ THE SOLUTION: Aletheia's Two-Tier Architecture
Aletheia abandons the "one-size-fits-all" approach for a defense-in-depth strategy:

**Level 1: The Patrol (Rapid Response)**
We still use fast statistics to catch the 90% of lazy bots and older models (ChatGPT 3.5, Llama 2).
*   **Perplexity Sensor**: Detecting low-entropy "smooth" text.
*   **Burstiness Analyzer**: Flagging monotonous sentence structures.
*   **GLTR (Giant Language Model Test Room)**: Visualizing the "Sea of Green" — highlighting tokens that were in the top-10 predicted probability.

**Level 2: The Expert (Semantic Investigation)**
This is the heavy artillery. When statistical scores are inconclusive (e.g., against Gemini 3), Aletheia escalates to Level 2.
*   **Semantic Consistency Analyzer**: We use a Meta-Cognitive agent to "read" the text for logical flaws.
*   **The Hallucination Loop**: AI models eventually lose track of their own lies. Our analyzer spots "Context Decay" — where the narrative logic fractures (e.g., an object changing properties, a timeline breaking).
*   **Adversarial Profiling**: Detecting the "Safe Fluff" — the lack of specific, falsifiable anecdotes that humans naturally include.

---

### 💻 HOW TO USE IT
You can run Aletheia on your own machine today.

**1. Installation**
```bash
git clone https://github.com/julienmiquel/Aletheia.git
cd aletheia-detector
pip install -e .
python -m spacy download en_core_web_sm
```

**2. Running a Quick Scan**
```bash
./run.sh app
```
This launches the **Streamlit Dashboard**, where you can paste text and watch the "Sea of Green" visualization light up AI-generated segments in real-time.

**3. Advanced: Red Team Mode**
Want to test your own detection skills? Use our adversarial generator to create "Humanized" AI attacks:
```bash
./run.sh adversarial
```

---

### 🔬 TECH STACK
Aletheia is built for researchers and developers:
*   **Core**: Python 3.12+
*   **NLP**: SpaCy, Transformers (Hugging Face), PyTorch
*   **GenAI**: Google Gemini API (for Semantic Analysis)
*   **Visualization**: Streamlit, Plotly, Mermaid.js
*   **Architecture**: Modular Sensor Fusion with Stacked Generalization (Condorcet Jury Theorem)

---

### 📚 CHAPTERS
0:00 - The Death of Statistical Detection
1:45 - Introducing Aletheia
3:20 - Level 1: The Statistical Patrol (Perplexity & GLTR)
5:10 - Demo: Detecting ChatGPT 3.5
6:45 - The Boss Fight: Gemini 3 Pro
8:30 - Level 2: Semantic Consistency & Hallucination Loops
10:15 - Code Walkthrough & Installation
12:00 - Conclusion: The Future of Digital Forensics

---

### 🔗 REFERENCES & CITATIONS
This project implements theories from:
*   *GLTR: Statistical Detection of Fake Text* (Gehrmann et al., ACL 2019)
*   *Stacked Generalization* (Wolpert, 1992)
*   *Google Gemini 1.5 Technical Report* (2024)

**Read the full documentation:**
https://github.com/julienmiquel/Aletheia/tree/main/docs

**License:** MIT License (Open Source)

---

### 🏷️ TAGS
#AIDetection #DigitalForensics #MachineLearning #CyberSecurity #NLP #GeminiPro #GPT4 #OpenSource #Python #ArtificialIntelligence #Deepfakes #LLM #ComputerScience #Tech #Coding #DataScience #AIEthics

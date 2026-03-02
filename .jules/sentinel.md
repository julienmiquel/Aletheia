## 2025-02-28 - Missing Input Length Limits (DoS Risk)
**Vulnerability:** The application accepted unbounded text inputs for expensive analysis (Perplexity, Neural inference, LLM evaluation).
**Learning:** Accepting arbitrarily long text strings in NLP forensic tools poses a Denial of Service (DoS) risk, as it allows attackers to consume excessive CPU, memory, and API resources.
**Prevention:** Always set `max_chars` or a specific length limit on input components (`st.text_area`, endpoints) connecting to expensive NLP operations.

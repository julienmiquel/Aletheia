## 2024-05-24 - Denial of Service (DoS) Risk with Unbounded Text Input
**Vulnerability:** The Streamlit application (`app.py`) allowed users to submit arbitrarily large text blocks via `st.text_area` without any length limits.
**Learning:** Because the application processes the text using computationally expensive operations (e.g., Perplexity, GLTR, semantic models), processing massive text inputs can overwhelm the application, causing CPU/Memory exhaustion and potential Denial of Service (DoS).
**Prevention:** Always enforce explicit length limits (e.g., `max_chars`) on user text inputs before passing them to expensive NLP operations.

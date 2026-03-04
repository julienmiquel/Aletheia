## 2025-02-28 - Missing Input Length Limits (DoS Risk)
**Vulnerability:** The application accepted unbounded text inputs for expensive analysis (Perplexity, Neural inference, LLM evaluation).
**Learning:** Accepting arbitrarily long text strings in NLP forensic tools poses a Denial of Service (DoS) risk, as it allows attackers to consume excessive CPU, memory, and API resources.
**Prevention:** Always set `max_chars` or a specific length limit on input components (`st.text_area`, endpoints) connecting to expensive NLP operations.

## 2025-02-28 - Cross-Site Scripting (XSS) via Unsanitized Verdict Strings
**Vulnerability:** The application rendered dynamic assessment fields (like the verdict string) via Streamlit's `st.markdown(..., unsafe_allow_html=True)` without HTML escaping.
**Learning:** Even internal labels or model outputs that are not strictly raw user text can introduce XSS if they can be manipulated (e.g., through adversarial prompts altering output verdicts) and are rendered with `unsafe_allow_html=True`.
**Prevention:** Always use `html.escape()` on any variable string embedded in HTML, even if it represents an application-generated classification label, before rendering it with `unsafe_allow_html=True`.

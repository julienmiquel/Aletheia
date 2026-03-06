## 2025-03-05 - Weak Hash Algorithm (MD5) for Cache Keys
**Vulnerability:** The application used `hashlib.md5()` to generate keys for cached analysis results, creating a risk of hash collision attacks and violating secure coding standards for cryptographic hashing.
**Learning:** Even for non-cryptographic purposes like cache keys, using explicitly weak algorithms like MD5 can trigger security alerts, introduce theoretical collision risks with adversarial input, and fail compliance checks.
**Prevention:** Use a cryptographically secure hash function like SHA-256 (`hashlib.sha256()`) by default for all hashing needs, even cache keys, unless performance requirements strictly necessitate a non-cryptographic hash (like MurmurHash or cityhash).

## 2025-02-28 - Missing Input Length Limits (DoS Risk)
**Vulnerability:** The application accepted unbounded text inputs for expensive analysis (Perplexity, Neural inference, LLM evaluation).
**Learning:** Accepting arbitrarily long text strings in NLP forensic tools poses a Denial of Service (DoS) risk, as it allows attackers to consume excessive CPU, memory, and API resources.
**Prevention:** Always set `max_chars` or a specific length limit on input components (`st.text_area`, endpoints) connecting to expensive NLP operations.

## 2025-02-28 - Cross-Site Scripting (XSS) via Unsanitized Verdict Strings
**Vulnerability:** The application rendered dynamic assessment fields (like the verdict string) via Streamlit's `st.markdown(..., unsafe_allow_html=True)` without HTML escaping.
**Learning:** Even internal labels or model outputs that are not strictly raw user text can introduce XSS if they can be manipulated (e.g., through adversarial prompts altering output verdicts) and are rendered with `unsafe_allow_html=True`.
**Prevention:** Always use `html.escape()` on any variable string embedded in HTML, even if it represents an application-generated classification label, before rendering it with `unsafe_allow_html=True`.

## 2025-03-05 - Information Leakage via Unhandled Exceptions in Streamlit
**Vulnerability:** The Streamlit application did not catch exceptions during deep analysis (e.g., `detector.predict()`), leading to raw Python stack traces being displayed directly in the user interface.
**Learning:** In Streamlit applications, unhandled exceptions render detailed stack traces in the browser, which can leak sensitive internal paths, logic, and environment details.
**Prevention:** Always wrap critical or external logic in a `try...except` block in Streamlit apps. Use `st.error()` to display a generic user-friendly message, log the actual error securely (e.g., `print(e)` or standard logging), and halt further problematic execution (e.g., using `st.stop()`).

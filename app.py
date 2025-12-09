import streamlit as st
import time
import os
import plotly.graph_objects as go
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Aletheia - AI Text Forensics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Aletheia" styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .verdict-human {
        color: #10B981;
        font-size: 2em;
        font-weight: bold;
    }
    .verdict-ai {
        color: #EF4444;
        font-size: 2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🔎 Aletheia")
st.sidebar.markdown("Advanced AI Text Forensics & Detection Framework")
st.sidebar.markdown("---")

st.sidebar.header("Configuration")
use_semantic = st.sidebar.checkbox("Enable Semantic Analysis (Gemini)", value=False, help="Incurs API cost but improves detection on recent models (Gemini 3, GPT-4).")

st.sidebar.markdown("---")
st.sidebar.info("Aletheia uses a multi-modal sensor fusion approach to detect AI artifacts in text.")

# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
    from ia_detector.ensemble import EnsembleDetector
    return EnsembleDetector()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.markdown("<h1 class='main-header'>Aletheia: AI Text Forensics</h1>", unsafe_allow_html=True)

# Input
text_input = st.text_area("Analysis Target", placeholder="Paste suspicious text here...", height=200)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    analyze_btn = st.button("Analyze Text", type="primary", use_container_width=True)

if analyze_btn and text_input:
    if len(text_input) < 50:
        st.warning("Text is too short for reliable analysis. Please provide at least 50 characters.")
    else:
        with st.spinner("Initializing Forensic Sensors..."):
            detector = load_detector()
            
        with st.spinner("Analyzing Signals (Perplexity, Burstiness, GLTR...)"):
            start_time = time.time()
            result = detector.predict(text_input, use_semantic=use_semantic)
            elapsed = time.time() - start_time
            
        st.success(f"Analysis complete in {elapsed:.2f}s")
        
        # ---------------------------------------------------------
        # Results Dashboard
        # ---------------------------------------------------------
        score = result['combined_score']
        verdict = result['verdict']
        metrics = result['metrics']
        
        # Top Level Verdict
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        
        with c2:
            st.markdown(f"<div style='text-align: center;'><h3>AI Likelihood Score</h3></div>", unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 50], 'color': "#D1FAE5"}, # Green
                        {'range': [50, 100], 'color': "#FEE2E2"} # Red
                    ],
                }
            ))
            fig.update_layout(height=250, margin={'t':0,'b':0,'l':0,'r':0})
            st.plotly_chart(fig, use_container_width=True)
            
            verdict_class = "verdict-ai" if verdict == "AI" else "verdict-human"
            st.markdown(f"<div style='text-align: center;'><span class='{verdict_class}'>{verdict.upper()}</span></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Detailed Metrics
        st.subheader("📡 Forensic Signals")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        # Helper for metric interpretation
        def get_interpretation(value, threshold, is_greater_ai=True):
            is_ai = value > threshold if is_greater_ai else value < threshold
            color = "red" if is_ai else "green"
            label = "Likely AI" if is_ai else "Likely Human"
            return f":{color}[{label}]"

        with m_col1:
            ppl = metrics.get('perplexity', 0)
            st.metric("Perplexity", f"{ppl:.1f}", delta="Low = AI", delta_color="inverse", help="Measures how 'surprised' a model is by the text. AI text is very low perplexity (predictable).")
            st.markdown(get_interpretation(ppl, 40, False))
            st.progress(min(100, int(ppl))/100) # Simple visual, capped at 100
            
        with m_col2:
            burst = metrics.get('burstiness', 0)
            st.metric("Burstiness", f"{burst:.3f}", delta="Low = AI", delta_color="inverse", help="Measures the variation in sentence structure. AI text is monotone (low burstiness).")
            st.markdown(get_interpretation(burst, 0.4, False))
            st.progress(min(1.0, burst))
            
        with m_col3:
            green = metrics.get('gltr_green', 0)
            st.metric("GLTR Green %", f"{green*100:.1f}%", delta="High = AI", delta_color="inverse", help="Percentage of words that are in the Top-10 expected predictions. High means the text follows the 'most likely' path.")
            st.markdown(get_interpretation(green, 0.6, True))
            st.progress(min(1.0, green))
            
        with m_col4:
            tfidf = metrics.get('tfidf_prob', 0)
            st.metric("Stylistic Match", f"{tfidf*100:.1f}%", delta="High = AI", delta_color="inverse", help="Probability that the writing style matches known AI artifacts (e.g. overused transition words).")
            st.markdown(get_interpretation(tfidf, 0.5, True))
            st.progress(min(1.0, tfidf))
            
        # Semantic Section
        if use_semantic:
            st.markdown("#### 🧠 Semantic Analysis")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                const_score = metrics.get('semantic_consistency', 0)
                st.metric("Consistency Score", f"{const_score}/100", help="LLM-based check for logical contradictions.")
            with s_col2:
                coh_score = metrics.get('coherence_metric', 0)
                # Coherence: High = Human/Safe, Low = Disjointed. 
                # Note: Very high can be AI repetition. 
                st.metric("Coherence Metric", f"{coh_score:.1f}", help="Vector-based flow analysis. Very low (<10) indicates hallucination/disjointed text.")
            with s_col3:
                judge_score = metrics.get('llm_judge_score', 0)
                st.metric("LLM Judge Suspicion", f"{judge_score}%")
                
        # Raw Data Expander
        with st.expander("View Raw JSON Report"):
            st.json(result)

elif analyze_btn:
    st.error("Please enter some text to analyze.")

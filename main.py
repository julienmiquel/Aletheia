from ia_detector.perplexity import PerplexityCalculator
from ia_detector.burstiness import BurstinessAnalyzer
from ia_detector.features import TfidfDetector
from ia_detector.gltr import GLTRAnalyzer
from ia_detector.binoculars import BinocularsDetector
from ia_detector.ensemble import EnsembleDetector
from ia_detector.llm_judge import LLMJudge
import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def analyze_text(text, args):
    results = {}
    
    print(f"\nAnalyzing text (Length: {len(text)} chars)...")

    # 1. Perplexity
    if not args.skip_perplexity:
        try:
            print("\n[+] Running Perplexity Analysis...")
            ppl_calc = PerplexityCalculator()
            ppl = ppl_calc.calculate(text)
            results['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")
        except Exception as e:
            print(f"Perplexity failed: {e}")

    # 2. Burstiness
    if not args.skip_burstiness:
        try:
            print("\n[+] Running Burstiness Analysis...")
            burst_calc = BurstinessAnalyzer()
            metrics = burst_calc.analyze(text)
            results['burstiness'] = metrics
            print(f"Burstiness Output: {metrics}")
        except Exception as e:
            print(f"Burstiness failed: {e}")

    # 3. TF-IDF (Requires model)
    if not args.skip_tfidf:
        try:
            print("\n[+] Running TF-IDF Analysis (Feature-based)...")
            tfidf = TfidfDetector()
            if os.path.exists(tfidf.model_path):
                res = tfidf.predict(text)
                results['tfidf'] = res
                print(f"TF-IDF Prediction: {res}")
            else:
                print("TF-IDF model not found. Skipping.")
        except Exception as e:
            print(f"TF-IDF failed: {e}")

    # 4. GLTR
    if not args.skip_gltr:
        try:
            print("\n[+] Running GLTR Analysis...")
            gltr = GLTRAnalyzer()
            gltr_res = gltr.analyze(text)
            fractions = gltr.get_fraction_clean(gltr_res)
            results['gltr_fractions'] = fractions
            print(f"GLTR Fractions: {fractions}")
        except Exception as e:
            print(f"GLTR failed: {e}")

    # 5. Binoculars
    if not args.skip_binoculars:
        try:
            print("\n[+] Running Binoculars Analysis...")
            bino = BinocularsDetector()
            if bino.available:
                bino_res = bino.detect(text)
                results['binoculars'] = bino_res
                print(f"Binoculars Result: {bino_res}")
            else:
                print("Binoculars library not installed or not available.")
        except Exception as e:
            print(f"Binoculars failed: {e}")

    # --- Combined Scoring ---
    # This call to get_combined_score will become unused if the new main is adopted.
    # Keeping it as per instruction to only modify get_combined_score and main definitions.
    # Ensemble
    ensemble = EnsembleDetector()
    
    combined_score = get_combined_score(results, ensemble, text_length=len(text))
    judge_verdict = None
    
    # LLM Judge (Optional)
    if "--judge" in sys.argv:
        print("\nRunning LLM Judge (Gemini)...")
        judge = LLMJudge()
        judge_res = judge.evaluate(text)
        print(f"Judge Score: {judge_res['score']:.2f}%")
        print(f"Reasoning: {judge_res['reasoning']}")
        
        # Weighted combination if Judge is active
        # Giving Judge 50% weight if available, as it's semantic vs statistical
        combined_score = (combined_score + judge_res['score']) / 2
        
    print(f"\n>> Final AI Likelihood: {combined_score:.2f}%")
    
    verdict = "AI-Generated" if combined_score > 50 else "Human-Written"
    print(f">> Verdict: {verdict}")

    return results

def get_combined_score(metrics, ensemble_detector=None, text_length=None):
    """
    Combines individual metrics into a single AI likelihood score (0-100%).
    Uses Ensemble Model if available, otherwise falls back to heuristics.
    """
    
    # 1. Ensemble Prediction (Preferred)
    if ensemble_detector:
        # Features: [ppl, burst, entropy, gltr_green, tfidf_prob]
        ppl = metrics.get('perplexity', 100)
        burst_res = metrics.get('burstiness', {})
        burst = burst_res.get('burstiness_coefficient', 0)
        entropy = burst_res.get('lexical_entropy', 0)
        gltr_green = metrics.get('gltr_fractions', {}).get('green', 0)
        tfidf_prob = metrics.get('tfidf', {}).get('ai_probability', 0.5)
        
        feature_vector = [ppl, burst, entropy, gltr_green, tfidf_prob]
        ensemble_score = ensemble_detector.predict_proba(feature_vector, text_length=text_length)
        
        if ensemble_score is not None:
            return ensemble_score

    # 2. Heuristic Fallback
    score = 0
    weight = 0
    
    # 1. Perplexity (Low PPL -> AI)
    # Heuristic: < 20 -> 100% AI, > 80 -> 0% AI
    ppl = metrics.get('perplexity')
    if ppl is not None:
        ppl_score = max(0, min(100, (80 - ppl) * (100 / 60)))
        score += ppl_score * 2.0 # Weight 2.0
        weight += 2.0
        
    # 2. Burstiness (Low Burstiness -> AI)
    # Heuristic: < 0.2 -> 100% AI, > 0.5 -> 0% AI
    burst = metrics.get('burstiness', {}).get('burstiness_coefficient')
    if burst is not None:
        burst_score = max(0, min(100, (0.5 - burst) * (100 / 0.3)))
        score += burst_score * 1.5
        weight += 1.5
        
    # 3. GLTR (High Green -> AI)
    # Heuristic: > 0.6 Green -> High AI
    gltr = metrics.get('gltr_fractions')
    if gltr:
        green = gltr.get('Green', 0)
        gltr_score = max(0, min(100, (green - 0.2) * (100 / 0.6)))
        score += gltr_score * 1.5
        weight += 1.5

    # 4. TF-IDF Probability
    tfidf = metrics.get('tfidf')
    if tfidf:
        prob = tfidf.get('ai_probability', 0)
        score += prob * 100 * 2.5 # High weight if available
        weight += 2.5

    # 5. Binoculars (if available) - Optional additional signal
    # Not used in fetch_and_test.py heuristic, keeping consistent with that for now
    # or could add it. fetch_and_test does not check Binoculars.
        
    if weight == 0:
        return 0
        
    return score / weight

def main():
    parser = argparse.ArgumentParser(description="IA Detector - Analyze text for AI generation traces.")
    parser.add_argument('input', help="Text string or path to text file to analyze.")
    parser.add_argument('--skip-perplexity', action='store_true', help="Skip Perplexity calculation")
    parser.add_argument('--skip-burstiness', action='store_true', help="Skip Burstiness calculation")
    parser.add_argument('--skip-tfidf', action='store_true', help="Skip TF-IDF calculation")
    parser.add_argument("--skip-gltr", action="store_true", help="Skip GLTR analysis")
    parser.add_argument("--skip-binoculars", action="store_true", help="Skip Binoculars analysis")
    parser.add_argument("--judge", action="store_true", help="Enable LLM Judge (Gemini) analysis")
    
    args = parser.parse_args()
    
    input_text = ""
    if os.path.isfile(args.input):
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = args.input
        
    if not input_text.strip():
        print("Error: Empty input.")
        sys.exit(1)
        
    analyze_text(input_text, args)

if __name__ == "__main__":
    main()

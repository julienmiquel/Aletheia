from datasets import load_dataset
import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from ia_detector.perplexity import PerplexityCalculator
from ia_detector.burstiness import BurstinessAnalyzer
from ia_detector.gltr import GLTRAnalyzer
from ia_detector.features import TfidfDetector

def get_combined_score(metrics):
    """
    Heuristic to calculate a combined 'Likelihood of beign AI' score (0-100%).
    """
    score = 0
    weight = 0
    
    # 1. Perplexity (Low PPL -> AI)
    # Heuristic: < 20 -> 100% AI, > 80 -> 0% AI
    ppl = metrics.get('perplexity')
    if ppl is not None:
        ppl_score = max(0, min(100, (80 - ppl) * (100/60)))
        score += ppl_score * 2.0 # Weight 2.0
        weight += 2.0
        
    # 2. Burstiness (Low Burstiness -> AI)
    # Heuristic: < 0.2 -> 100% AI, > 0.5 -> 0% AI
    burst = metrics.get('burstiness', {}).get('burstiness_coefficient')
    if burst is not None:
        burst_score = max(0, min(100, (0.5 - burst) * (100/0.3)))
        score += burst_score * 1.5
        weight += 1.5
        
    # 3. GLTR (High Green -> AI)
    # Heuristic: > 0.6 Green -> High AI
    gltr = metrics.get('gltr')
    if gltr:
        green = gltr.get('Green', 0)
        gltr_score = max(0, min(100, (green - 0.2) * (100/0.6)))
        score += gltr_score * 1.5
        weight += 1.5

    # 4. TF-IDF Probability
    tfidf = metrics.get('tfidf')
    if tfidf:
        prob = tfidf.get('ai_probability', 0)
        score += prob * 100 * 2.5 # High weight if available
        weight += 2.5
        
    if weight == 0:
        return 0
        
    return score / weight

def run_test():
    print("Loading datasets for Training & Testing...")
    
    # Load more data for training TF-IDF
    N_TRAIN = 50
    N_TEST = 5
    
    # 1. Human (IMDb)
    print(f"Fetching {N_TRAIN+N_TEST} records from Human/IMDb...")
    try:
        ds_human = load_dataset("imdb", split="train", streaming=True)
        human_data = list(ds_human.take(N_TRAIN + N_TEST))
        human_texts = [x['text'] for x in human_data]
        train_human = human_texts[:N_TRAIN]
        test_human = human_texts[N_TRAIN:]
    except Exception as e:
        print(f"Error loading IMDb: {e}")
        return

    # 2. AI (Alpaca)
    # Alpaca 'output' is the AI text
    print(f"Fetching {N_TRAIN+N_TEST} records from AI/Alpaca...")
    try:
        ds_ai = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        # Filter for reasonable length > 100 chars
        ai_data = list(ds_ai.filter(lambda x: len(x['output']) > 100).take(N_TRAIN + N_TEST))
        ai_texts = [x['output'] for x in ai_data]
        train_ai = ai_texts[:N_TRAIN]
        test_ai = ai_texts[N_TRAIN:]
    except Exception as e:
        print(f"Error loading Alpaca: {e}")
        return

    # Train TF-IDF
    print("\nTraining TF-IDF Detector on gathered samples...")
    tfidf_detector = TfidfDetector()
    tfidf_detector.train(train_human, train_ai)
    tfidf_detector.save() # Save the model for main.py to use
    
    # Initialize other detectors
    print("Initializing Perplexity, Burstiness, and GLTR models...")
    ppl_calc = PerplexityCalculator()
    burst_calc = BurstinessAnalyzer()
    gltr_calc = GLTRAnalyzer()

    def evaluate_text(text, label_type):
        print(f"\n--- Testing {label_type} Sample ---")
        print(f"Preview: {text[:80].replace(chr(10), ' ')}...")
        
        metrics = {}
        
        # PPL
        try:
            metrics['perplexity'] = ppl_calc.calculate(text)
        except: pass
            
        # Burstiness
        try:
            metrics['burstiness'] = burst_calc.analyze(text)
        except: pass
            
        # GLTR
        try:            
            gltr_res = gltr_calc.analyze(text)
            metrics['gltr'] = gltr_calc.get_fraction_clean(gltr_res)
        except: pass
        
        # TF-IDF
        try:
            metrics['tfidf'] = tfidf_detector.predict(text)
        except: pass
        
        # Display
        p = metrics.get('perplexity', 0)
        b = metrics.get('burstiness', {}).get('burstiness_coefficient', 0)
        g = metrics.get('gltr', {}).get('Green', 0)
        t = metrics.get('tfidf', {}).get('ai_probability', 0)
        
        print(f"Metrics: PPL={p:.2f} | Burst={b:.3f} | GLTR_Green={g:.2f} | TFIDF_Prob={t:.2f}")
        
        combined = get_combined_score(metrics)
        print(f">> Combined AI Likelihood: {combined:.2f}%")
        
        # Accuracy check
        is_ai_prediction = combined > 50
        correct = (is_ai_prediction and label_type=="AI") or (not is_ai_prediction and label_type=="Human")
        print(f">> Verdict: {'RIGHT' if correct else 'WRONG'} (Ground Truth: {label_type})")
        return correct

    # Run Analysis on Test Set
    correct_count = 0
    total = 0
    
    print("\n=== STARTING EVALUATION ===")
    
    # Test Human
    for txt in test_human:
        if evaluate_text(txt, "Human"):
            correct_count += 1
        total += 1
        
    # Test AI
    for txt in test_ai:
        if evaluate_text(txt, "AI"):
            correct_count += 1
        total += 1
        
    print(f"\n=== FINAL ACCURACY: {correct_count}/{total} ({correct_count/total*100:.1f}%) ===")

if __name__ == "__main__":
    run_test()

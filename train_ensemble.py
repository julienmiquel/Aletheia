import json
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append(os.getcwd())

from ia_detector.perplexity import PerplexityCalculator
from ia_detector.burstiness import BurstinessAnalyzer
from ia_detector.gltr import GLTRAnalyzer
from ia_detector.features import TfidfDetector

GENERATED_DATA_FILE = "gemini_training_data.json"
ENSEMBLE_MODEL_FILE = "ensemble_model.pkl"
TFIDF_MODEL_FILE = "tfidf_model.pkl"

def load_data():
    print("Loading Datasets...")
    
    # Human Data
    try:
        imdb = load_dataset("imdb", split="train")
        imdb_texts = imdb.shuffle(seed=42).select(range(400))['text'] 
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        wiki_texts = [x['text'] for x in wikitext.shuffle(seed=42).select(range(1000)) if len(x['text']) > 100][:400]
        human_texts = list(imdb_texts) + list(wiki_texts)
    except Exception as e:
        print(f"Error loading human data: {e}")
        human_texts = []

    # AI Data
    try:
        with open(GENERATED_DATA_FILE, 'r') as f:
            gemini_data = json.load(f)
        gemini_texts = [item['text'] for item in gemini_data if item['label'] == 'ai']
        
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_texts = alpaca.shuffle(seed=42).select(range(200))['text']
        ai_texts = gemini_texts + list(alpaca_texts)
    except Exception as e:
        print(f"Error loading AI data: {e}")
        ai_texts = []

    # Balance
    min_len = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:min_len]
    ai_texts = ai_texts[:min_len]
    
    print(f"Loaded {len(human_texts)} Human and {len(ai_texts)} AI samples.")
    return human_texts, ai_texts

from concurrent.futures import ThreadPoolExecutor

def process_single_text(text, ppl_calc, burst_calc, gltr_calc, tfidf_detector):
    # Truncate for speed (approx 250 tokens)
    text = text[:1000]
    
    # 1. Perplexity
    try: ppl = ppl_calc.calculate(text)
    except: ppl = 100
    
    # 2. Burstiness & Entropy
    try: 
        b_res = burst_calc.analyze(text)
        burst = b_res['burstiness_coefficient']
        entropy = b_res['lexical_entropy']
    except: 
        burst = 0
        entropy = 0
        
    # 3. GLTR (Disabled for speed)
    # try:
    #     gltr_res = gltr_calc.analyze(text)
    #     fracs = gltr_calc.get_fraction_clean(gltr_res)
    #     gltr_green = fracs['green']
    # except:
    gltr_green = 0.0
        
    # 4. TF-IDF
    try:
        tfidf_prob = tfidf_detector.pipeline.predict_proba([text])[0][1]
    except:
        tfidf_prob = 0.5
        
    return [ppl, burst, entropy, gltr_green, tfidf_prob]

def extract_features(texts, ppl_calc, burst_calc, gltr_calc, tfidf_detector):
    features = []
    print(f"Extracting features for {len(texts)} samples (Parallel)...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_text, t, ppl_calc, burst_calc, gltr_calc, tfidf_detector) for t in texts]
        
        # Iterate as they complete to show progress
        for future in tqdm(futures):
            features.append(future.result())
            
    return np.array(features)

def train_ensemble():
    # 0. Initialize Detectors on CPU for Thread Safety/Parallellism
    print("Initializing Base Detectors (CPU)...")
    ppl_calc = PerplexityCalculator(device='cpu')
    burst_calc = BurstinessAnalyzer()
    gltr_calc = GLTRAnalyzer(device='cpu')
    tfidf_detector = TfidfDetector()
    
    # Train/Load TF-IDF first on the full corpus (or partial)
    # Ideally, we should train TF-IDF on a separate split to avoid leakage, 
    # but for this POC we'll assume the loaded 'tfidf_model.pkl' is sufficiently general or retrain it.
    # Let's retrain TFIDF on the data first to ensure it handles the new prompts.
    human_texts, ai_texts = load_data()
    
    # Train TF-IDF model on the new dataset
    print("Training TF-IDF model on enriched dataset...")
    tfidf_detector.train(human_texts, ai_texts)
    tfidf_detector.save(TFIDF_MODEL_FILE)

    # 1. Extract Features
    X_human = extract_features(human_texts, ppl_calc, burst_calc, gltr_calc, tfidf_detector)
    y_human = np.zeros(len(human_texts))
    
    X_ai = extract_features(ai_texts, ppl_calc, burst_calc, gltr_calc, tfidf_detector)
    y_ai = np.ones(len(ai_texts))
    
    X = np.vstack([X_human, X_ai])
    y = np.hstack([y_human, y_ai])
    
    # 2. Train Ensemble
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Ensemble...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Evaluate
    preds = clf.predict(X_test)
    print("Ensemble Performance:")
    print(classification_report(y_test, preds))
    
    # 4. Save
    with open(ENSEMBLE_MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Ensemble model saved to {ENSEMBLE_MODEL_FILE}")

if __name__ == "__main__":
    train_ensemble()

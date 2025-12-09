import json
import os
import sys
import pickle
from datasets import load_dataset
from ia_detector.features import TfidfDetector

# Add current directory to path
sys.path.append(os.getcwd())

GENERATED_DATA_FILE = "gemini_training_data.json"
MODEL_FILE = "tfidf_model.pkl"

def load_generated_data():
    if not os.path.exists(GENERATED_DATA_FILE):
        print(f"Error: {GENERATED_DATA_FILE} not found.")
        return []
    with open(GENERATED_DATA_FILE, 'r') as f:
        return json.load(f)

def train_augmented_model():
    print("--- Training Augmented TF-IDF Model ---")
    
    # 1. Load Human Data (IMDb)
    print("Loading Human samples (IMDb)...")
    try:
        imdb = load_dataset("imdb", split="train")
        # Use more samples since we have more AI data now
        imdb_texts = imdb.shuffle(seed=42).select(range(300))['text'] 
        print(f"Loaded {len(imdb_texts)} IMDb samples.")
        
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter empty lines
        wiki_texts = [x['text'] for x in wikitext.shuffle(seed=42).select(range(1000)) if len(x['text']) > 100][:300]
        print(f"Loaded {len(wiki_texts)} WikiText samples.")
        
        human_texts = list(imdb_texts) + list(wiki_texts)
        print(f"Total Human samples: {len(human_texts)}")
    except Exception as e:

        print(f"Failed to load IMDb: {e}")
        return

    # 2. Load AI Data (Gemini)
    print("Loading AI samples (Gemini)...")
    gemini_data = load_generated_data()
    ai_texts = [item['text'] for item in gemini_data if item['label'] == 'ai']
    print(f"Loaded {len(ai_texts)} Augmented AI samples.")
    
    # 3. Load Baseline AI Data (Alpaca) - to keep it robust against non-Gemini AI
    print("Loading Baseline AI samples (Alpaca)...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_texts = alpaca.shuffle(seed=42).select(range(100))['text'] # Mix in some Alpaca
        ai_texts.extend(alpaca_texts)
        print(f"Added {len(alpaca_texts)} Alpaca samples. Total AI: {len(ai_texts)}")
    except Exception as e:
        print(f"Failed to load Alpaca: {e}")

    # Balance datasets if needed
    min_len = min(len(human_texts), len(ai_texts))
    print(f"Balancing datasets to {min_len} samples each...")
    human_texts = human_texts[:min_len]
    ai_texts = ai_texts[:min_len]

    # 4. Train
    print("Training TF-IDF Detector...")
    detector = TfidfDetector()
    detector.train(human_texts, ai_texts)
    
    # 5. Save
    detector.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_augmented_model()

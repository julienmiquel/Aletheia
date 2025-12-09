
import os
import sys
import argparse
import pandas as pd
import json
import tqdm
import logging
from datasets import load_dataset
from datetime import datetime

# Import detectors
# Add root directory to path to allow importing ia_detector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ia_detector.perplexity import PerplexityCalculator
from ia_detector.burstiness import BurstinessAnalyzer
from ia_detector.gltr import GLTRAnalyzer
from ia_detector.features import TfidfDetector
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer
from ia_detector.ensemble import EnsembleDetector

class BenchmarkSuite:
    def __init__(self, output_dir="docs/benchmarks/data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing Detectors...")
        self.ppl = PerplexityCalculator()
        self.burst = BurstinessAnalyzer()
        self.gltr = GLTRAnalyzer()
        self.tfidf = TfidfDetector() # Assumes pre-trained model exists or handles loading
        self.semantic = SemanticConsistencyAnalyzer()
        self.ensemble = EnsembleDetector()
        
    def load_data(self, n_samples):
        print(f"Loading {n_samples} samples per category...")
        
        # Human: IMDb
        ds_human = load_dataset("imdb", split="train", streaming=True).take(n_samples)
        human_texts = [x['text'] for x in ds_human]
        
        # AI: Alpaca (Long enough for analysis)
        ds_ai = load_dataset("tatsu-lab/alpaca", split="train", streaming=True).filter(lambda x: len(x['output']) > 200).take(n_samples)
        ai_texts = [x['output'] for x in ds_ai]
        
        return human_texts, ai_texts

    def run(self, n_samples=10):
        human_texts, ai_texts = self.load_data(n_samples)
        results = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def process_text(text, label):
            row = {"timestamp": timestamp, "label": label, "length": len(text)}
            
            # 1. Perplexity
            try:
                row["perplexity"] = self.ppl.calculate(text)
            except: row["perplexity"] = None
            
            # 2. Burstiness
            try:
                b_res = self.burst.analyze(text)
                row["burstiness"] = b_res.get("burstiness_coefficient")
            except: row["burstiness"] = None
            
            # 3. GLTR
            try:
                g_res = self.gltr.analyze(text)
                row["gltr_green"] = self.gltr.get_fraction_clean(g_res).get("Green")
            except: row["gltr_green"] = None
            
            # 4. TF-IDF
            try:
                t_res = self.tfidf.predict(text)
                row["tfidf_prob"] = t_res.get("ai_probability")
            except: row["tfidf_prob"] = None
            
            # 5. Semantic (Expensive)
            # Only run for small N or if explicitly enabled (omitted for speed in general runs unless N is small)
            if n_samples <= 20: 
                try:
                    s_res = self.semantic.analyze(text)
                    row["semantic_consistency"] = s_res.get("consistency_score")
                except: row["semantic_consistency"] = None
            else:
                row["semantic_consistency"] = None

            # 6. Ensemble (Verdict)
            try:
                e_res = self.ensemble.predict(text)
                row["ensemble_score"] = e_res.get("combined_score")
            except: row["ensemble_score"] = None
            
            return row

        print("\n--- Benchmarking Human Texts ---")
        for i, text in enumerate(human_texts):
            sys.stdout.write(f"\rHuman {i+1}/{n_samples}")
            sys.stdout.flush()
            results.append(process_text(text, "Human"))
            
        print("\n--- Benchmarking AI Texts ---")
        for i, text in enumerate(ai_texts):
            sys.stdout.write(f"\rAI {i+1}/{n_samples}")
            sys.stdout.flush()
            results.append(process_text(text, "AI"))
            
        # Compile
        df = pd.DataFrame(results)
        output_file = f"{self.output_dir}/comprehensive_benchmark_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n\nBenchmark Complete. Results saved to: {output_file}")
        
        # summary
        print("\n=== Summary Stats ===")
        print(df.groupby("label")[["perplexity", "burstiness", "gltr_green", "tfidf_prob", "ensemble_score"]].mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Samples per category")
    args = parser.parse_args()
    
    suite = BenchmarkSuite()
    suite.run(n_samples=args.n)

import os
import sys
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer
from datasets import load_dataset
import pandas as pd
import time

def run_semantic_benchmark(n_samples=10):
    """
    Runs a benchmark of the SemanticConsistencyAnalyzer on Human vs AI text.
    
    Args:
        n_samples (int): Number of samples per category to test.
                         WARNING: Each sample incurs an API cost. Keep small for dev.
    """
    print(f"Initializing Benchmark (N={n_samples} per category)...")
    analyzer = SemanticConsistencyAnalyzer()
    
    if not analyzer.client:
        print("ERROR: API Key not found. Cannot run benchmark.")
        return

    # 1. Load Data
    print("Loading datasets...")
    # Human: IMDb (Reviews are usually consistent/grounded opinions)
    ds_human = load_dataset("imdb", split="train", streaming=True).take(n_samples)
    human_texts = [x['text'] for x in ds_human]
    
    # AI: Alpaca (Generated instructions/stories, mixed consistency)
    ds_ai = load_dataset("tatsu-lab/alpaca", split="train", streaming=True).filter(lambda x: len(x['output']) > 200).take(n_samples)
    ai_texts = [x['output'] for x in ds_ai]

    results = []

    # 2. Test Human
    print("\n--- Testing Human Samples ---")
    for i, text in enumerate(human_texts):
        sys.stdout.write(f"\rProcessing Human {i+1}/{n_samples}...")
        sys.stdout.flush()
        res = analyzer.analyze(text)
        results.append({
            "type": "Human",
            "score": res['consistency_score'],
            "reasoning": res['reasoning'][:50] + "..."
        })
        time.sleep(1) # Rate limit politeness

    # 3. Test AI
    print("\n\n--- Testing AI Samples ---")
    for i, text in enumerate(ai_texts):
        sys.stdout.write(f"\rProcessing AI {i+1}/{n_samples}...")
        sys.stdout.flush()
        res = analyzer.analyze(text)
        results.append({
            "type": "AI",
            "score": res['consistency_score'],
            "reasoning": res['reasoning'][:50] + "..."
        })
        time.sleep(1)

    # 4. Analysis
    df = pd.DataFrame(results)
    print("\n\n=== Benchmark Results ===")
    print(df.groupby("type")["score"].agg(["mean", "std", "min", "max"]))
    
    print("\nDetailed Preview:")
    print(df.head())
    
    # Save
    os.makedirs("docs/benchmarks/data", exist_ok=True)
    df.to_csv("docs/benchmarks/data/semantic_benchmark_results.csv", index=False)
    print("\nResults saved to docs/benchmarks/data/semantic_benchmark_results.csv")

if __name__ == "__main__":
    # Check for CLI arg for N
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    args = parser.parse_args()
    
    run_semantic_benchmark(n_samples=args.n)

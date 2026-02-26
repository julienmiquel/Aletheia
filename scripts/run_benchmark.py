import os
import sys
import json
import argparse

# Add parent directory to path to allow importing ia_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ia_detector.validation import ValidationDataset, ExperimentRunner, ValidationExample
from ia_detector.llm_judge import LLMJudge
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Run validation benchmark for Aletheia models.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-exp", help="Model name to test")
    parser.add_argument("--detector", type=str, choices=["llm_judge", "semantic"], default="llm_judge", help="Detector to test")
    parser.add_argument("--dataset", type=str, help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to output JSON file")

    args = parser.parse_args()

    # Load or Create Dataset
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading dataset from {args.dataset}...")
        dataset = ValidationDataset.from_json(args.dataset)
    else:
        print("No dataset provided. Using sample dataset.")
        dataset = ValidationDataset([
            ValidationExample(text="The quick brown fox jumps over the lazy dog.", label=0), # Human
            ValidationExample(text="As an AI language model, I cannot provide personal opinions.", label=1), # AI
        ])

    # Select Detector Class
    if args.detector == "llm_judge":
        detector_class = LLMJudge
    else:
        detector_class = SemanticConsistencyAnalyzer

    # Run Experiment
    print(f"Running benchmark with Model: {args.model}, Detector: {args.detector}...")
    runner = ExperimentRunner()
    result = runner.run(
        dataset=dataset,
        detector_class=detector_class,
        model_name=args.model
    )

    # Print Metrics
    print("\n--- Results ---")
    print(f"Avg Score: {result.metrics['avg_score']:.2f}")
    print(f"Avg Latency: {result.metrics['avg_latency']:.4f}s")
    print(f"Sample Count: {result.metrics['sample_count']}")

    # Save Results
    output_data = {
        "model": result.model_name,
        "metrics": result.metrics,
        "details": result.details
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()

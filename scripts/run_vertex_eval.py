
import os
import sys
import json
import pandas as pd
# The tutorial specifies using the modern vertexai.Client and types
from vertexai import Client, types
from google.cloud import aiplatform

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ia_detector import config
from ia_detector.vertex_metrics import aletheia_ai_score

def run_evaluation():
    # 1. Load Data
    data_path = config.DATA_DIR / "adversarial_samples.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run scripts/generate_adversarial_data.py first.")
        return

    with open(data_path, 'r') as f:
        samples = json.load(f)

    # 2. Prepare DataFrame
    # Modern SDK evaluation expects 'prompt' and 'response' columns
    eval_data = []
    for s in samples:
        eval_data.append({
            "prompt": f"Humanize this text about: {s['original_topic']}",
            "response": s['humanized_text']
        })

    df = pd.DataFrame(eval_data)

    # 3. Initialize Unified GenAI Client
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    print(f"Initializing Unified GenAI Client (Project: {project_id}, Location: {location})...")

    try:
        # According to the tutorial: "unified, client-based architecture accessed via the vertexai.Client class"
        if project_id:
            client = Client(project=project_id, location=location)
        else:
            # We skip evaluation if no credentials to avoid hard crash, but warn user.
            print("Warning: GOOGLE_CLOUD_PROJECT is not set. Mocking GenAI client for dry-run.")
            # We'll just run the detector part
            client = None
    except Exception as e:
        print(f"GenAI Client Init Failed: {e}")
        return

    # 4. Define Evaluation Metrics List
    metrics_list = [
        "coherence",
        "fluency",
        "safety"
    ]

    if client:
        try:
            # Using the modern SDK types.LLMMetric and types.MetricPromptBuilder
            # As documented: "The SDK facilitates this level of customization through the LLMMetric and MetricPromptBuilder classes."
            if hasattr(types, 'LLMMetric') and hasattr(types, 'MetricPromptBuilder'):
                 human_likeness_metric = types.LLMMetric(
                    name="human_likeness",
                    prompt_template=types.MetricPromptBuilder(
                        instruction="You are an expert linguist. Rate how human-like this text sounds.",
                        criteria={
                            "Fluency": "The text flows naturally without robotic transitions.",
                            "Colloquialism": "The text uses idioms or casual language appropriate for a human."
                        },
                        rating_scores={
                            "5": "Completely indistinguishable from human writing.",
                            "4": "Very human-like, minor stiffness.",
                            "3": "Somewhat human-like, but has some AI-like phrasing.",
                            "2": "Clearly machine-generated but readable.",
                            "1": "Robotic, repetitive, and obviously AI."
                        }
                    )
                 )
                 metrics_list.append(human_likeness_metric)
        except Exception as e:
             print(f"Warning: Custom LLMMetric definition failed: {e}")

    # 5. Calculate Aletheia score locally and add to DataFrame
    print("Running Aletheia Detector on samples...")
    df['aletheia_score'] = df['response'].apply(lambda text: aletheia_ai_score({"response": text}))

    # 6. Run Evaluation using client.evals.evaluate()
    if client:
        print("Running Vertex AI Evaluation...")
        try:
            eval_result = client.evals.evaluate(
                dataset=df,
                metrics=metrics_list,
            )

            # 7. Analyze & Report
            print("\n=== Evaluation Results ===")

            # Note: The tutorial demonstrates checking eval_result.show() in notebooks.
            # In a script, we typically extract the metrics_table or similar attribute.
            if hasattr(eval_result, 'metrics_table'):
                results_df = eval_result.metrics_table
            elif hasattr(eval_result, 'summary_metrics'):
                # Some versions might store raw row results differently
                print("Summary Metrics:", eval_result.summary_metrics)
                results_df = df.copy() # fallback
            else:
                 print("Warning: Could not extract metrics table directly. Falling back to input dataframe.")
                 results_df = df.copy()

            # Ensure local scores are present
            if 'aletheia_score' not in results_df.columns:
                 results_df['aletheia_score'] = df['aletheia_score'].values

            print(results_df.head())

            # Save raw results
            output_file = "docs/benchmarks/data/vertex_eval_results.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")

            # --- Correlation Analysis ---
            # Look for the column name output by the eval service
            judge_col = 'human_likeness/score'

            if judge_col in results_df.columns and 'aletheia_score' in results_df.columns:
                # Convert Likeness (1-5) to AI Prob (0-100)
                results_df['judge_ai_prob'] = (5 - results_df[judge_col]) * 25

                correlation = results_df['aletheia_score'].corr(results_df['judge_ai_prob'])
                print(f"\nCorrelation (Aletheia vs Judge AI Probability): {correlation:.2f}")

                # Highlight Disagreements
                missed_ai = results_df[
                    (results_df['judge_ai_prob'] > 75) & (results_df['aletheia_score'] < 40)
                ]

                if not missed_ai.empty:
                    print(f"\n[Improvement Opportunity] Detector missed {len(missed_ai)} samples that Judge identified as AI-like:")
                    for idx, row in missed_ai.iterrows():
                        print(f"- Sample {idx}: Judge Prob={row['judge_ai_prob']}, Aletheia={row['aletheia_score']}")

        except Exception as e:
            print(f"Evaluation Failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n=== Dry Run Results ===")
        print("Aletheia Scores Calculated:")
        print(df[['prompt', 'aletheia_score']].head())

if __name__ == "__main__":
    run_evaluation()

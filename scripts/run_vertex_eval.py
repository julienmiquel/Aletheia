
import os
import sys
import json
import pandas as pd
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
from vertexai import init as vertexai_init

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
    eval_data = []
    for s in samples:
        eval_data.append({
            "prompt": f"Humanize this text about: {s['original_topic']}",
            "response": s['humanized_text'],
            # "reference": s['humanized_text']
        })

    df = pd.DataFrame(eval_data)

    # 3. Initialize Vertex AI Client
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    print(f"Initializing Vertex AI (Project: {project_id}, Location: {location})...")

    try:
        # If project_id is None, it might work if Application Default Credentials are set to a project.
        if project_id:
            vertexai_init(project=project_id, location=location)
        else:
            vertexai_init(location=location)
    except Exception as e:
        print(f"Vertex AI Init Warning: {e}")
        # Proceeding might fail later, but let's try.

    # 4. Define Custom Metrics

    # A. Human-Likeness (LLM-based)
    # Define criteria and rating scheme
    human_likeness_criteria = {
        "Fluency": "The text flows naturally without robotic transitions.",
        "Colloquialism": "The text uses idioms or casual language appropriate for a human.",
        "Imperfection": "The text has minor structural variety typical of humans."
    }

    human_likeness_rating_scheme = {
        "5": "Completely indistinguishable from human writing.",
        "4": "Very human-like, minor stiffness.",
        "3": "Somewhat human-like, but has some AI-like phrasing.",
        "2": "Clearly machine-generated but readable.",
        "1": "Robotic, repetitive, and obviously AI."
    }

    # Create the Prompt Template object
    human_likeness_prompt_template = PointwiseMetricPromptTemplate(
        criteria=human_likeness_criteria,
        rating_scheme=human_likeness_rating_scheme
    )

    # Create the Metric object
    human_likeness_metric = PointwiseMetric(
        metric="human_likeness",
        metric_prompt_template=human_likeness_prompt_template
    )

    # B. Aletheia Detector (Python-based)
    # Passed directly as a callable: aletheia_ai_score

    # 5. Define Evaluation Task
    print("Defining EvalTask...")

    metrics_list = [
        # Built-in metrics (strings)
        "coherence",
        "fluency",
        "safety",
        # Custom LLM Metric
        human_likeness_metric,
        # Custom Python Metric
        aletheia_ai_score
    ]

    try:
        eval_task = EvalTask(
            dataset=df,
            metrics=metrics_list,
            experiment="aletheia-adversarial-eval"
        )

        # 6. Run Evaluation
        print("Running Vertex AI Evaluation...")
        result = eval_task.evaluate()

        # 7. Analyze & Report
        print("\n=== Evaluation Results ===")

        # Access the metrics table
        if hasattr(result, 'metrics_table'):
            results_df = result.metrics_table
            # Display first few rows
            print(results_df.head())

            # Save raw results
            output_file = "docs/benchmarks/data/vertex_eval_results.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")

            # --- Correlation Analysis ---
            # Identify the column names for our custom metrics
            # Usually: 'metric_name/score'
            judge_col = 'human_likeness/score'
            # Python metrics usually result in a column named after the function
            # Note: Depending on SDK version, might be 'aletheia_ai_score' or 'aletheia_ai_score/score'
            aletheia_col = 'aletheia_ai_score'
            if aletheia_col not in results_df.columns:
                 if 'aletheia_ai_score/score' in results_df.columns:
                     aletheia_col = 'aletheia_ai_score/score'

            if judge_col in results_df.columns and aletheia_col in results_df.columns:
                # Convert Likeness (1-5) to AI Prob (0-100)
                # 5 (Human) -> 0 AI Prob
                # 1 (AI) -> 100 AI Prob
                results_df['judge_ai_prob'] = (5 - results_df[judge_col]) * 25 # Scale 0-4 to 0-100

                correlation = results_df[aletheia_col].corr(results_df['judge_ai_prob'])
                print(f"\nCorrelation (Aletheia vs Judge AI Probability): {correlation:.2f}")

                # Highlight Disagreements
                # High Judge AI Prob (Low Likeness) AND Low Aletheia Score -> False Negative (Detector missed it)
                missed_ai = results_df[
                    (results_df['judge_ai_prob'] > 75) & (results_df[aletheia_col] < 40)
                ]

                if not missed_ai.empty:
                    print(f"\n[Improvement Opportunity] Detector missed {len(missed_ai)} samples that Judge identified as AI-like:")
                    for idx, row in missed_ai.iterrows():
                        print(f"- Sample {idx}: Judge Prob={row['judge_ai_prob']}, Aletheia={row[aletheia_col]}")
                        print(f"  Topic: {row.get('prompt', '')}")

        else:
            print("Warning: Could not extract metrics table directly.")
            print(result)

    except Exception as e:
        print(f"Evaluation Failed: {e}")
        # print full stack trace
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()

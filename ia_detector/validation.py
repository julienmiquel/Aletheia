import time
import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ValidationExample:
    text: str
    label: int  # 0 for Human, 1 for AI (or similar scale)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ValidationDataset:
    def __init__(self, examples: List[ValidationExample]):
        self.examples = examples

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Handle list of dicts or dict with 'examples' key
        if isinstance(data, dict) and 'examples' in data:
            data = data['examples']

        examples = []
        for item in data:
            # Filter keys that are valid for ValidationExample
            valid_keys = {'text', 'label', 'metadata'}
            filtered_item = {k: v for k, v in item.items() if k in valid_keys}

            # Ensure required keys exist
            if 'text' not in filtered_item or 'label' not in filtered_item:
                continue

            examples.append(ValidationExample(**filtered_item))

        return cls(examples)

@dataclass
class ExperimentResult:
    model_name: str
    prompt_template: Optional[str]
    generation_config: Optional[Dict]
    metrics: Dict[str, float]
    details: List[Dict[str, Any]]

class ExperimentRunner:
    def __init__(self):
        pass

    def run(self,
            dataset: ValidationDataset,
            detector_class: Any,
            model_name: str,
            prompt_template: Optional[str] = None,
            generation_config: Optional[Dict] = None) -> ExperimentResult:

        # Instantiate the detector
        detector = detector_class(
            model_name=model_name,
            prompt_template=prompt_template,
            generation_config=generation_config
        )

        results = []
        latencies = []
        scores = []

        for example in dataset.examples:
            start_time = time.time()
            try:
                # Detect method name
                if hasattr(detector, 'evaluate'):
                    output = detector.evaluate(example.text)
                    # LLMJudge returns 'score' (0=Human, 100=AI)
                    score = output.get('score', 50)
                elif hasattr(detector, 'analyze'):
                    output = detector.analyze(example.text)
                    # SemanticConsistencyAnalyzer returns 'consistency_score' (0=AI, 100=Human)
                    # Normalize to the same scale as LLMJudge (higher = more AI-like)
                    consistency_score = output.get('consistency_score', 50)
                    score = 100 - consistency_score
                else:
                    raise ValueError(f"Detector {detector_class.__name__} does not have evaluate or analyze method")

                end_time = time.time()
                latency = end_time - start_time

                results.append({
                    "text_snippet": example.text[:50] + "...",
                    "label": example.label,
                    "score": score,
                    "latency": latency,
                    "raw_output": output
                })
                latencies.append(latency)
                scores.append(score)

            except Exception as e:
                print(f"Error processing example: {e}")
                results.append({
                    "text_snippet": example.text[:50] + "...",
                    "label": example.label,
                    "error": str(e)
                })

        # Calculate metrics
        avg_latency = statistics.mean(latencies) if latencies else 0
        avg_score = statistics.mean(scores) if scores else 0

        # Basic accuracy check?
        # Difficult without knowing the threshold and direction (AI vs Human score).
        # We leave detailed metric interpretation to the user/script analyzing the result.

        metrics = {
            "avg_latency": avg_latency,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "avg_score": avg_score,
            "sample_count": len(latencies)
        }

        return ExperimentResult(
            model_name=model_name,
            prompt_template=prompt_template,
            generation_config=generation_config,
            metrics=metrics,
            details=results
        )

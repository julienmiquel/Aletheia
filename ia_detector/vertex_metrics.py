from ia_detector.ensemble import EnsembleDetector

class AletheiaMetric:
    """
    Wrapper for Aletheia EnsembleDetector to be used as a custom metric
    in Vertex AI Evaluation.
    """
    def __init__(self):
        # Initialize the detector once
        print("Initializing Aletheia EnsembleDetector for Vertex Eval...")
        self.detector = EnsembleDetector()

    def score(self, row):
        """
        Calculates the AI Probability score for a given text.

        Args:
            row (dict): A dictionary containing the evaluation row.
                        Expected to have a 'response' key containing the text.

        Returns:
            float: The Aletheia 'combined_score' (0-100, where 100 is AI).
        """
        text = row.get("response", "")
        if not text:
            return 0.0

        try:
            # We don't need semantic checks for every single row in a large batch
            # unless we want high precision. Let's default to False for speed,
            # or maybe True if we want the full power.
            # Given this is an "Evaluation" of the detector, we should probably use its best mode.
            # However, semantic checks are slow and costly.
            # Let's stick to the fast ensemble for now, or make it configurable.
            # For this implementation, I'll use semantic=False to be safe on quotas,
            # but ideally it should be True for "Deep" evaluation.
            result = self.detector.predict(text, use_semantic=False)
            return result.get("combined_score", 0.0)
        except Exception as e:
            print(f"AletheiaMetric Error: {e}")
            return 0.0

# Singleton instance to avoid reloading models for every row if the framework re-imports
_aletheia_instance = None

def aletheia_ai_score(row):
    """
    Functional interface for the metric.
    """
    global _aletheia_instance
    if _aletheia_instance is None:
        _aletheia_instance = AletheiaMetric()

    # Vertex AI Eval often passes 'response' as a string OR as a dict,
    # but the custom metric signature is typically (row: dict) -> float
    # We should handle cases where 'row' might be the text itself if mistakenly passed.
    if isinstance(row, str):
        row = {"response": row}

    return _aletheia_instance.score(row)

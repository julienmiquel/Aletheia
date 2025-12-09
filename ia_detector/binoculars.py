try:
    from binoculars import Binoculars
except ImportError:
    Binoculars = None

class BinocularsDetector:
    def __init__(self, observer_model_name="tiiuae/falcon-7b", performer_model_name="tiiuae/falcon-7b-instruct"):
        """
        Wrapper for Binoculars.
        """
        self.available = Binoculars is not None
        self.bino = None
        if self.available:
            try:
                print("Initializing Binoculars (this may take a while)...")
                # Using default models as specified in library or custom ones
                # Note: Binoculars construtor might differ based on version, assuming standard usage
                self.bino = Binoculars(observer_name_or_path=observer_model_name, performer_name_or_path=performer_model_name)
            except Exception as e:
                print(f"Failed to initialize Binoculars: {e}")
                self.available = False

    def detect(self, text):
        """
        Uses the Binoculars zero-shot method to detect AI text.
        """
        if not self.available or self.bino is None:
            return {"error": "Binoculars library not installed or model failed to load."}

        try:
            # compute_score returns a single float for single string input
            score = self.bino.compute_score(text)
            # predict returns 'Most likely AI-Generated' or 'Most likely Human-Written'
            prediction = self.bino.predict(text)
            
            is_ai = (prediction == "Most likely AI-Generated")
            
            return {
                "score": float(score),
                "prediction": prediction,
                "is_ai": is_ai
            }
        except Exception as e:
            return {"error": str(e)}

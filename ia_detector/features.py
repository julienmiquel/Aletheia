from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os
from ia_detector import config

class TfidfDetector:
    def __init__(self, model_path=config.TFIDF_MODEL_PATH):
        """
        Initializes the TF-IDF Detector. Model is loaded lazily.
        """
        self.model_path = model_path
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            if os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                # If no model exists, return None or handle appropriately
                # For training, self.pipeline will be set by train()
                pass
        return self._pipeline

    def train(self, human_texts, ai_texts):
        """
        Trains a lightweight TF-IDF Logistic Regression detector.
        
        Args:
            human_texts (list): List of human-authored strings.
            ai_texts (list): List of AI-generated strings.
        """
        # Create labels: 0 for Human, 1 for AI
        labels = [0] * len(human_texts) + [1] * len(ai_texts)
        corpus = human_texts + ai_texts
        
        # Build a pipeline with N-gram features (unigrams and bigrams)
        self._pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression(random_state=42))
        ])
        
        print("Training TF-IDF Detector...")
        self._pipeline.fit(corpus, labels)
        print("Training complete.")

    def predict(self, text):
        """
        Predicts if text is AI generated.
        Returns prediction (0/1) and probability.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
            
        prediction = self.pipeline.predict([text])[0]
        # Probability of class 1 (AI)
        probability = self.pipeline.predict_proba([text])[0][1]
        
        return {
            "prediction_label": "AI-Generated" if prediction == 1 else "Human-Written",
            "ai_probability": probability,
            "is_ai": bool(prediction == 1)
        }

    def save(self, path=None):
        if path is None:
            path = self.model_path
        if self.pipeline:
            with open(path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.model_path
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._pipeline = pickle.load(f)
            print(f"Model loaded from {path}")
        else:
             print(f"Model file {path} not found.")


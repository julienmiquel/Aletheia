import pickle
import numpy as np
import os

class EnsembleDetector:
    def __init__(self, model_path='ensemble_model.pkl'):
        self.model = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ensemble model: {e}")
        else:
            # print(f"Warning: Ensemble model {model_path} not found.")
            pass

    def predict_proba(self, features, text_length=None):
        """
        Predicts AI probability with dynamic thresholding for short text.
        Args:
            features (list): [ppl, burst, entropy, gltr_green, tfidf_prob]
            text_length (int): Length of the text in chars.
        Returns:
            float: AI probability (0-100)
        """
        if not self.model:
            return None # Indicate no model functionality
        
        # predict_proba returns [prob_human, prob_ai]
        try:
            score = self.model.predict_proba([features])[0][1] * 100
            
            # Dynamic Thresholding / Uncertainty dampening for short text
            if text_length and text_length < 400:
                # Pull score towards 50 (Uncertain) if text is short
                # This reduces false positives/negatives on low-signal text
                if score > 50:
                    score = max(50, score - 15)
                else:
                    score = min(50, score + 15)
                    
            return score
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return None

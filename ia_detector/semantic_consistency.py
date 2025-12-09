import os
import json
from google import genai
from google.genai import types

class SemanticConsistencyAnalyzer:
    def __init__(self, model_name="gemini-3-pro-preview"):
        """
        Initializes the Semantic Consistency Analyzer using Google GenAI.
        """
        self.client = self._get_client()
        self.model_name = model_name

    def _get_client(self):
        """
        Initializes Google GenAI Client.
        Reuses logic from LLMJudge for consistency.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # In this project, we prioritize AI Studio key if available for simple setup
        if api_key:
             return genai.Client(api_key=api_key)
        
        # Fallback to Vertex AI if configured (implied by usage in other modules)
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if project_id:
            try:
                return genai.Client(vertexai=True, project=project_id, location=location)
            except Exception as e:
                print(f"SemanticConsistencyAnalyzer: Vertex AI Init failed: {e}")

        return None

    def _calculate_coherence_metric(self, text):
        """
        Calculates a quantitative 'Coherence Score' based on lexical overlap 
        between adjacent sentences using TF-IDF and Cosine Similarity.
        """
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import spacy
        
        # Lazy load spacy for segmentation if not already loaded
        if not hasattr(self, 'nlp'):
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 5]
        
        if len(sentences) < 2:
            return 100.0
            
        # Vectorize sentences (internal self-similarity)
        # Using TF-IDF on the document itself captures local keyword repetition
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            similarities = []
            for i in range(len(sentences) - 1):
                # Compare sentence i with sentence i+1
                sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])[0][0]
                similarities.append(sim)
                
            # Scale 0-1 similarity to 0-100 score
            # A score of 0.1-0.2 is actually quite normal for human text. 
            # AI often has HIGHER coherence (repetition).
            # But the requirement is "Semantic Consistency".
            # Let's return the raw 0-100 metric for now, interpreted as "Lexical Cohesion".
            avg_sim = np.mean(similarities)
            return float(avg_sim * 100)
            
        except Exception as e:
            print(f"Coherence Calc Error: {e}")
            return 50.0

    def analyze(self, text):
        """
        Analyzes the text for logical contradictions and 'hallucination-like' inconsistencies.
        Now includes a quantitative 'Coherence Score'.
        
        Returns:
            dict: {
                "consistency_score": float, # Combined Score
                "reasoning": str,
                "coherence_metric": float # Quantitative Lexical Cohesion
            }
        """
        coherence_score = self._calculate_coherence_metric(text)
        
        if not self.client:
            return {
                "consistency_score": 50, 
                "reasoning": "LLM Analyzer not initialized. Returning baseline.",
                "coherence_metric": coherence_score
            }

        prompt = f"""
        You are an expert AI forensics analyst. Analyze the following text for internal logical consistency and "hallucination-like" patterns.
        
        AI-generated text, while grammatically correct, often contains subtle logical contradictions or "dream-like" shifts in narrative availability. Human text is usually grounded in a consistent reality.
        
        Task:
        1. Identify any internal contradictions (e.g., "The car was red" then later "The blue car").
        2. Identify specific, verifiable details vs. generic "fluff".
        3. Rate the "Semantic Consistency" from 0 to 100.
           - 0: Highly inconsistent, contains obvious contradictions (Likely AI Hallucination).
           - 100: Perfectly consistent, grounded, and logical (Likely Human).
           
        Text to Analyze:
        -----
        {text[:4000]}
        -----
        
        Provide your response in JSON format:
        {{
            "reasoning": "Concise explanation of found inconsistencies or confirmation of consistency.",
            "consistency_score": <int 0-100>
        }}
        """

        llm_score = 50
        reasoning = "Analysis Failed."

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            if response.text:
                data = json.loads(response.text)
                llm_score = float(data.get("consistency_score", 50))
                reasoning = data.get("reasoning", "No reasoning provided.")
            
        except Exception as e:
            print(f"SemanticConsistencyAnalyzer LLM Error: {e}")

        # Combine logic: 
        # For now, we return both. The user can decide how to blend them.
        # But 'consistency_score' is the main API contract. 
        # Let's keep consistency_score as the LLM verdict for now, but expose the metric.
        return {
            "consistency_score": llm_score,
            "reasoning": reasoning,
            "coherence_metric": coherence_score
        }


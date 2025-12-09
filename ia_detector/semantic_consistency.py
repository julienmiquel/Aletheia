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

    def analyze(self, text):
        """
        Analyzes the text for logical contradictions and 'hallucination-like' inconsistencies.
        
        Returns:
            dict: {
                "consistency_score": float, # 0 (Inconsistent/AI-like) to 100 (Consistent/Human-like)
                "reasoning": str
            }
        """
        if not self.client:
            return {"consistency_score": 50, "reasoning": "Analyzer not initialized (No credentials)."}

        prompt = f"""
        You are an expert AI forensics analyst. Analyze the following text for internal logical consistency and "hallucination-like" patterns.
        
        AI-generated text, while grammatically correct, often contains subtle logical contradictions or "dream-like" shifts in narrative availability when generating long contexts. Human text is usually grounded in a consistent reality.
        
        Task:
        1. Identify any internal contradictions (e.g., "The car was red" then later "The blue car").
        2. Identify specific, verifiable details vs. generic "fluff".
        3. Rate the "Semantic Consistency" from 0 to 100.
           - 0: Highly inconsistent, contains obvious contradictions or nonsensical shifts (Likely AI Hallucination).
           - 100: Perfectly consistent, grounded, and logical (Likely Human or highly advanced AI).
           
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
                return {
                    "consistency_score": float(data.get("consistency_score", 50)),
                    "reasoning": data.get("reasoning", "No reasoning provided.")
                }
            
        except Exception as e:
            print(f"SemanticConsistencyAnalyzer Error: {e}")
            
        return {"consistency_score": 50, "reasoning": "Analysis Failed."}

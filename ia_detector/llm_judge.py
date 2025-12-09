import os
import json
from google import genai
from google.genai import types

class LLMJudge:
    def __init__(self, model_name="gemini-3-pro-preview"):
        """
        Initializes the LLM Judge using Google GenAI.
        """
        self.client = self._get_client()
        self.model_name = model_name

    def _get_client(self):
        """
        Initializes Google GenAI Client.
        Prioritizes AI Studio (API Key) unless a specific regional Vertex location is needed.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global") # Default to global

        # 1. AI Studio (Preferred for Global/Preview models)
        if api_key and location == "global":
            # print("LLMJudge: Using AI Studio...")
            return genai.Client(api_key=api_key)

        # 2. Vertex AI (Fallback or Region-Specific)
        if project_id:
            try:
                # print(f"LLMJudge: Using Vertex AI (Project: {project_id}, Location: {location})...")
                return genai.Client(vertexai=True, project=project_id, location=location)
            except Exception as e:
                print(f"LLMJudge: Vertex AI Init failed: {e}")

        # 3. Last Resort AI Studio
        if api_key:
             return genai.Client(api_key=api_key)

        print("LLMJudge Warning: No credentials found.")
        return None

    def evaluate(self, text):
        """
        Analyzes the text and returns a probability score (0-100) of it being AI-generated.
        Returns:
            dict: {
                "score": float, # 0 (Human) to 100 (AI)
                "reasoning": str
            }
        """
        if not self.client:
            return {"score": 50, "reasoning": "LLM Judge not initialized (No credentials)."}

        prompt = f"""
        You are an expert AI text forensics analyst. Your task is to determine if the following text was written by a Human or an AI.
        
        Analyze the text for:
        1. Absence of personal nuance or specific, verifiable anecdotes.
        2. Excessive "balance" or lack of strong opinion.
        3. Repetitive sentence structures (low burstiness).
        4. Overuse of transition words (e.g., "Furthermore", "In conclusion").
        5. "Hallucination-like" generic statements.

        Text to Analyze:
        -----
        {text[:4000]} 
        -----
        
        Provide your analysis in JSON format with two keys:
        - "reasoning": A concise explanation of your findings.
        - "ai_probability": A score from 0 to 100, where 0 is definitely Human and 100 is definitely AI.
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
                    "score": float(data.get("ai_probability", 50)),
                    "reasoning": data.get("reasoning", "No reasoning provided.")
                }
            
        except Exception as e:
            print(f"LLMJudge Error: {e}")
            
        return {"score": 50, "reasoning": "Analysis Failed."}

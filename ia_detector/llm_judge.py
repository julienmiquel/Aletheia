import os
import json
from google import genai
from google.genai import types

class LLMJudge:
    def __init__(self, model_name="gemini-3-pro-preview", prompt_template=None, generation_config=None):
        """
        Initializes the LLM Judge using Google GenAI.
        Args:
            model_name (str): The name of the Gemini model to use.
            prompt_template (str, optional): A template string with a `{text}` placeholder.
            generation_config (dict, optional): Configuration for generation (temperature, etc.).
        """
        self.client = self._get_client()
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.generation_config = generation_config

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

        if self.prompt_template:
            # Use custom prompt template
            if "{text}" in self.prompt_template:
                try:
                    prompt = self.prompt_template.format(text=text[:4000])
                except Exception:
                     # Fallback if format fails for other reasons
                    prompt = self.prompt_template + "\n\n" + text[:4000]
            else:
                # Append text if placeholder is missing
                prompt = self.prompt_template + "\n\n" + text[:4000]
        else:
            # Default prompt
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

        # Merge default config with user config
        config_params = {
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
        if self.generation_config:
            config_params.update(self.generation_config)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_params)
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

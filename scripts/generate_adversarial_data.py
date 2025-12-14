
import os
import json
import asyncio
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from ia_detector import config
from dotenv import load_dotenv
import sys

# Add current directory to path
sys.path.append(os.getcwd())

# Load environment variables from .env file
load_dotenv()
# Define Pydantic models for structured output
class AdversarialSample(BaseModel):
    original_topic: str = Field(description="The topic of the text")
    ai_text: str = Field(description="The original AI-generated text")
    humanized_text: str = Field(description="The adversarial 'humanized' version of the text")
    applied_strategy: str = Field(description="Strategy used (e.g., 'Add typos', 'Vary sentence length')")

class AdversarialBatch(BaseModel):
    samples: List[AdversarialSample]

class AdversarialGenerator:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY") # Or from config
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
            
        self.client = genai.Client(api_key=api_key)
        self.output_file = config.DATA_DIR / "adversarial_samples.json"

    def generate_data(self):
        topics = [
            "The impact of quantum computing on cryptography",
            "A review of a mediocre local italian restaurant",
            "Why cats are better pets than dogs",
            "An explanation of how a bicycle works",
            "A short story about a lost key"
        ]

        # 1. Generate Base AI Text
        print("Generating base AI text...")
        base_samples = []
        for topic in topics:
            prompt = f"Write a 200 word paragraph about: {topic}"
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            base_samples.append({"topic": topic, "text": response.text})

        # 2. Humanize (Adversarial Step)
        print("Generating adversarial (humanized) versions...")
        adversarial_samples = []
        
        for sample in base_samples:
            prompt = f"""
            You are an expert at evading AI detection. 
            Take the following AI-generated text and rewrite it to sound more HUMAN.
            
            Strategies to apply:
            1. Vary sentence length drastically (Burstiness).
            2. Use specific, vivid anecdotes or metaphors.
            3. Add slight colloquialisms or subjective opinions.
            4. Make it less "perfect" or structure it less rigidly.

            Original Text:
            {sample['text']}
            """
            
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AdversarialSample,
                        temperature=0.9
                    )
                )
                
                # Parsing is handled automatically by the response_schema
                result = response.parsed
                # Manually inject the original text/topic if the model didn't (though schema asks for it)
                # But to be safe let's rely on what we have
                if not result.ai_text:
                    result.ai_text = sample['text']
                if not result.original_topic:
                    result.original_topic = sample['topic']
                    
                adversarial_samples.append(result.model_dump())
                print(f"Generated sample for: {sample['topic']}")
                
            except Exception as e:
                print(f"Error generating sample for {sample['topic']}: {e}")

        # 3. Save
        with open(self.output_file, "w") as f:
            json.dump(adversarial_samples, f, indent=2)
        
        print(f"Saved {len(adversarial_samples)} adversarial samples to {self.output_file}")

if __name__ == "__main__":
    generator = AdversarialGenerator()
    generator.generate_data()

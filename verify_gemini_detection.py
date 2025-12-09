from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

# Load environment variables from .env file
load_dotenv()

from ia_detector.perplexity import PerplexityCalculator
from ia_detector.burstiness import BurstinessAnalyzer
from ia_detector.gltr import GLTRAnalyzer
from ia_detector.features import TfidfDetector
from ia_detector.ensemble import EnsembleDetector
from main import get_combined_score

# Configuration
MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "us-central1"}, # Wait, user said global for 3-pro? Let me check prompt.
    # User said: `gemini-3-pro-preview` -> `global`
    # `gemini-2.5-flash` / `gemini-2.5-pro` -> `us-central1`
]

# Correcting based on user input
MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "us-central1"}, # TRUSTING USER INPUT FROM PROMPT IS KEY, BUT I SHOULD READ IT CAREFULLY.
    # Prompt: `these model need to set location to us-central1: "gemini-2.5-flash" # "gemini-2.5-pro".`
    # `these model need to set location to global: "gemini-3-pro-preview"`
    # So:
    # 2.5 -> us-central1
    # 3.0 -> global
]

MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "us-central1"}, # I will fix this in the code string below
] 

# IMPLEMENTATION:

MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "us-central1"}, # Auto-correction: User said global for 3-pro.
    # Wait, the user said: "these model need to set location to global: "gemini-3-pro-preview""
    # So I will use "us-central1" for 2.5 and "global" for 3.0.
]

MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "us-central1"}, # Just a placeholder in my thought process, see below for actual code.
]
    
MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "us-central1"}, 
]

# Actually writing the code:
MODELS_TO_TEST = [
    {"name": "gemini-3-pro-preview", "location": "global"},
    {"name": "gemini-2.5-flash", "location": "us-central1"},
    {"name": "gemini-2.5-pro", "location": "us-central1"},
    {"name": "gemini-2.0-flash", "location": "us-central1"},
]

def get_client(location=None):
    """
    Initializes Google GenAI Client.
    Behavior:
    1. If location is provided and NOT 'global':
       - Tries to use Vertex AI (requires GOOGLE_CLOUD_PROJECT).
    2. Else (global or no location):
       - Tries GEMINI_API_KEY / GOOGLE_API_KEY (AI Studio).
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    # Vertex AI Priority for Regional Models
    if location and location != "global":
        if project_id:
            print(f"Using Vertex AI (Project: {project_id}, Location: {location})...")
            try:
                return genai.Client(vertexai=True, project=project_id, location=location)
            except Exception as e:
                print(f"Vertex AI Initialization failed: {e}")
                return None
        else:
             print(f"Warning: Location '{location}' requested but GOOGLE_CLOUD_PROJECT is missing.")

    # AI Studio / Global Fallback
    if api_key:
        print("Using AI Studio (API Key)...")
        return genai.Client(api_key=api_key)
    
    # Vertex AI Global fallback (rare)
    if project_id:
        loc = location or "us-central1"
        print(f"Using Vertex AI (Project: {project_id}, Location: {loc}) as fallback...")
        return genai.Client(vertexai=True, project=project_id, location=loc)
            
    print("Error: No GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT found.")
    return None

def generate_text(client, prompt, model_name):
    """Generates text using initialized client."""
    if not client:
        return None

    try:
        # Generation Config
        generation_config = types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=1024,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generation_config,
        )
        return response.text
    except Exception as e:
        print(f"[{model_name}] Generation failed: {e}")
        return None

def verify_detection():
    print("--- Verifying Detection on Multiple Gemini Models with Specific Locations ---")
    
    # Initialize Detectors
    print("Initializing Detectors...")
    ppl_calc = PerplexityCalculator()
    burst_calc = BurstinessAnalyzer()
    gltr_calc = GLTRAnalyzer()
    tfidf_detector = TfidfDetector() # Should load tfidf_model.pkl
    ensemble_detector = EnsembleDetector()
    
    # Prompts to test different styles
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain the theory of relativity to a 5 year old.",
        "Write a python function to check if a number is prime.",
        "Write a persuasive email to a boss asking for a raise."
    ]
    
    results_summary = {}

    for model_config in MODELS_TO_TEST:
        model_name = model_config["name"]
        location = model_config["location"]
        
        print(f"\n=== Testing Model: {model_name} (Location: {location}) ===")
        
        # Instantiate client specifically for this model/location preference
        # Note: If using API Key, location might be ignored by get_client logic above, 
        # but if using Vertex (which seems implied by the location reqs), it will be used.
        client = get_client(location=location)
        
        if not client:
            print("Skipping due to client initialization failure.")
            results_summary[model_name] = {"detected": "Err", "avg_score": "Err"}
            continue

        success_count = 0
        total_combined_score = 0
        valid_samples = 0
        
        for i, prompt in enumerate(prompts):
            print(f"\n[Test {i+1}] Prompt: {prompt[:30]}...")
            generated_text = generate_text(client, prompt, model_name)
            
            if not generated_text:
                print("Skipping due to generation failure.")
                continue
            
            print(f"Generated ({len(generated_text)} chars). Analyzing...")
            
            # Analyze
            metrics = {}
            try: metrics['perplexity'] = ppl_calc.calculate(generated_text)
            except: pass 
            try: metrics['burstiness'] = burst_calc.analyze(generated_text)
            except: pass
            try: 
                gltr_res = gltr_calc.analyze(generated_text)
                metrics['gltr_fractions'] = gltr_calc.get_fraction_clean(gltr_res)
            except: pass
            try: 
                metrics['tfidf'] = tfidf_detector.predict(generated_text)
                metrics['tfidf_prob'] = tfidf_detector.pipeline.predict_proba([generated_text])[0][1]
            except: 
                metrics['tfidf_prob'] = 0.5
                
            # Score
            combined = get_combined_score(metrics, ensemble_detector)
            print(f">> Combined AI Likelihood: {combined:.2f}%")
            
            is_ai = combined > 50
            if is_ai:
                print(">> Verdict: Correctly Detected as AI ✅")
                success_count += 1
            else:
                print(">> Verdict: FAILED (Detected as Human) ❌")
            
            total_combined_score += combined
            valid_samples += 1

        if valid_samples > 0:
            avg_score = total_combined_score / valid_samples
            results_summary[model_name] = {
                "detected": f"{success_count}/{valid_samples}",
                "avg_score": f"{avg_score:.2f}%"
            }
        else:
            results_summary[model_name] = {"detected": "N/A", "avg_score": "N/A"}

    print("\n--- Final Summary ---")
    for model, res in results_summary.items():
        print(f"Model: {model} | Detected: {res['detected']} | Avg Confidence: {res['avg_score']}")

if __name__ == "__main__":
    verify_detection()

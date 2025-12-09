from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import time
from tqdm import tqdm

load_dotenv()

# Configuration
MODELS = [
    {"name": "gemini-3-pro-preview", "location": "global"},
    {"name": "gemini-2.5-flash", "location": "us-central1"},
    {"name": "gemini-2.5-pro", "location": "us-central1"},
    {"name": "gemini-2.0-flash", "location": "us-central1"},
]

PROMPTS = [
    # --- Standard Open-Ended ---
    "Write a paragraph about the importance of sleep.",
    "Explain quantum entanglement.",
    "Describe a sunset on Mars.",
    "Write a review of a fictional movie.",
    "Explain how a bicycle works.",
    # --- Professional / Corporate ---
    "Write a polite email declining a job offer.",
    "Draft a memo about a new dress code policy.",
    "Write a LinkedIn post announcing a product launch.",
    "Create a meeting agenda for a project kickoff.",
    # --- Creative / Literary ---
    "Write a poem about a lost key in the style of Edgar Allan Poe.",
    "Describe a character realizing they are being watched.",
    "Write a dialogue between a coffee cup and a teaspoon.",
    "Write the opening sentence of a mystery novel.",
    # --- Technical / Academic ---
    "Explain the concept of 'recursion' in programming.",
    "Summarize the causes of the French Revolution.",
    "Describe the function of mitochondria.",
    "Write a Python function to fast fourier transform.",
    "Explain the difference between TCP and UDP.",
    # --- Adversarial / Human-Mimicry (Hard) ---
    "Write a casual text message to a friend about being late, use slang, lowercase, and no punctuation.",
    "Write a rant about a bad restaurant experience, include typos and run-on sentences.",
    "Write a short story that sounds like it was written by a 10-year-old.",
    "Explain gravity but make it sound uncertain, broken, and conversational.",
    "Write a forum post asking for relationship advice, be extremely emotional and repetitive.",
    "Write a tweet about a cat video, use hashtags and emojis.",
    "Describe a dream you had, make it disjointed, confusing, and non-linear.",
    "Write a Yelp review that is angry and incoherent.",
    # --- Mixed / Abstract ---
    "Write a recipe for spicy tacos.",
    "Explain why the sky is blue.",
    "Describe the feeling of nostalgia.",
    "Write a letter to your future self.",
    "Explain the rules of chess to a complete beginner.",
    "Debate whether hotdogs are sandwiches.",
    "Describe the smell of rain."
] * 4 # Duplicate to ensure we have enough base for temperature variations

OUTPUT_FILE = config.TRAINING_DATA_PATH

def get_client(location):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return genai.Client(vertexai=True, project=project_id, location=location)
    return None

from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_sample(client, model_name, prompt, temp_offset):
    try:
        config = types.GenerateContentConfig(
            temperature=0.7 + temp_offset,
            max_output_tokens=150
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        if response.text:
            return {
                "text": response.text,
                "label": "ai",
                "source": model_name
            }
    except Exception:
        return None
    return None

def generate_batch():
    all_data = []
    
    for model_config in MODELS:
        model_name = model_config["name"]
        location = model_config["location"]
        print(f"\nGenerating samples for {model_name} ({location})...")
        
        client = get_client(location)
        if not client:
            print(f"Skipping {model_name}: No client.")
            continue
            
        tasks = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Queue 100 tasks (4 iterations of 25 prompts)
            for i in range(4):
                for prompt in PROMPTS:
                    tasks.append(executor.submit(generate_sample, client, model_name, prompt, i * 0.1))
            
            pbar = tqdm(total=len(tasks))
            for future in as_completed(tasks):
                res = future.result()
                if res:
                    all_data.append(res)
                pbar.update(1)
            pbar.close()

    print(f"Saving {len(all_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_data, f, indent=2)

if __name__ == "__main__":
    generate_batch()

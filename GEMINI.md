# Gemini Agent Development Guide

This guide provides instructions and best practices for Gemini agents working on the Text-To-AudioBook project.

## 1. Project Overview

The Text-To-AudioBook project is a fully automated pipeline that transforms text-based content into high-quality audiobooks. It leverages AI for narration, background music, and cover art.

- **Inputs**: RSS feeds, user text, or existing story files.
- **Core Features**: Story generation, dynamic Text-to-Speech (TTS), AI-generated music, and cover art creation.
- **Deployment**: The application is deployed on Google Cloud Run, with assets stored in Google Cloud Storage.
- **Web Interface**: A Streamlit-based UI for generating, managing, and browsing audiobooks.

## 2. System Architecture

The application follows a modular pipeline architecture orchestrated by `src/core/main.py`. The key components are:

- **`Orchestrator` (`src/core/orchestrator.py`)**: Manages the end-to-end workflow.
- **`GeneratorFactory` (`src/core/generator_factory.py`)**: Selects the appropriate story or audio generator.
- **Specialized Modules**:
  - `StoryGeneratorAgent`: Crafts the story.
  - `AudioGenerator`: Handles TTS.
  - `MusicGenerator`: Creates background music.
  - `ImageGenerator`: Generates cover art.
- **Data Models**: Pydantic models in `src/core/data_models.py` ensure data consistency.
- **Configuration**: Managed through environment variables and `src/core/config.py`.

## 3. Technologies and Dependencies

The project is built with Python and leverages several Google Cloud services and AI models.

- **Python Version**: `>=3.13,<4`
- **Key Libraries**:
  - `google-generativeai`: For story generation.
  - `google-cloud-texttospeech`: For TTS.
  - `google-cloud-aiplatform`: For music and image generation.
  - `streamlit`: For the web interface.
  - `pydantic`: For data modeling.
- **Dependencies**: A complete list is available in `pyproject.toml`.

## 4. Setup and Local Development

To set up the project locally:

1. **Prerequisites**:
   - Python 3.13+
   - `uv` (recommended) or `pip`
   - A Google Cloud Platform project with necessary APIs enabled (Gemini, Text-to-Speech, Imagen).

2. **System Dependencies**:
   - Install the following system dependencies:

     ```bash
     sudo apt-get update && sudo apt-get install -y portaudio19-dev libgeos-dev
     ```

3. **Virtual Environment**:
   - Create a virtual environment with Python 3.13:

     ```bash
     uv venv --python 3.13
     ```

4. **Installation**:
   - Activate the virtual environment and install the dependencies:

     ```bash
     source .venv/bin/activate
     uv pip install -r requirements.txt
     ```

5. **Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your Google API key and project details.

6. **Running the Application**:
   - **Web Interface**:

     ```bash
     streamlit run src/api/streamlit_app.py
     ```

   - **Command Line**:

     ```bash
     export INSPIRATION_MODE="USER"
     export USER_INPUT="A story about a robot who learns to paint."
     export FOLDER_MODE_TITLE="Robot Painter"
     bash run.sh
     ```

   - **Command Line with File Input**:

     When `INSPIRATION_MODE` is set to `USER`, you can also provide the input from a file. This is useful for longer inputs.

     ```bash
     export INSPIRATION_MODE="USER"
     export USER_FILE_INPUT="/path/to/your/story.txt"
     export FOLDER_MODE_TITLE="Robot Painter from File"
     bash run.sh
     ```

     *Note: If `USER_FILE_INPUT` is set, it will be used as the input, overriding `USER_INPUT`.*

## 5. Contribution Guidelines

When contributing to the project, please adhere to the following guidelines:

- **Code Style**: Follow PEP 8 and maintain consistency with the existing codebase.
- **Testing**: Run the test suite before submitting changes:

  ```bash
  ./run_tests.sh
  ```

- **Pull Requests**:
  - Create a new branch for each feature or bug fix.
  - Provide a clear description of the changes in the pull request.
  - Ensure all tests pass before requesting a review.
- **Updating Dependencies**:
  - If you add or update dependencies in `pyproject.toml`, regenerate `requirements.txt`:

    ```bash
    uv pip compile pyproject.toml -o requirements.txt
    ```

## 6. Gemini Coding Guidelines

This document outlines the coding guidelines and best practices to be followed when working on this project.

### General Principles

- **Clarity and Simplicity:** Write code that is easy to read and understand.
- **Consistency:** Follow the existing coding style and patterns in the codebase.
- **Robustness:** Write code that is resilient to errors and edge cases.

### Python Style Guide

- Follow [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
- Use an automated formatter like `black` to ensure consistent formatting.
- Use type hints for all function signatures and variables to improve code clarity and allow for static analysis.

### Naming Conventions

- **Modules:** `lowercase_with_underscores`
- **Classes & Enums:** `PascalCase`
- **Functions and Variables:** `snake_case`
- **Constants:** `UPPERCASE_WITH_UNDERSCORES`

### Data Modeling

- Use `pydantic`'s `BaseModel` for all data structures to ensure type validation and clear schema definitions.
- Define data models in the `src/core/data_models.py` file.

### Commenting

- Write docstrings for all public modules, classes, and functions, explaining their purpose, arguments, and return values.
- Use inline comments to explain *why* a specific piece of logic is implemented in a certain way, especially for complex or non-obvious code.

### Testing

- Use `pytest` as the testing framework.
- Place test files in the `tests/` directory.
- Name test files following the `test_*.py` pattern.
- Separate unit tests from integration tests. Mark tests that require external services (e.g., Google Cloud) with `@pytest.mark.skip` to avoid running them in standard test runs.

### Dependency Management

- The project uses `poetry` for dependency management.
- Add new dependencies to the `pyproject.toml` file.

### Configuration

- Centralize configuration parameters in the `src/core/config.py` module.
- Do not hardcode configuration values directly in the code.

### Project Structure

- All source code should reside within the `src/` directory.
- Organize code into logical modules (e.g., `api`, `core`, `services`, `gcp`).

### Calling the Gemini API

This section outlines the best practices for interacting with the Gemini generative AI models using the `google-genai` Python library.

#### Client Initialization

- The `genai.Client` should be initialized once and reused.
- Configure the client to use Vertex AI, sourcing credentials and configuration (e.g., `project`, `location`) from the `src/core/config.py` module.

#### Structured JSON Output with Pydantic

- To ensure reliable and validated JSON output from the model, always define a `pydantic.BaseModel` that represents the desired schema.
- This is the most critical best practice for predictable results.

#### Generation Configuration

- Use `google.genai.types.GenerateContentConfig` to bundle all generation parameters (`temperature`, `top_p`, etc.).
- To get a structured JSON output, set `response_mime_type="application/json"` and pass your Pydantic class to the `response_schema` parameter.

#### Prompt Management

- Store all prompts in a dedicated module, such as `src/core/prompts_parameters.py`. This keeps them separate from the application logic and makes them easier to manage and version.
- Use f-strings to inject dynamic data into your prompts at runtime.

#### Example Implementation

Here is a condensed example that demonstrates these best practices:

```python
import logging
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from src.core import config

# 1. Define your Pydantic schema for the output
class RankingOutput(BaseModel):
    originality: int = Field(description="How unique the concept is.")
    coherency: int = Field(description="How well the story flows.")
    overall_rating: float = Field(description="An overall score from 1.0 to 10.0.")
    justification: str = Field(description="The reasoning for the scores.")

# 2. Initialize the client using project configuration
try:
    client = genai.Client(
        vertexai=True,
        project=config.PROJECT_ID,
        location=config.LOCATION,
    )
except Exception as e:
    logging.error(f"Failed to initialize Generative AI model: {e}")
    # Handle initialization failure
    client = None

def get_story_ranking(story_content: str) -> RankingOutput | None:
    """
    Ranks a story using the Gemini model with structured output.
    """
    if not client:
        return None

    # 3. Centrally manage your prompt templates
    prompt_template = "Please rank the following story:\n{story}"

    # 4. Configure the generation settings
    try:
        generation_config = types.GenerateContentConfig(
            response_schema=RankingOutput,
            response_mime_type="application/json",
            temperature=0.9,
            max_output_tokens=8192,
        )

        # 5. Call the model and get the parsed response
        response = client.models.generate_content(
            model=config.MODEL_TEXT_GENERATION,  # Model name from config
            contents=prompt_template.format(story=story_content),
            config=generation_config,
        )

        # The .parsed attribute directly returns the populated Pydantic object
        return response.parsed

    except Exception as e:
        logging.error(f"Failed to calculate ranking: {e}")
        return None

```

#### Use only latest version of gemini model

**MODEL_TEXT_GENERATION = "gemini-3-pro-preview" # "gemini-2.5-flash" # "gemini-2.5-pro"**
**MODEL_GENERATION_FLASH = "gemini-3-pro-preview" # "gemini-2.5-flash"**
**MODEL_TTS = "gemini-2.5-pro-preview-tts" # "gemini-2.5-pro-preview-tts"**

```python

MODEL_TEXT_GENERATION = "gemini-3-pro-preview" # "gemini-2.5-flash" # "gemini-2.5-pro"
MODEL_GENERATION_FLASH = "gemini-2.5-flash" # "gemini-2.5-flash"
MODEL_TTS = "gemini-2.5-pro-tts" # "gemini-2.5-pro-preview-tts"

```

## Gemini TTS model

2 versions of GEmini TTS model can be used:

### AI studio model

Latest model: model = "gemini-2.5-pro-tts-preview-12-2025"

CODE:

#### To run this code you need to install the following dependencies

#### pip install google-genai

#### Code example

```python
import base64
import mimetypes
import os
import re
import struct
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro-tts-preview-12-2025"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Zephyr"
                )
            )
        ),
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            file_name = f"ENTER_FILE_NAME_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(chunk.text)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    generate()

```

### Vertex AI Gemini TTS model

Latest model : "gemini-2.5-pro-tts"

Code example for Vertex AI implementation:

```python
import logging
import mimetypes
import struct
from google import genai
from google.genai import types
from src.core import config

class AudioGeneratorGemini:
    def __init__(self):
        # Initialize client with Vertex AI settings (Project/Location)
        self.client = genai.Client(
            vertexai=True,
            project=config.PROJECT_ID,
            location=config.LOCATION,
        )
        self.tts_model_name = "gemini-2.5-pro-tts"

    def generate_audio_for_part(self, text: str, voice_name: str, output_path: str):
        """
        Generates audio using Vertex AI Gemini TTS.
        """
        tts_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        # Wrap content in structured objects as required
        contents_obj = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]

        try:
            logging.info(f"Calling Gemini API for audio generation...")
            response = self.client.models.generate_content(
                model=self.tts_model_name,
                contents=contents_obj,
                config=tts_config,
            )

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                part_content = response.candidates[0].content.parts[0]
                if part_content.inline_data and part_content.inline_data.data:
                    inline_data = part_content.inline_data
                    data_buffer = inline_data.data

                    # MIME type handling and WAV conversion
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    if file_extension is None:
                        # Force WAV conversion if extension unknown (likely raw PCM)
                        data_buffer = self._convert_to_wav(inline_data.data, inline_data.mime_type)

                    with open(output_path, "wb") as f:
                        f.write(data_buffer)
                    logging.info(f"Successfully saved audio to {output_path}")
                else:
                    raise Exception("No inline audio data in TTS response")
            else:
                 raise Exception("No content in TTS response")

        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")
            raise

    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Generates a WAV file header for the given audio data."""
        parameters = self._parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
            sample_rate, sample_rate * num_channels * (bits_per_sample // 8),
            num_channels * (bits_per_sample // 8), bits_per_sample,
            b"data", data_size
        )
        return header + audio_data

    def _parse_audio_mime_type(self, mime_type: str) -> dict[str, int | None]:
        """Parses bits per sample and rate from MIME type."""
        bits_per_sample = 16
        rate = 24000
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                 try:
                    rate = int(param.split("=", 1)[1])
                 except: pass
            elif param.startswith("audio/L"):
                 try:
                    bits_per_sample = int(param.split("L", 1)[1])
                 except: pass
        return {"bits_per_sample": bits_per_sample, "rate": rate}
```

# CODELAB 3: Getting Started with the Live API using Gen AI SDK

## Overview

The Live API enables low-latency bidirectional voice and video interactions with Gemini. The API can process text, audio, and video input, and it can provide text and audio output.

This tutorial demonstrates simple examples to help you get started with the Live API in Vertex AI using [Google Gen AI SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview).

### Key Capabilities:

- **Using Gemini 2.0 Flash**
  - Text-to-text generation
  - Text-to-audio generation
  - Text-to-audio conversation
  - Function calling
  - Code execution
  - Audio transcription
- **Using Gemini 2.5 Flash native audio dialog**
  - Proactive audio
  - Affective dialog

## Getting Started

### Installation

```python
%pip install --upgrade --quiet google-genai
```

### Authentication (Colab only)

```python
import sys

if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user()
```

### Required Libraries

```python
from IPython.display import Audio, Markdown, display
from google.genai.types import (
    AudioTranscriptionConfig,
    Content,
    GoogleSearch,
    LiveConnectConfig,
    Part,
    PrebuiltVoiceConfig,
    ProactivityConfig,
    SpeechConfig,
    Tool,
    ToolCodeExecution,
    VoiceConfig,
)
import numpy as np
```

### Google Cloud Project Setup

```python
import os

PROJECT_ID = "your-project-id"  # Replace with your project ID
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

from google import genai

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
```

## Using Gemini 2.0 Flash Model

The Live API is a capability introduced with the Gemini 2.0 Flash model.

```python
MODEL_ID = "gemini-2.0-flash-live-preview-04-09"
```

### Example 1: Text-to-text Generation

Send a text prompt and receive a text message.

**Important Notes:**

- A `session` represents a WebSocket connection between client and server
- Configuration includes model, parameters, system instructions, and tools
- `response_modalities` accepts `TEXT` or `AUDIO`
- Use `end_of_turn=True` to indicate server should start content generation

```python
async with client.aio.live.connect(
    model=MODEL_ID,
    config=LiveConnectConfig(response_modalities=["TEXT"]),
) as session:
    text_input = "Hello? Gemini are you there?"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    response = []

    async for message in session.receive():
        if message.text:
            response.append(message.text)

    display(Markdown(f"**Response >** {''.join(response)}"))
```

### Example 2: Text-to-audio Generation

Send a text prompt and receive an audio response.

**Available Voices:**

- `Puck`
- `Charon`
- `Kore`
- `Fenrir`
- `Aoede`
- `Leda`
- `Orus`
- `Zephyr`

```python
voice_name = "Aoede"

config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(
                voice_name=voice_name,
            )
        ),
    ),
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Hello? Gemini are you there?"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    audio_data = []
    async for message in session.receive():
        if (
            message.server_content.model_turn
            and message.server_content.model_turn.parts
        ):
            for part in message.server_content.model_turn.parts:
                if part.inline_data:
                    audio_data.append(
                        np.frombuffer(part.inline_data.data, dtype=np.int16)
                    )

    if audio_data:
        display(Audio(np.concatenate(audio_data), rate=24000, autoplay=True))
```

### Example 3: Text-to-audio Conversation

Set up an interactive conversation where you send text and receive audio responses.

**Note:** While the model tracks in-session interactions, explicit session history access isn't available yet. When a session terminates, the context is erased.

```python
config = LiveConnectConfig(response_modalities=["AUDIO"])

async def main() -> None:
    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:

        async def send() -> bool:
            text_input = input("Input > ")
            if text_input.lower() in ("q", "quit", "exit"):
                return False
            await session.send_client_content(
                turns=Content(role="user", parts=[Part(text=text_input)])
            )
            return True

        async def receive() -> None:
            audio_data = []
            async for message in session.receive():
                if (
                    message.server_content.model_turn
                    and message.server_content.model_turn.parts
                ):
                    for part in message.server_content.model_turn.parts:
                        if part.inline_data:
                            audio_data.append(
                                np.frombuffer(part.inline_data.data, dtype=np.int16)
                            )

                if message.server_content.turn_complete:
                    display(Markdown("**Response >**"))
                    display(
                        Audio(np.concatenate(audio_data), rate=24000, autoplay=True)
                    )
                    break
            return

        while True:
            if not await send():
                break
            await receive()

# Run with:
# await main()
```

### Example 4: Function Calling

Create function descriptions and pass them to the model. The response includes the function name and arguments to call it with.

**Note:**

- All functions must be declared at session start by sending tool definitions
- Currently only one tool is supported in the API

```python
def get_current_weather(location: str) -> str:
    """Example method. Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    weather_map: dict[str, str] = {
        "Boston, MA": "snowing",
        "San Francisco, CA": "foggy",
        "Seattle, WA": "raining",
        "Austin, TX": "hot",
        "Chicago, IL": "windy",
    }
    return weather_map.get(location, "unknown")

config = LiveConnectConfig(
    response_modalities=["TEXT"],
    tools=[get_current_weather],
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Get the current weather in Boston."
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    async for message in session.receive():
        if message.tool_call:
            for function_call in message.tool_call.function_calls:
                display(Markdown(f"**FunctionCall >** {str(function_call)}"))
```

### Example 5: Code Execution

Generate and execute Python code directly within the API.

````python
config = LiveConnectConfig(
    response_modalities=["TEXT"],
    tools=[Tool(code_execution=ToolCodeExecution())],
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Write tool code to calculate the 15th fibonacci number then find the nearest palindrome to it"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    response = []

    async for message in session.receive():
        if message.text:
            response.append(message.text)
        if message.server_content.model_turn:
            if message.server_content.model_turn.parts:
                for part in message.server_content.model_turn.parts:
                    if part.executable_code:
                        display(
                            Markdown(
                                f"""
**Executable code:**
```py
{part.executable_code.code}
````

"""
)
)
if part.code_execution_result:
display(
Markdown(
f"""
**Code execution result:**

```py
{part.code_execution_result.output}
```

"""
)
)

    display(Markdown(f"**Response >** {''.join(response)}"))

````

### Example 6: Google Search

Use the `google_search` tool to let the model conduct Google searches for recent information.

```python
config = LiveConnectConfig(
    response_modalities=["TEXT"],
    tools=[Tool(google_search=GoogleSearch())],
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Tell me about the largest earthquake in California the week of Dec 5 2024?"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    response = []

    async for message in session.receive():
        if message.text:
            response.append(message.text)

    display(Markdown(f"**Response >** {''.join(response)}"))
````

### Example 7: Audio Transcription

The Live API provides transcriptions for both input and output audio.

```python
config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=AudioTranscriptionConfig(),
    output_audio_transcription=AudioTranscriptionConfig(),
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Hello? Gemini are you there?"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    audio_data = []
    input_transcription = []
    output_transcription = []

    async for message in session.receive():
        if (
            message.server_content.input_transcription
            and message.server_content.input_transcription.text
        ):
            input_transcription.append(message.server_content.input_transcription)
        if (
            message.server_content.output_transcription
            and message.server_content.output_transcription.text
        ):
            output_transcription.append(
                message.server_content.output_transcription.text
            )
        if (
            message.server_content.model_turn
            and message.server_content.model_turn.parts
        ):
            for part in message.server_content.model_turn.parts:
                if part.inline_data:
                    audio_data.append(
                        np.frombuffer(part.inline_data.data, dtype=np.int16)
                    )

    if input_transcription:
        display(Markdown(f"**Input transcription >** {''.join(input_transcription)}"))

    if audio_data:
        display(Audio(np.concatenate(audio_data), rate=24000, autoplay=True))

    if output_transcription:
        display(Markdown(f"**Output transcription >** {''.join(output_transcription)}"))
```

## Using Gemini 2.5 Flash Native Audio Dialog

Gemini 2.5 Flash with Live API features enhanced native audio dialog capabilities:

- Enhanced voice quality and adaptability
- Proactive audio
- Affective dialog

**Note:** These capabilities are currently in private preview only.

```python
MODEL_ID = "gemini-2.5-flash-preview-native-audio-dialog"
```

### Example 8: Proactive Audio

When enabled, the model only responds when it's relevant. It generates transcripts and audio responses only for queries directed to the device.

```python
config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=AudioTranscriptionConfig(),
    output_audio_transcription=AudioTranscriptionConfig(),
    proactivity=ProactivityConfig(proactive_audio=True),
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Hello? Gemini are you there?"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    audio_data = []
    input_transcription = []
    output_transcription = []

    async for message in session.receive():
        if (
            message.server_content.input_transcription
            and message.server_content.input_transcription.text
        ):
            input_transcription.append(message.server_content.input_transcription)
        if (
            message.server_content.output_transcription
            and message.server_content.output_transcription.text
        ):
            output_transcription.append(
                message.server_content.output_transcription.text
            )
        if (
            message.server_content.model_turn
            and message.server_content.model_turn.parts
        ):
            for part in message.server_content.model_turn.parts:
                if part.inline_data:
                    audio_data.append(
                        np.frombuffer(part.inline_data.data, dtype=np.int16)
                    )

    if input_transcription:
        display(Markdown(f"**Input transcription >** {''.join(input_transcription)}"))

    if audio_data:
        display(Audio(np.concatenate(audio_data), rate=24000, autoplay=True))

    if output_transcription:
        display(Markdown(f"**Output transcription >** {''.join(output_transcription)}"))
```

### Example 9: Affective Dialog

When enabled, the model can understand and respond to users' emotional expressions for more nuanced conversations.

```python
config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    enable_affective_dialog=True,
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Hello? Gemini are you there? It's really a good day!"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    audio_data = []
    async for message in session.receive():
        if (
            message.server_content.model_turn
            and message.server_content.model_turn.parts
        ):
            for part in message.server_content.model_turn.parts:
                if part.inline_data:
                    audio_data.append(
                        np.frombuffer(part.inline_data.data, dtype=np.int16)
                    )

    if audio_data:
        display(Audio(np.concatenate(audio_data), rate=24000, autoplay=True))
```

## What's Next

- Learn how to [build a web application that enables you to use your voice and camera to talk to Gemini 2.0 through the Live API](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/multimodal-live-api/websocket-demo-app)
- See the [Live API reference docs](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-live)
- See the [Google Gen AI SDK reference docs](https://googleapis.github.io/python-genai/)
- Explore other notebooks in the [Google Cloud Generative AI GitHub repository](https://github.com/GoogleCloudPlatform/generative-ai)

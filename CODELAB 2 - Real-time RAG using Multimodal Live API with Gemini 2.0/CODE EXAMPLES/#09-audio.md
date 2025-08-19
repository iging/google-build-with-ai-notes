# Example 9-A: Audio Generation with Multimodal Live API

This example demonstrates how to generate audio responses using the Multimodal Live API with Gemini 2.0.

```python
import asyncio
import numpy as np
from google.genai.types import LiveConnectConfig
from IPython.display import Audio, display

async def generate_audio_content(query: str):
    """Function to generate audio response for provided query using Gemini Multimodal Live API.

    Args:
      query: The query to generate audio response for.

    Returns:
      Displays the audio response (no return value).
    """
    # Configure the session for audio output
    config = LiveConnectConfig(response_modalities=["AUDIO"])

    # Establish connection with the model
    async with client.aio.live.connect(model=MODEL, config=config) as session:

        # Send the query and mark end of turn
        await session.send(input=query, end_of_turn=True)

        # Collect audio data parts
        audio_parts = []
        async for message in session.receive():
            if message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data:
                        # Convert binary audio data to numpy array
                        audio_parts.append(
                            np.frombuffer(part.inline_data.data, dtype=np.int16)
                        )

            # Check if the response is complete
            if message.server_content.turn_complete:
                if audio_parts:
                    # Combine all audio segments
                    audio_data = np.concatenate(audio_parts, axis=0)
                    # Brief pause for better user experience
                    await asyncio.sleep(0.4)
                    # Play the audio
                    display(Audio(audio_data, rate=24000, autoplay=True))
                break

# Test the function with our sample query
async def test_audio_generation():
    query = "What is the price of a basic tune-up at Cymbal Bikes?"

    print(f"Query: {query}")
    print("\nGenerating audio response... (listen when complete)")

    await generate_audio_content(query)

# Execute the test function
await test_audio_generation()
```

## Explanation

This code demonstrates how to use the Multimodal Live API to generate audio responses directly:

1. **Audio Configuration**:

   - Uses `LiveConnectConfig` with `response_modalities=["AUDIO"]` to specify we want audio output.
   - This is a key difference from the text generation example - we're requesting audio directly from the API.
   - The model will generate speech without requiring a separate text-to-speech step.

2. **Session Management**:

   - Creates an asynchronous session with the Gemini model using `client.aio.live.connect()`.
   - Uses an async context manager to ensure proper resource handling.
   - Sends the query with `session.send()` and `end_of_turn=True` to mark the end of user input.

3. **Audio Data Processing**:

   - Processes the streaming response using `async for` to handle chunks as they arrive.
   - Inspects each message for the presence of `inline_data` in model turn parts.
   - Converts the binary audio data to a NumPy array using `np.frombuffer`.
   - Specifies the data type as 16-bit integers (`dtype=np.int16`), which is the format used by the API.
   - Collects all audio chunks in the `audio_parts` list.

4. **Audio Playback**:

   - Detects the completion of the response with the `turn_complete` flag.
   - Combines all audio segments using `np.concatenate()` to create a single continuous audio stream.
   - Introduces a small delay with `asyncio.sleep(0.4)` for better user experience.
   - Plays the audio using `IPython.display.Audio` with:
     - A sample rate of 24000 Hz (the rate used by the Gemini Live API)
     - `autoplay=True` to start playback automatically

5. **Direct Audio Generation Benefits**:

   - **Efficiency**: Bypasses the separate text-to-speech step, reducing latency.
   - **Voice Quality**: The model's built-in voice generation often provides more natural intonation.
   - **Streaming**: Audio can begin playing before the entire response is complete.
   - **Integration**: Simplified pipeline with fewer components and potential points of failure.

This approach is particularly valuable for applications requiring natural-sounding voice responses with minimal latency, such as voice assistants, accessibility applications, and interactive customer support systems.

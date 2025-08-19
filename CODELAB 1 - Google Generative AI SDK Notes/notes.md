# CODELAB 1: Google Generative AI SDK Notes

## Introduction

- Google Gen AI SDK provides a unified interface to Google's generative AI services
- Simplifies integration of generative AI capabilities into applications
- Enables developers to leverage Google's advanced AI models for various tasks

## Getting Started

- Install the SDK using: `pip install google-genai`
- Two API service options:
  - **Google AI for Developers**: For experimentation, prototyping, small projects
  - **Vertex AI**: For enterprise-ready projects on Google Cloud

## Core Capabilities

### Basic Text Generation

- Use `generate_content()` method for text prompts
- Simple text responses with `.text` property
- Example: "What's the largest planet in our solar system?"

### Multimodal Prompts

- Include text, images, PDFs, audio, and video in your prompts
- Images can be passed as files or URLs using `Part.from_uri()`
- Generate text responses from multimodal inputs

### Model Control

#### System Instructions

- Steer model behavior with system instructions
- Provide additional context for specific tasks
- Example: "You are a helpful language translator"

#### Model Parameters

- Customize generation with parameters:
  - `temperature`: Controls randomness (lower = more deterministic)
  - `top_p` and `top_k`: Influence token selection diversity
  - `seed`: For reproducible outputs
  - `max_output_tokens`: Limit response length

#### Safety Filters

- Adjust safety thresholds across categories:
  - Dangerous content
  - Harassment
  - Hate speech
  - Sexually explicit content
- Block harmful content with `HarmBlockThreshold`
- Inspect safety ratings with `candidates[0].safety_ratings`

### Advanced Features

#### Multi-turn Chat

- Create persistent conversations with `chats.create()`
- Send multiple messages in sequence with `chat.send_message()`
- Maintain context across conversation turns

#### Controlled Output

- Constrain model output to structured formats
- Define schemas as Pydantic Models or JSON
- Generate structured data (e.g., JSON) from natural language

#### Streaming Responses

- Stream content as it's generated with `generate_content_stream()`
- Process response chunks as they arrive

#### Asynchronous Requests

- Send async requests with `client.aio.models`
- Use `await` with async methods like `client.aio.models.generate_content()`

#### Token Management

- Count tokens before sending with `count_tokens()`
- Calculate token usage with `compute_tokens()`

#### Function Calling

- Define functions that the model can call
- Create `FunctionDeclaration` with name, description, parameters
- Group declarations into `Tool` objects
- Receive structured function calls from the model

#### Context Caching

- Store frequently used inputs in dedicated cache
- Reference cached content in subsequent requests
- Improve efficiency with repeated contexts (PDFs, system instructions)
- Available only for stable models with fixed versions

#### Batch Prediction

- Process large numbers of requests asynchronously
- More efficient and cost-effective for non-latency-sensitive tasks
- Accept inputs from BigQuery or Cloud Storage
- Output results to Cloud Storage or BigQuery

#### Text Embeddings

- Generate vector representations with `embed_content()`
- Configure output dimensions (1-768)
- Use for semantic search, clustering, classification

## Getting Started Tips

- Begin with simple text prompts to understand basics
- Experiment with different parameters to find optimal settings
- Use safety filters appropriate for your use case
- Consider batch processing for large-scale tasks
- Leverage context caching for repeated content

## Resources

- [Google Cloud Generative AI GitHub repository](https://github.com/GoogleCloudPlatform/generative-ai)
- [Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models) for exploring AI models
- [Google Colab Notebook with Examples](https://colab.research.google.com/drive/1rV16-wPdhtQA-MyhIiPNnDJeR0A3Rwjm?usp=drive_link)

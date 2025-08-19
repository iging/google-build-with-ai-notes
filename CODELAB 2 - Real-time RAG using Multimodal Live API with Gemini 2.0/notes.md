# CODELAB 2: Real-time RAG using Multimodal Live API with Gemini 2.0

## Colab Link

- [Google Colab Notebook](https://colab.research.google.com/drive/1bJlslEpR6qgsDCDC-zxHkIeu4HuaGx3q?usp=drive_link)

## Introduction

- Multimodal Live API enables building real-time applications with Gemini 2.0
- Supports text input with text and audio output
- Provides lower latency than traditional generation approaches
- Uses WebSockets for continuous connection and real-time communication

## Gemini 2.0 Features

- Multimodal Live API for real-time vision and audio streaming with tool use
- 3x improvement in time to first token (TTFT) over 1.5 Flash
- Quality comparable to larger models like Gemini 2.0 and GPT-4o
- Improved multimodal understanding, coding, complex instruction following
- Native image generation and controllable text-to-speech capabilities

## Retrieval Augmented Generation (RAG)

- RAG helps ground LLM responses in specific, accurate information
- Prevents hallucinations by providing relevant context from reliable sources
- Critical for domain-specific applications where accuracy is essential
- Combines information retrieval with language generation

## RAG Architecture Components

### Data Preparation

1. **Chunking**: Dividing documents into smaller, manageable segments
2. **Embedding**: Transforming text chunks into numerical vectors representing semantic meaning
3. **Indexing**: Organizing embeddings for efficient similarity search

### Inference

1. **Retrieval**: Finding the most relevant chunks based on query embedding
2. **Query Augmentation**: Enhancing the query with retrieved context
3. **Generation**: Creating a coherent, informative answer based on the augmented query

## Implementing RAG with Multimodal Live API

### Document Processing

- Extract text from PDFs and segment into manageable chunks
- Generate embeddings for each chunk using Vertex AI text embedding models
- Build a searchable index using pandas DataFrame (or use more advanced vector stores)
- Store metadata with each chunk (document name, page number, etc.)

### Retrieval Mechanism

- Generate embedding for the user query
- Calculate similarity between query embedding and document chunk embeddings
- Use cosine similarity or other similarity metrics
- Rank and retrieve the most relevant chunks
- Return top-k chunks based on similarity scores

### Generation with Multimodal Live API

- Construct prompt with retrieved context and user query
- Send through Multimodal Live API with desired output modality (text or audio)
- Process the response in real-time as it's generated
- Return or play generated content based on selected modality

## Implementation Details

### Setting Up

```python
# Install required packages
%pip install --upgrade --quiet google-genai PyPDF2

# Initialize Vertex AI client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# Set model IDs
MODEL_ID = "gemini-2.0-flash-live-preview-04-09"
MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
text_embedding_model = "text-embedding-005"
```

### Multimodal Live API Usage

#### Text Generation

```python
async def generate_content(query: str) -> str:
    config = LiveConnectConfig(response_modalities=["TEXT"])
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send(input=query, end_of_turn=True)

        response = []
        async for message in session.receive():
            try:
                if message.text:
                    response.append(message.text)
            except AttributeError:
                pass

            if message.server_content.turn_complete:
                response = "".join(str(x) for x in response)
                return response
```

#### Audio Generation

```python
async def generate_audio_content(query: str):
    config = LiveConnectConfig(response_modalities=["AUDIO"])
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send(input=query, end_of_turn=True)

        audio_parts = []
        async for message in session.receive():
            if message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data:
                        audio_parts.append(
                            np.frombuffer(part.inline_data.data, dtype=np.int16)
                        )

            if message.server_content.turn_complete:
                if audio_parts:
                    audio_data = np.concatenate(audio_parts, axis=0)
                    await asyncio.sleep(0.4)
                    display(Audio(audio_data, rate=24000, autoplay=True))
                break
```

### Building the RAG Pipeline

#### Creating Document Embeddings

```python
def build_index(document_paths, embedding_client, embedding_model, chunk_size=512):
    # Extract text from PDFs
    # Divide text into chunks
    # Generate embeddings for each chunk
    # Return indexed DataFrame
```

#### Retrieving Relevant Context

```python
def get_relevant_chunks(query, vector_db, embedding_client, embedding_model, top_k=3):
    # Generate embedding for query
    # Calculate similarity with document chunks
    # Return top-k most relevant chunks
```

#### Generating Grounded Responses

```python
async def generate_answer(query, context, llm_client, modality="text"):
    # Construct prompt with context and query
    # Generate response using appropriate modality
    # Return generated answer
```

#### Complete RAG Pipeline

```python
async def rag(question, vector_db, embedding_client, embedding_model,
             llm_client, top_k, llm_model, modality="text"):
    # Get relevant context
    # Generate answer using context
    # Return response in requested modality
```

## Key Benefits of This Approach

### Accuracy and Reliability

- Grounds responses in verified information from provided documents
- Reduces hallucinations and unsupported statements
- Provides citations or sources for information

### Real-time Capabilities

- Low latency response generation
- Direct audio output without separate text-to-speech step
- Interactive, conversational experience

### Flexibility

- Supports both text and audio output from the same pipeline
- Easily adaptable to different document sets
- Can be extended to support additional modalities

## Optimization Tips

### For Retrieval

- Experiment with different chunk sizes to balance context and relevance
- Try different embedding models for improved semantic representation
- Consider more sophisticated vector stores for large-scale applications
- Implement filters based on metadata for targeted retrieval

### For Generation

- Fine-tune system prompts for more accurate responses
- Balance context length with model performance
- Consider implementing streaming for immediate feedback
- Use asynchronous operations for improved responsiveness

## Potential Applications

- Customer support systems with voice capabilities
- Interactive knowledge bases and documentation
- Educational tools with multimedia responses
- Accessibility applications requiring audio output
- Domain-specific assistants grounded in proprietary information

## Resources

- [Multimodal Live API reference docs](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-live)
- [Google Gen AI SDK reference docs](https://googleapis.github.io/python-genai/)
- [Vertex AI RAG documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings)
- [Building web applications with Multimodal Live API](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/multimodal-live-api/websocket-demo-app)

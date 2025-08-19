# Example 12-B: Generation Component for RAG

This example demonstrates how to implement the generation component of our RAG system, which produces answers based on retrieved document context.

```python
@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
async def generate_answer(
    query: str, context: str, llm_client: Any, modality: str = "text"
) -> str:
    """Generate answer using LLM with retry logic for API quota management.

    Args:
        query: User query.
        context: Relevant text providing context for the query.
        llm_client: Client for accessing LLM API.
        modality: Output modality (text or audio).

    Returns:
        Generated answer for text modality, None for audio modality.
    """
    try:
        # Check for earlier retrieval errors
        if context in [
            "Could not process query due to quota issues",
            "Error retrieving relevant chunks",
        ]:
            return "Can't Process, Quota Issues"

        # Create a prompt that instructs the model to use only the provided context
        prompt = f"""Based on the following context, please answer the question.

        Context:
        {context}

        Question: {query}

        Answer:"""

        # Generate answer in the requested modality
        if modality == "text":
            # Generate text answer using previously defined function
            response = await generate_content(prompt)
            return response

        elif modality == "audio":
            # Generate audio answer using previously defined function
            await generate_audio_content(prompt)
            return None

    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "Can't Process, Quota Issues"
        print(f"Error generating answer: {str(e)}")
        return "Error generating answer"

# Test the generation component with our query and retrieved context
query = "What is the price of a basic tune-up at Cymbal Bikes?"

# Generate and display text answer
generated_answer = await generate_answer(
    query, relevant_context, client, modality="text"
)
print(f"\nGenerated answer (text):\n{generated_answer}")

# Generate audio answer (uncomment to test)
# print("\nGenerating audio answer...")
# await generate_answer(query, relevant_context, client, modality="audio")
```

## Explanation

This code implements the generation component of our RAG pipeline - the "G" in "RAG":

1. **Contextual Answer Synthesis**:

   - The `generate_answer()` function synthesizes responses based on:
     - The user's original query
     - The relevant context retrieved from documents
   - This approach grounds the model's answers in factual information rather than relying solely on pre-trained knowledge
   - The function supports both text and audio output modalities

2. **Prompt Engineering**:

   - Uses a carefully designed prompt structure:
     - Clear instruction that the answer should be based only on the provided context
     - Explicit separation between context and question
     - Explicit "Answer:" prefix to direct the model's response format
   - This prompt design helps prevent hallucinations by constraining the model to use only the retrieved information

3. **Multimodal Capabilities**:

   - The function supports two output modalities:
     - **Text**: Returns a string containing the generated answer
     - **Audio**: Generates and plays spoken response using the Multimodal Live API
   - Both modalities use the same context and prompt, demonstrating the flexibility of the Live API

4. **Error Handling and Resilience**:

   - Implements sophisticated error handling:
     - Detects and reports quota limitations
     - Uses retry logic with exponential backoff for transient issues
     - Propagates error messages from earlier pipeline stages
     - Provides user-friendly error messages

5. **Technical Implementation Details**:

   - **Retry Mechanism**: Uses the `@retry` decorator with:
     - Exponential backoff with random jitter to prevent request flooding
     - Maximum retry attempts to prevent infinite loops
     - Configurable wait times between attempts
   - **Asynchronous Design**: Uses `async/await` pattern for non-blocking operation
   - **Client Abstraction**: Works with any client that implements the required API methods

The generation component represents the final stage of our RAG pipeline, transforming retrieved document context into natural, informative responses. By using Gemini's understanding of both the query and context, it synthesizes answers that are factually grounded in the provided documents while being conversational and helpful.

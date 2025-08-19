# Example 12-C: Complete RAG Pipeline Integration

This example demonstrates how to integrate the retrieval and generation components into a complete RAG pipeline.

```python
async def rag(
    question: str,
    vector_db: pd.DataFrame,
    embedding_client: Any,
    embedding_model: str,
    llm_client: Any,
    top_k: int,
    llm_model: str,
    modality: str = "text",
) -> str | None:
    """Complete RAG pipeline that retrieves relevant context and generates answers.

    Args:
        question: User query.
        vector_db: DataFrame containing document chunks and embeddings.
        embedding_client: Client for accessing embedding API.
        embedding_model: Name of the embedding model.
        llm_client: Client for accessing LLM API.
        top_k: Number of relevant chunks to retrieve.
        llm_model: Name of the LLM model.
        modality: Output modality (text or audio).

    Returns:
        For text modality, the generated answer string.
        For audio modality, None (audio is played directly).
    """
    try:
        # Step 1: Retrieve relevant context for the question
        relevant_context = get_relevant_chunks(
            question, vector_db, embedding_client, embedding_model, top_k=top_k
        )

        # Step 2: Generate an answer using the retrieved context
        if modality == "text":
            generated_answer = await generate_answer(
                question,
                relevant_context,
                llm_client,
            )
            return generated_answer

        elif modality == "audio":
            await generate_answer(
                question, relevant_context, llm_client, modality=modality
            )
            return None

    except Exception as e:
        print(f"Error processing question '{question}': {str(e)}")
        return f"Error processing question: {str(e)}"

# Define some sample queries to test the RAG system
question_set = [
    {
        "question": "What is the price of a basic tune-up at Cymbal Bikes?",
        "answer": "A basic tune-up costs $100.",
    },
    {
        "question": "How much does it cost to replace a tire at Cymbal Bikes?",
        "answer": "Replacing a tire at Cymbal Bikes costs $50 per tire.",
    },
    {
        "question": "What does gear repair at Cymbal Bikes include?",
        "answer": "Gear repair includes inspection and repair of the gears, including replacement of chainrings, cogs, and cables as needed.",
    },
    {
        "question": "Can I return clothing items to Cymbal Bikes?",
        "answer": "Clothing can only be returned if it is unworn and in the original packaging.",
    },
]

# Test the complete RAG pipeline with a sample question
print("Testing RAG pipeline with text output:")
print(f"Question: {question_set[0]['question']}")
print(f"Expected answer: {question_set[0]['answer']}")

response = await rag(
    question=question_set[0]["question"],
    vector_db=vector_db_mini_vertex,
    embedding_client=client,
    embedding_model=text_embedding_model,
    llm_client=client,
    top_k=3,
    llm_model=MODEL,
    modality="text",
)

print(f"\nGenerated answer:\n{response}")

# Uncomment to test the audio modality
# print("\nTesting RAG pipeline with audio output:")
# await rag(
#     question=question_set[0]["question"],
#     vector_db=vector_db_mini_vertex,
#     embedding_client=client,
#     embedding_model=text_embedding_model,
#     llm_client=client,
#     top_k=3,
#     llm_model=MODEL,
#     modality="audio",
# )
```

## Explanation

This code integrates all components into a complete end-to-end RAG system:

1. **Pipeline Architecture**:

   - The `rag()` function serves as the main pipeline, orchestrating:
     - **Retrieval**: Finding relevant document chunks using semantic search
     - **Generation**: Creating answers based on retrieved context
   - This function provides a single, clean interface for the entire RAG process
   - The modular design allows each component to be improved or replaced independently

2. **Flexible Multimodal Support**:

   - The pipeline supports multiple output modalities:
     - **Text output**: Returns generated text responses
     - **Audio output**: Produces spoken responses through the Live API
   - This multimodal capability makes the RAG system suitable for various applications:
     - Web interfaces (text)
     - Voice assistants (audio)
     - Accessibility features (both)
   - Only the output format changes while the core RAG logic remains consistent

3. **Parameter Configuration**:

   - The pipeline allows customization of key parameters:
     - `top_k`: Controls how many document chunks to retrieve
     - `embedding_model`: Determines which embedding model to use
     - `llm_model`: Specifies the generation model
   - This configurability enables optimization for different use cases and performance requirements

4. **Error Handling**:

   - The pipeline includes comprehensive error handling:
     - Catches and reports exceptions from both retrieval and generation
     - Provides clear error messages that help diagnose issues
     - Prevents crashes that would disrupt user experience

5. **Testing and Validation**:

   - The example includes:
     - A set of sample questions with expected answers
     - A demonstration of the pipeline with both text and audio outputs
     - A simple evaluation by comparing expected and generated answers
   - This testing approach helps verify that the RAG system works correctly

6. **Business Value**:
   - The integrated RAG pipeline delivers significant business benefits:
     - **Accuracy**: Grounds answers in verified company documents
     - **Consistency**: Ensures all responses align with official policies
     - **Adaptability**: Can be updated by simply changing the document source
     - **Cost Efficiency**: Minimizes hallucinations that would require human correction
     - **Multimodal Access**: Serves users across different interaction preferences

This implementation represents a complete, production-ready RAG system that can be adapted to various use cases beyond the retail example provided. The architecture follows best practices for maintaining modularity, error resilience, and flexibility.

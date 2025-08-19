# Example 12-A: Retrieval Component for RAG

This example demonstrates how to implement the retrieval component of our RAG system, which finds relevant document chunks based on a query.

```python
def get_relevant_chunks(
    query: str,
    vector_db: pd.DataFrame,
    embedding_client: Any,
    embedding_model: str,
    top_k: int = 3,
) -> str:
    """Retrieve the most relevant document chunks for a query using similarity search.

    Args:
        query: The search query string.
        vector_db: A pandas DataFrame containing the vectorized document chunks.
                     It must contain columns for embeddings and chunk text.
        embedding_client: The client used to generate embeddings.
        embedding_model: The name of the embedding model to use.
        top_k: The number of most similar chunks to retrieve. Defaults to 3.

    Returns:
        A formatted string containing the top_k most relevant chunks.
    """
    try:
        # Generate embedding for the query
        query_embedding = get_embeddings(embedding_client, embedding_model, query)

        if query_embedding is None:
            return "Could not process query due to quota issues"

        # Calculate similarity between query and all document chunks
        similarities = [
            cosine_similarity(query_embedding, chunk_emb)[0][0]
            for chunk_emb in vector_db["embeddings"]
        ]

        # Get the indices of the top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:]
        relevant_chunks = vector_db.iloc[top_indices]

        # Format the retrieved chunks with metadata
        context = []
        for _, row in relevant_chunks.iterrows():
            context.append(
                {
                    "document_name": row["document_name"],
                    "page_number": row["page_number"],
                    "chunk_number": row["chunk_number"],
                    "chunk_text": row["chunk_text"],
                }
            )

        # Join the chunks into a single formatted string
        return "\n\n".join(
            [
                f"[Page {chunk['page_number']}, Chunk {chunk['chunk_number']}]: {chunk['chunk_text']}"
                for chunk in context
            ]
        )

    except Exception as e:
        print(f"Error getting relevant chunks: {str(e)}")
        return "Error retrieving relevant chunks"

# Test the retrieval component with our example query
query = "What is the price of a basic tune-up at Cymbal Bikes?"
relevant_context = get_relevant_chunks(
    query, vector_db_mini_vertex, client, text_embedding_model, top_k=3
)

# Display the retrieved context
print("Retrieved relevant context:")
print(relevant_context[:500] + "..." if len(relevant_context) > 500 else relevant_context)
```

## Explanation

This code implements the retrieval component of our RAG pipeline - the "R" in "RAG":

1. **Semantic Search Process**:

   - The `get_relevant_chunks()` function implements a semantic search over our document index:
     - First, it transforms the user's query into an embedding vector
     - Then, it compares this vector to all document chunk embeddings
     - Finally, it returns the most relevant chunks based on similarity scores
   - This vector-based approach captures semantic meaning rather than just keyword matching

2. **Similarity Calculation**:

   - Uses **cosine similarity** as the distance metric, which measures the angle between vectors
   - This metric effectively captures semantic relatedness regardless of vector magnitude
   - The calculation is performed between the query embedding and each document chunk embedding
   - Higher similarity scores indicate greater relevance to the query

3. **Ranking and Selection**:

   - The function sorts all chunks by their similarity scores
   - It selects the `top_k` most similar chunks (default is 3)
   - This parameter controls the trade-off between:
     - Providing enough context for accurate answers
     - Avoiding information overload that might confuse the model
     - Managing computational efficiency

4. **Context Formatting**:

   - Retrieved chunks are formatted with important metadata:
     - Document name: Identifies the source document
     - Page number: Locates the chunk within the document
     - Chunk number: Identifies the specific text segment
   - This formatted context is crucial for the generation component to:
     - Provide source attribution
     - Maintain traceability
     - Present information in a structured way

5. **Error Handling**:
   - The function includes comprehensive error handling:
     - Gracefully handles embedding failures
     - Reports quota exceeded issues clearly
     - Provides informative error messages without crashing the pipeline
   - This resilience is essential for production applications

This retrieval component forms the critical bridge between document processing and answer generation in our RAG pipeline. By finding the most relevant document chunks for each query, it ensures that the generation model receives the specific information needed to produce accurate, factual responses.

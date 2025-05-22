def condense_context(chunks, tokenizer, model, device="cpu", max_chunks=3):
    """
    Summarize the retrieved chunks into a single condensed context.

    Args:
        chunks (list): List of text chunks retrieved by the search index.
        tokenizer: Huggingface tokenizer for the language model.
        model: The language model to generate summaries.
        device (str): Device for model inference (e.g., 'cpu' or 'cuda').
        max_chunks (int): Max number of chunks to include in the summary prompt.

    Returns:
        str: A concise summary of the retrieved chunks.
    """
    
    selected_chunks = chunks[:max_chunks]
    
 
    joined = "\n".join(selected_chunks)

    # Create summarization prompt
    prompt = (
        "Summarize the following content in a clear, concise way to help answer a user's question:\n\n"
        f"{joined}\n\nSummary:"
    )

    # Generate summary using your LLM-based generator
    from llm_inference import generate_answer
    return generate_answer(tokenizer, model, prompt, device)

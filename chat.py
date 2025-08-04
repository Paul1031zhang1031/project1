import tiktoken
import time

# count tokens in a string.
def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
   
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

#  Summarizes text using a Map-Reduce strategy.
def get_summary(client, text_to_summarize: str, model_name: str, prompt_title: str):
    if not client:
        return "API Client is not initialized."
    
    MAX_TOKENS = 4000
    if num_tokens_from_string(text_to_summarize, model_name) <= MAX_TOKENS:
        prompt = f"Please provide a concise summary of {prompt_title}:\n\n{text_to_summarize}"
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred during summarization: {e}"

    # Map-Reduce logic for large texts
    sub_chunks = [text_to_summarize[i:i+8000] for i in range(0, len(text_to_summarize), 8000)]
    partial_summaries = []
    
    for i, chunk in enumerate(sub_chunks):
        prompt = f"This is part {i+1} of a larger document. Please summarize just this part:\n\n{chunk}"
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
            partial_summaries.append(response.choices[0].message.content)
            time.sleep(2) # Respect rate limits
        except Exception as e:
            print(f"Error on summary chunk {i+1}: {e}. Skipping.")
            continue
            
    if not partial_summaries:
        return "Map-Reduce failed as all sub-chunk summaries resulted in an error."

    combined_text = "\n\n---\n\n".join(partial_summaries)
    final_prompt = f"Synthesize these partial summaries into one final, comprehensive summary:\n\n{combined_text}"
    try:
        final_response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": final_prompt}])
        return final_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during final summary synthesis: {e}"


# Generates a final answer from a pre-selected context provided by the RAG engine.
def get_qa_answer(client, question: str, context: str, model_name: str):
   
    if not client:
        return "API Client is not initialized."
    
    if not context.strip() or len(context.strip().split()) < 30:
       return "I'm sorry, the document does not contain enough relevant information to answer that question."
    
    final_prompt = f"""
    Based ONLY on the context provided below, answer the user's question.
    If the answer is not found in the context, respond with "I'm sorry, the answer could not be found in the document."

    CONTEXT:\n---\n{context}\n---\nQUESTION: {question}"""
    try:
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": final_prompt}])
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"
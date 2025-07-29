
import streamlit as st
import tiktoken
import time
import re

def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def summarize_text_map_reduce(client, text_to_summarize: str, model_name: str, prompt_title: str = "the following text"):
    """
    Summarizes text using a Map-Reduce strategy, with a safeguard for empty results.
    """
    if not client:
        return {"final_summary": "API Client is not initialized.", "partial_summaries": []}

    MAX_TOKENS_FOR_SINGLE_REQUEST = 4000

    if num_tokens_from_string(text_to_summarize, model_name) <= MAX_TOKENS_FOR_SINGLE_REQUEST:
        prompt = f"Please provide a concise summary of {prompt_title}:\n\n{text_to_summarize}"
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
            return {"final_summary": response.choices[0].message.content, "partial_summaries": []}
        except Exception as e:
            return {"final_summary": f"An error occurred during direct summarization: {e}", "partial_summaries": []}

    st.text(f"  -> Document is large, applying Map-Reduce for {prompt_title}...")
    
    sub_chunks = [text_to_summarize[i:i+8000] for i in range(0, len(text_to_summarize), 8000)]
    partial_summaries = []
    
    with st.expander(f"  -> Show detailed Map-Reduce progress ({len(sub_chunks)} chunks)", expanded=True):
        for i, chunk in enumerate(sub_chunks):
            st.write(f"Processing chunk {i+1} of {len(sub_chunks)}...")
            prompt = f"This is part {i+1} of a larger document. Please summarize just this part:\n\n{chunk}"
            try:
                response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
                partial_summaries.append(response.choices[0].message.content)
                time.sleep(2)
            except Exception as e:
                st.error(f"Error on chunk {i+1}: {e}. Skipping this chunk.")
                time.sleep(5)
                continue

 
    # After the loop, check if the list of summaries is empty.
    if not partial_summaries:
        return {
            "final_summary": f"The Map-Reduce process failed for {prompt_title} because all sub-chunk summarizations resulted in an error. This is likely due to API rate limits. Please check the errors in the progress expander above.",
            "partial_summaries": []
        }


    with st.spinner(f"  -> Combining partial summaries for {prompt_title}..."):
        combined_text = "\n\n---\n\n".join(partial_summaries)
        #final_prompt = f"The following are several partial summaries of {prompt_title}. Synthesize them into one final, comprehensive summary."
        combined_text = "\n\n---\n\n".join(partial_summaries)
        final_prompt = (
        f"The following are several partial summaries of {prompt_title}. "
        "Synthesize them into one final, comprehensive summary:\n\n"
        f"{combined_text}"
       )


        try:
            final_response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": final_prompt}])
            return {"final_summary": final_response.choices[0].message.content, "partial_summaries": partial_summaries}
        except Exception as e:
            return {"final_summary": f"An error occurred during final synthesis: {e}", "partial_summaries": partial_summaries}

def answer_question_within_section(client, question: str, selected_chunk: dict, model_name: str):
    """
    Answers a question based ONLY on the text of a single, user-selected document section.
    """
    if not client: return "API Client is not initialized."

    context = selected_chunk['text']
    section_title = selected_chunk['title']
    MAX_TOKENS_FOR_CONTEXT = 4000

    if num_tokens_from_string(context, model_name) > MAX_TOKENS_FOR_CONTEXT:
        distill_prompt = f"From the following text (from section '{section_title}'), extract ONLY the information directly relevant to answering this question: \"{question}\"\n\nTEXT:\n{context[:16000]}"
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": distill_prompt}])
            context = response.choices[0].message.content
            st.info("Context distilled. Generating final answer.")
            time.sleep(2)
        except Exception as e:
            return f"An error occurred while distilling the context: {e}"

    final_prompt = f"""
    Based strictly on the context provided from document section '{section_title}', answer the user's question.
    - If the answer is found in the context, provide only the answer.
    - If the answer is NOT found in the context, respond with only the phrase: "The answer to that question is not found in the '{section_title}' section."
    Do not add any conversational text or explanations.

    CONTEXT:
    ---
    {context}
    ---
    QUESTION: {question}
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict, factual Q&A engine. Your task is to answer questions using only the provided text."},
                {"role": "user", "content": final_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the final answer: {e}"
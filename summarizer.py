import fitz
import streamlit as st
import numpy as np
from transformers import AutoTokenizer
from chat import num_tokens_from_string
import time

# --- PDF Extraction ---
def ext_text_from_pdf(file):
    if hasattr(file, "read"):
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# --- Chunking ---
def split_text_into_chunks(text, max_tokens=1000):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        tentative_chunk = current_chunk + sentence + ". "
        if len(tokenizer.tokenize(tentative_chunk)) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        else:
            current_chunk = tentative_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# --- Summarization ---
def summarize_chunks(client, chunks, model, chunk_limit=5):
    # FIX: Made client a required positional argument and added a guard clause.
    if not client:
        st.error("API Client not provided to summarize_chunks.")
        return []
        
    summaries = []
    for i, chunk in enumerate(chunks[:chunk_limit]):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": f"Summarize this chunk:\n\n{chunk}"}
                ],
                max_tokens=256,
                temperature=0.2
            )
            summary = response.choices[0].message.content.strip()
            summaries.append(summary)
        except Exception as e:
            st.error(f"❌ Error summarizing chunk {i+1}: {e}")
            continue
    return summaries

def create_context_from_summaries(client, text, model_name="llama3-8b-8192", limit=5):
    # FIX: Made client a required positional argument.
    chunks = split_text_into_chunks(text)
    # FIX: Pass the client to the corrected summarize_chunks function.
    summaries = summarize_chunks(client, chunks, model=model_name, chunk_limit=limit)
    return "\n---\n".join(summaries)

# --- Question Answering ---
def ask_question(client, question, context, model_name, system_prompt="Answer the question based on the provided context."):
    # FIX: Made client a required positional argument and removed client=None from the end.
    if not client:
        st.error(f"API Client not provided to ask_question for model `{model_name}`.")
        return "⚠️ Unable to generate answer due to an internal configuration error."
        
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    token_count = num_tokens_from_string(prompt, model_name)
    st.info(f"🔢 Tokens in request: {token_count}")

    try:
        time.sleep(2)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # FIX: Changed the error message slightly to distinguish from the original output, providing more clarity.
        st.error(f"❌ Error during question answering with model `{model_name}`: {e}")
        return f"⚠️ Unable to generate answer for model `{model_name}` due to an API error."
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
def summarize_chunks(chunks, model, chunk_limit=5, client=None):
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

def create_context_from_summaries(text, model_name="llama3-8b-8192", limit=5, client=None):
    chunks = split_text_into_chunks(text)
    summaries = summarize_chunks(chunks, model=model_name, chunk_limit=limit, client=client)
    return "\n---\n".join(summaries)

# --- Question Answering ---
def ask_question(question, context, model_name, system_prompt="Answer the question based on the provided context.", client=None):
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
        st.error(f"❌ Error during question answering with model `{model_name}`: {e}")
        return "⚠️ Unable to generate answer due to token limit or API error."

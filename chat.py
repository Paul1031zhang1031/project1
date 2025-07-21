import time
import streamlit as st
import tiktoken
from groq import Groq
import fitz  # PyMuPDF
import re


# --- Initialize Groq API Client ---

def summarize_section(client, section_title: str, section_text: str, model_name: str):
    """Summarizes a single section of the document."""
    if not client:
        return "API Client is not initialized."

    if len(section_text) > 15000:
        section_text = section_text[:15000]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that concisely summarizes the provided text from a document section."},
                {"role": "user", "content": f"Please summarize the section titled '{section_title}':\n\n{section_text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during summarization: {e}"

def answer_question(client,question: str, document_chunks: list, model_name: str):
    """Finds relevant chunks using simple keyword search and uses them to answer a question."""
    if not client:
        return "API Client is not initialized."

    question_words = set(question.lower().split())
    relevant_chunks = []

    for chunk in document_chunks:
        if any(word in chunk['text'].lower() for word in question_words):
            relevant_chunks.append(
                f"--- From section '{chunk['title']}' (p.{chunk['start_page']}) ---\n{chunk['text']}"
            )

    if not relevant_chunks:
        return "I could not find any relevant information in the document to answer that question."

    context = "\n\n".join(relevant_chunks)

    if len(context) > 15000:
        context = context[:15000]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful Q&A assistant. Answer the user's question based ONLY on the provided context from the document. If the answer is not in the context, say so."},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while answering the question: {e}"

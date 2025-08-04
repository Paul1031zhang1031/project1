import streamlit as st
import fitz, re, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from summarizer_engine import get_chapter_text

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def create_document_index(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        st.error(f"QA Engine Error: Failed to read PDF '{pdf_path}': {e}")
        return None, None
    chunks = re.split(r'\n\s*\n', full_text)
    text_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
    if not text_chunks: return None, None
    model = get_embedding_model()
    chunk_embeddings = model.encode(text_chunks)
    return text_chunks, chunk_embeddings

def find_context_in_relevant_chapter(question: str, summary_data: list):
    """
    Finds the most relevant chapter using an "augmented search" (title + content snippet),
    then uses a hybrid scoring model to find the most precise sub-sections for the answer.
    """
    model = get_embedding_model()
    
    # --- MODIFICATION: Create an "augmented" text for each chapter for better search ---
    # This combines the title with the first 500 characters of the chapter text.
    chapter_titles = [item['title'] for item in summary_data]
    searchable_chapter_texts = [f"{item['title']}\n\n{item['text'][:500]}" for item in summary_data]
    
    question_embedding = model.encode([question])
    # Generate embeddings from the new, richer text
    title_embeddings = model.encode(searchable_chapter_texts)
    
    # Find the best matching chapter based on the augmented text
    similarities = cosine_similarity(question_embedding, title_embeddings)[0]
    best_chapter_index = np.argmax(similarities)
    best_chapter_title = chapter_titles[best_chapter_index] # Get the original clean title

    chapter_text = get_chapter_text(summary_data, best_chapter_title)
    if not chapter_text:
        return None

    # Split chapter into sections based on ALL-CAPS headings
    parts = re.split(r'(?m)(^\s*[A-Z][A-Z\s.()&’]{4,99}\s*$)', chapter_text)
    
    structured_chunks = []
    i = 1 if not parts[0].strip() else 0
    while i < len(parts) -1:
        heading = parts[i].strip()
        content = parts[i+1].strip()
        if heading and content:
            structured_chunks.append({
                'heading': heading,
                'content': content,
                'search_text': f"{heading}\n{content}"
            })
        i += 2

    if len(structured_chunks) > 1:
        headings = [chunk['heading'] for chunk in structured_chunks]
        search_texts = [chunk['search_text'] for chunk in structured_chunks]

        heading_embeddings = model.encode(headings)
        search_text_embeddings = model.encode(search_texts)

        heading_similarities = cosine_similarity(question_embedding, heading_embeddings)[0]
        search_text_similarities = cosine_similarity(question_embedding, search_text_embeddings)[0]

        combined_scores = (0.7 * heading_similarities) + (0.3 * search_text_similarities)

        top_k_indices = np.argsort(combined_scores)[-2:][::-1]
        relevant_context = "\n\n---\n\n".join([search_texts[i] for i in top_k_indices])
    else:
        return chapter_text, best_chapter_title

    return relevant_context, best_chapter_title
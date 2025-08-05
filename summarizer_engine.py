import streamlit as st
import fitz
import json

@st.cache_data
def load_summary_data(pdf_path, toc_path):
    try:
        with open(toc_path, 'r') as f:
            toc_data = json.load(f)
    except Exception as e:
        st.error(f"Summary Engine Error: Could not load or parse '{toc_path}': {e}")
        return []
    
    offset = toc_data.get("page_offset", 0)
    toc_list = toc_data.get("chapters", [])
    if not toc_list: return []
    
    doc = fitz.open(pdf_path)
    chunks = []
    for i, item in enumerate(toc_list):
        title, start_page = item['title'], item['page'] - 1 + offset
        end_page = toc_list[i + 1]['page'] - 2 + offset if i + 1 < len(toc_list) else len(doc) - 1
        start_page = max(0, min(start_page, len(doc) - 1))
        end_page = max(start_page, min(end_page, len(doc) - 1))
        text = "".join(doc[p].get_text() for p in range(start_page, end_page + 1))
        chunks.append({"title": title, "text": text.strip()})
    doc.close()
    return chunks

def get_chapter_text(chapters_data: list, title: str) -> str:
    """A simple helper to find the text for a given chapter title."""
    return next((item['text'] for item in chapters_data if item['title'] == title), "")
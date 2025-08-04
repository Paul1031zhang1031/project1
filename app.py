import streamlit as st
from groq import Groq
from PIL import Image
import os
import pandas as pd
import base64
from io import BytesIO
from pathlib import Path
import shutil

from qa import create_document_index, find_context_in_relevant_chapter
from summarizer_engine import load_summary_data, get_chapter_text
from chat import get_summary, get_qa_answer
from eval import run_consensus_evaluation

# ==============================================================================
# DATA LOADING & INITIALIZATION
# ==============================================================================
PDF_PATH = "./data/BU.pdf"
TOC_PATH = "./data/toc.json"
IMAGE_PATH = "./images/bishop_logo.png"
#MODELS_TO_EVALUATE = ["gemma2-9b-it", "llama3-8b-8192","llama3-70b-8192"]
QA_MODELS_TO_EVALUATE = ["gemma2-9b-it", "llama3-8b-8192", "llama3-70b-8192"]
# Based on Phase 1 results, llama3-8b is the best for summarization. We will use it exclusively.
SUMMARY_MODEL = "llama3-8b-8192" 

# We still need the original chunks for fallback, but SUMMARY_DATA is now key for Q&A
SUMMARY_DATA = load_summary_data(PDF_PATH, TOC_PATH)
TEXT_CHUNKS, CHUNK_EMBEDDINGS = create_document_index(PDF_PATH)

try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e: 
    st.error(f"Groq API client error: {e}"); st.stop()


log_dir = Path("logs")
if log_dir.exists():
    shutil.rmtree(log_dir)
log_dir.mkdir()

# ==============================================================================
# PAGE CONFIGURATION AND HEADER
# ==============================================================================
st.set_page_config(page_title="Document Analysis Hub", layout="wide", page_icon="📚")

try:
    image = Image.open(IMAGE_PATH)
    resized_image = image.resize((300, 100))  # Resize to 300x100 pixels

    buffered = BytesIO()
    resized_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align:center">
            <img src="data:image/png;base64,{img_str}" alt="Resized Image" />
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.warning(f"Header image not found at '{IMAGE_PATH}'.")


st.divider()
# ==============================================================================
# MAIN APPLICATION INTERFACE (Simplified - No Details Button)
# ==============================================================================

if 'qa_best_answer' not in st.session_state: st.session_state.qa_best_answer = ""
if 'summary_best_summary' not in st.session_state: st.session_state.summary_best_summary = ""
# Add session state to store the source chapter
if 'qa_source_chapter' not in st.session_state: st.session_state.qa_source_chapter = ""

if not SUMMARY_DATA or TEXT_CHUNKS is None:
    st.error("Data could not be loaded. Please check your source files.")
else:
    col1, col2 = st.columns(2, gap="large")

    # --- Left Column: Full-Document Q&A ---
    with col1:
        with st.container(border=True):
            st.subheader("Any Question?")
            question = st.text_input("Enter your question here:", placeholder="e.g., How much are the tuition fees?")
            
            if st.button("Get Answer", use_container_width=True, key="qa_button"):
                if question:
                    with st.spinner("Finding relevant chapter and generating answer..."):
                        # --- USE THE NEW, SMARTER FUNCTION ---
                        context_and_source = find_context_in_relevant_chapter(
                            question, SUMMARY_DATA
                        )
                        
                        if context_and_source:
                            relevant_context, source_chapter = context_and_source
                            st.session_state.qa_source_chapter = f"Source: Based on the '{source_chapter}' chapter."

                            qa_report = run_consensus_evaluation(
                                client=groq_client,
                                models=QA_MODELS_TO_EVALUATE,
                                task_type='qa',
                                context=relevant_context,
                                prompt=question
                            )
                            st.session_state.qa_best_answer = qa_report["best_result"]
                        else:
                            st.session_state.qa_source_chapter = ""
                            st.session_state.qa_best_answer = "Sorry, I could not find a relevant chapter in the document to answer your question."
                else:
                    st.warning("Please enter a question.")
            
            # Display the source chapter for transparency
            if st.session_state.qa_source_chapter:
                st.info(st.session_state.qa_source_chapter)

            st.text_area("Best Answer", value=st.session_state.qa_best_answer, height=500, disabled=True)

    # --- Right Column: Chapter-Based Summarizer ---
    with col2:
        with st.container(border=True):
            st.subheader("Summarize this, please")
            theme_titles = [item['title'] for item in SUMMARY_DATA]
            selected_theme_title = st.selectbox("Choose a subject to summarize:", theme_titles)
            selected_theme_text = get_chapter_text(SUMMARY_DATA, selected_theme_title)

            if st.button("Generate Summary", use_container_width=True, key="summary_button"):
                with st.spinner(f"Generating summary with {SUMMARY_MODEL}..."):
                    summary_report = run_consensus_evaluation(
                        client=groq_client,
                        models=[SUMMARY_MODEL],
                        task_type='summary',
                        context=selected_theme_text,
                        prompt=selected_theme_title
                    )
                    st.session_state.summary_best_summary = summary_report["best_result"]
            
            st.text_area("Best Summary", value=st.session_state.summary_best_summary, height=500, disabled=True)
        
st.markdown(
    """
    <div style="text-align: center; color: grey;">
        <small>Copyright © BU. Version 0.1. Last update August 2025</small>
    </div>
    """,
    unsafe_allow_html=True
)
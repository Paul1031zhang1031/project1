import streamlit as st
import fitz  # PyMuPDF
from groq import Groq

# --- Import all necessary functions 
from chat import summarize_section as thematic_summarize
from chat import answer_question as thematic_answer
from summarizer import ext_text_from_pdf, sum_text, ask_question as eval_ask_question
from e_rouge import compute_and_plot_rouge
from e_similarity import create_similarity_scores_against_reference
from e_qa import compare_model_answers  #  evaluate answers using Ninjia Api

# --- Page Configuration and API Client Initialization ---
st.set_page_config(page_title="Document Analysis Hub", layout="wide")
st.title("📄 Document Analysis Hub")

# Initialize API clients ONCE using secrets
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    api_ninja_key = st.secrets["API_NINJA_KEY"]
except Exception as e:
    st.error(f"Could not initialize API clients. Please check your .streamlit/secrets.toml file. Error: {e}")
    st.stop()

# --- Data Processing Logic (for Thematic Q&A) ---
def parse_toc_text(toc_text: str):
    parsed_toc = []
    lines = toc_text.strip().split('\n')
    for line in lines:
        words = line.strip().split()
        if not words: continue
        try:
            page_num = int(words[-1])
            title = ' '.join(words[:-1]).strip(" .")
            if title: parsed_toc.append({'title': title, 'page': page_num})
        except (ValueError, IndexError): pass
    return parsed_toc

def preprocess_pdf_by_toc(pdf_file, toc_text: str, offset: int):
    toc_list = parse_toc_text(toc_text)
    if not toc_list: raise ValueError("ToC parsing failed.")
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    chunks = []
    for i, item in enumerate(toc_list):
        title, start_page = item['title'], item['page'] - 1 + offset
        end_page = toc_list[i + 1]['page'] - 2 + offset if i + 1 < len(toc_list) else len(doc) - 1
        start_page = max(0, min(start_page, len(doc) - 1))
        end_page = max(start_page, min(end_page, len(doc) - 1))
        text = "".join(doc[p].get_text() for p in range(start_page, end_page + 1))
        chunks.append({"title": title, "start_page": start_page + 1, "end_page": end_page + 1, "text": text.strip()})
    doc.close()
    return chunks

# --- Session State Initialization (for both workflows) ---
if 'toc_processed_data' not in st.session_state: st.session_state.toc_processed_data = None
if 'toc_chat_history' not in st.session_state: st.session_state.toc_chat_history = []
if 'eval_pdf_text' not in st.session_state: st.session_state.eval_pdf_text = None
if 'eval_uploaded_file' not in st.session_state: st.session_state.eval_uploaded_file = None
if 'eval_single_summary' not in st.session_state: st.session_state.eval_single_summary = None
if 'eval_single_answer' not in st.session_state: st.session_state.eval_single_answer = None
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None

# --- Radio Button "Tab" Selector ---
st.radio(
    "Select Workflow:",
    ["Thematic Document Q&A", "Model Evaluation & Short Doc Analysis"],
    key="active_tab",
    horizontal=True,
)
st.markdown("---")

# --- Conditional Sidebar for Thematic Q&A ---
if st.session_state.active_tab == "Thematic Document Q&A":
    with st.sidebar:
        st.header("Thematic Q&A Controls")
        pdf_file_toc = st.file_uploader("Upload PDF", type="pdf", key="toc_pdf")
        toc_text = st.text_area("Paste Table of Contents", height=150, key="toc_text", placeholder="Introduction . . . 1")
        toc_offset = st.number_input("ToC Page Number Offset", min_value=-10, max_value=10, value=1)
        if st.button("Process Document", key="toc_process"):
            if pdf_file_toc and toc_text:
                with st.spinner("Processing document..."):
                    try:
                        st.session_state.toc_processed_data = preprocess_pdf_by_toc(pdf_file_toc, toc_text, toc_offset)
                        st.session_state.toc_chat_history = []
                        st.success(f"Document processed into {len(st.session_state.toc_processed_data)} sections.")
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
            else:
                st.warning("Please upload a PDF and provide the ToC.")

# ==============================================================================
# MAIN CONTENT AREA
# ==============================================================================

# --- CONTENT 1: Thematic Document Q&A (CORRECTED) ---
if st.session_state.active_tab == "Thematic Document Q&A":
    st.header("Interact with a Large Document via its Table of Contents")
    
    if st.session_state.toc_processed_data:
        processed_chunks = st.session_state.toc_processed_data
        st.success(f" Successfully processed document with {len(processed_chunks)} sections. Ready to interact.")
        
        model_name = st.selectbox("Choose a Model", ("gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192"), key="toc_model_selector")
        
        st.divider()

        # --- Section Summarizer ---
        st.subheader("1. Summarize a Section")
        section_titles = [chunk["title"] for chunk in processed_chunks]
        selected_title = st.selectbox("Choose a section to summarize:", section_titles)

        if st.button("Summarize Section", key="summarize_btn"):
            with st.spinner(f"Summarizing '{selected_title}'..."):
                selected_chunk = next(chunk for chunk in processed_chunks if chunk["title"] == selected_title)
                summary = thematic_summarize(groq_client, selected_title, selected_chunk["text"], model_name)
                st.info(f"Summary for '{selected_title}':")
                st.markdown(summary)

        st.divider()

        # --- Q&A Chatbot ---
        st.subheader("2. Ask a Question About the Document")
        
        for role, message in st.session_state.toc_chat_history:
            with st.chat_message(role):
                st.markdown(message)
                
        if question := st.chat_input("Ask a question about the document...", key="toc_chat_input"):
            st.session_state.toc_chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
                
            with st.chat_message("assistant"):
                with st.spinner("Finding an answer..."):
                    answer = thematic_answer(groq_client, question, processed_chunks, model_name)
                    st.markdown(answer)
            
            st.session_state.toc_chat_history.append(("assistant", answer))

    else:
        st.info("Upload a PDF and its Table of Contents in the sidebar to begin.")


# --- CONTENT 2: Model Evaluation & Short Doc Analysis (UNCHANGED and CORRECT) ---
elif st.session_state.active_tab == "Model Evaluation & Short Doc Analysis":
    st.header("Analyze & Evaluate Models on a Short Document")
    pdf_file_eval = st.file_uploader("Upload a PDF to Analyze", type="pdf", key="eval_pdf")

    if pdf_file_eval is not None:
        if st.session_state.eval_uploaded_file != pdf_file_eval.name:
            with st.spinner("Extracting text..."):
                st.session_state.eval_pdf_text = ext_text_from_pdf(pdf_file_eval)
                st.session_state.eval_uploaded_file = pdf_file_eval.name
                st.session_state.eval_single_summary = None
                st.session_state.eval_single_answer = None
                st.session_state.evaluation_results = None
            st.success("PDF text extracted.")

    if st.session_state.eval_pdf_text:
        eval_models = ["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        
        st.subheader("Quick Summarization")
        summarize_model = st.selectbox("Select a model for summarization:", eval_models, key="single_sum_model")
        if st.button("Summarize text"):
            with st.spinner("Summarizing..."):
                st.session_state.eval_single_summary = sum_text(groq_client, st.session_state.eval_pdf_text, summarize_model)
        if st.session_state.eval_single_summary:
            st.info(st.session_state.eval_single_summary)

        st.header("Ask & Compare Model Answers")
        question = st.text_input("Enter a question to ask all models:", key="compare_qa_input")

        if st.button("Compare Answers", disabled=not question):
            with st.spinner("Generating answers and building comparison report..."):
                # The function call now returns a single dictionary
                report = compare_model_answers(
                    groq_client,
                    st.session_state.eval_pdf_text,
                    question,
                    eval_models,
                    api_ninja_key
                )
                st.session_state.compare_qa_results = report
            st.success("Comparison complete!")
        
        # enhanded the qa results display with evaluation 
        if st.session_state.get('compare_qa_results'):
            results = st.session_state.compare_qa_results
            
            st.subheader(" Consensus Answer")
            st.markdown(f"The answer with the highest agreement among all models was from **{results['best_model']}**.")
            st.info(results['best_answer'])
            
            st.subheader(" Model Consensus Scores")
            st.markdown("This table shows the average similarity of each model's answer compared to all other answers.")
            st.dataframe(
                results['avg_scores_df'].style.format({'Average Similarity Score': '{:.2%}'}),
                use_container_width=True
            )
            st.pyplot(results['barchart_fig'])

            st.subheader("Detailed Comparison")
            with st.expander("Show Pairwise Similarity Heatmap & All Answers"):
                st.markdown("**Pairwise Similarity Heatmap**")
                st.pyplot(results['heatmap_fig'])
                st.markdown("**All Generated Answers**")
                for model, answer in results["answers"].items():
                    st.markdown(f"---")
                    st.markdown(f"**Answer from `{model}`:**")
                    st.write(answer)

        st.divider()

        st.header("Comprehensive Model Evaluation")
        
        reference_file = st.file_uploader("Upload Reference Summary (.txt)", type=["txt"])
        if reference_file is not None:
            reference_summary = reference_file.read().decode("utf-8")
            st.text_area("Reference Summary (from file):", value=reference_summary, height=150, disabled=True)
        else:
            reference_summary = st.text_area("Or enter Reference Summary manually:", height=150)
        
        has_reference = bool(reference_summary.strip())
        
        if st.button("Generate All Summaries and Evaluate", disabled=not has_reference):
            with st.spinner("Generating all summaries and calculating scores..."):
                summaries = {model: sum_text(groq_client, st.session_state.eval_pdf_text, model) for model in eval_models}
                similarity_df = create_similarity_scores_against_reference(summaries, reference_summary, api_ninja_key)
                rouge_df, rouge_fig = compute_and_plot_rouge(summaries, reference_summary)
                
                st.session_state.evaluation_results = {
                    "summaries": summaries,
                    "similarity_df": similarity_df,
                    "rouge_df": rouge_df,
                    "rouge_fig": rouge_fig
                }
            st.success("Evaluation complete!")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            st.subheader("Evaluation Results")
            st.markdown("**ROUGE Scores**")
            st.dataframe(results["rouge_df"])
            st.pyplot(results["rouge_fig"])
            
            st.markdown("**Similarity Scores**")
            st.dataframe(results["similarity_df"])

            st.markdown("**Generated Summaries**")
            for model_name, summary in results["summaries"].items():
                with st.expander(f"Summary from {model_name}"):
                    st.write(summary)
    else:
        st.info("Upload a PDF to begin analysis and evaluation.")
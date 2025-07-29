# app_v2.py
import streamlit as st
import fitz
from groq import Groq
import pandas as pd
import itertools

# Import all our functions
from chat import summarize_text_map_reduce, answer_question_within_section
from summarizer import ext_text_from_pdf
from e_rouge import compute_and_plot_rouge
from e_similarity import get_similarity_score
from e_qa import compare_model_answers
from e_graph import create_consensus_graph, create_reference_graph

# --- Page Configuration and API Client Initialization ---
st.set_page_config(page_title="Document Analysis Hub", layout="wide")
st.title("📄 Document Analysis Hub")

try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    api_ninja_key = st.secrets["API_NINJA_KEY"]
except Exception as e:
    st.error(f"Could not initialize API clients. Check your secrets. Error: {e}")
    st.stop()

# --- Data Processing Logic (Stable and correct) ---
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

# --- Session State Initialization ---
if 'toc_processed_data' not in st.session_state: st.session_state.toc_processed_data = None
if 'toc_chat_history' not in st.session_state: st.session_state.toc_chat_history = []
if 'toc_summary_result' not in st.session_state: st.session_state.toc_summary_result = None
if 'active_section_title' not in st.session_state: st.session_state.active_section_title = None
if 'eval_pdf_text' not in st.session_state: st.session_state.eval_pdf_text = None
if 'eval_uploaded_file' not in st.session_state: st.session_state.eval_uploaded_file = None
if 'eval_single_summary' not in st.session_state: st.session_state.eval_single_summary = None
if 'compare_qa_results' not in st.session_state: st.session_state.compare_qa_results = None
if 'consensus_results' not in st.session_state: st.session_state.consensus_results = None
if 'reference_eval_results' not in st.session_state: st.session_state.reference_eval_results = None
if 'reference_text' not in st.session_state: st.session_state.reference_text = ""

# --- UI Tab Selector ---
st.radio(
    "Select Workflow:",
    ["Thematic Document Q&A", "Model Evaluation & Short Doc Analysis"],
    key="active_tab",
    horizontal=True
)
st.markdown("---")

# --- Sidebar ---
if st.session_state.active_tab == "Thematic Document Q&A":
    with st.sidebar:
        st.header("Thematic Q&A Controls")
        pdf_file_toc = st.file_uploader("Upload PDF", type="pdf", key="toc_pdf")
        toc_text = st.text_area("Paste Table of Contents", height=150, key="toc_text", placeholder="Introduction . . . 1")
        toc_offset = st.number_input("ToC Page Number Offset", min_value=-20, max_value=20, value=0)
        if st.button("Process Document", key="toc_process"):
            if pdf_file_toc and toc_text:
                with st.spinner("Processing document..."):
                    try:
                        st.session_state.toc_processed_data = preprocess_pdf_by_toc(pdf_file_toc, toc_text, toc_offset)
                        st.session_state.toc_chat_history = []
                        st.session_state.toc_summary_result = None
                        if st.session_state.toc_processed_data:
                            st.session_state.active_section_title = st.session_state.toc_processed_data[0]['title']
                        st.success(f"Document processed into {len(st.session_state.toc_processed_data)} sections.")
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
            else:
                st.warning("Please upload a PDF and provide the ToC.")
elif st.session_state.active_tab == "Model Evaluation & Short Doc Analysis":
    with st.sidebar:
        st.header("Evaluation Controls")
        st.info("Upload a short PDF (1-10 pages) in the main panel to begin the evaluation workflow.")

# ==============================================================================
# MAIN CONTENT AREA
# ==============================================================================

if st.session_state.active_tab == "Thematic Document Q&A":
    st.header("Interact with a Large Document via its Table of Contents")
    if st.session_state.toc_processed_data:
        processed_chunks = st.session_state.toc_processed_data
        section_titles = [chunk["title"] for chunk in processed_chunks]
        st.subheader("Select Active Document Section")
        if not st.session_state.active_section_title:
            st.session_state.active_section_title = section_titles[0]
        try:
            active_section_index = section_titles.index(st.session_state.active_section_title)
        except ValueError:
            active_section_index = 0
        newly_selected_title = st.selectbox("Active Section:", section_titles, index=active_section_index, label_visibility="collapsed")
        if newly_selected_title != st.session_state.active_section_title:
            st.session_state.active_section_title = newly_selected_title
            st.session_state.toc_summary_result = None
            st.session_state.toc_chat_history = []
            st.rerun()
        model_name = st.selectbox("Choose a Model", ("gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192"), key="toc_model_sel")
        st.divider()
        st.subheader(f"1. Summarize Section: '{st.session_state.active_section_title}'")
        if st.button("Summarize this Section", key="summarize_btn"):
            with st.spinner(f"Summarizing '{st.session_state.active_section_title}'..."):
                chunk_to_summarize = next(chunk for chunk in processed_chunks if chunk["title"] == st.session_state.active_section_title)
                summary_dict = summarize_text_map_reduce(groq_client, chunk_to_summarize["text"], model_name, f"the section '{st.session_state.active_section_title}'")
                st.session_state.toc_summary_result = {"title": st.session_state.active_section_title, "final": summary_dict["final_summary"]}
        if st.session_state.toc_summary_result and st.session_state.toc_summary_result['title'] == st.session_state.active_section_title:
            res = st.session_state.toc_summary_result
            st.info(f"Summary for '{res['title']}':")
            st.markdown(res['final'])
        st.divider()
        st.subheader(f"2. Ask a Question About: '{st.session_state.active_section_title}'")
        for role, message in st.session_state.toc_chat_history:
            with st.chat_message(role):
                st.markdown(message)
        if question := st.chat_input(f"Ask about '{st.session_state.active_section_title}'..."):
            st.session_state.toc_chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner(f"Finding an answer in '{st.session_state.active_section_title}'..."):
                    chunk_to_ask = next(chunk for chunk in processed_chunks if chunk["title"] == st.session_state.active_section_title)
                    answer = answer_question_within_section(groq_client, question, chunk_to_ask, model_name)
                    st.markdown(answer)
            st.session_state.toc_chat_history.append(("assistant", answer))

elif st.session_state.active_tab == "Model Evaluation & Short Doc Analysis":
    st.header("Analyze & Evaluate Models on a Short Document")
    pdf_file_eval = st.file_uploader("Upload a PDF to Analyze", type="pdf", key="eval_pdf")
    if pdf_file_eval:
        if st.session_state.get('eval_uploaded_file') != pdf_file_eval.name:
            with st.spinner("Extracting text..."):
                st.session_state.eval_pdf_text = ext_text_from_pdf(pdf_file_eval)
                st.session_state.eval_uploaded_file = pdf_file_eval.name
                st.session_state.eval_single_summary = None
                st.session_state.compare_qa_results = None
                st.session_state.consensus_results = None
                st.session_state.reference_eval_results = None
                st.session_state.reference_text = ""
            st.success("PDF text extracted.")
    if st.session_state.get('eval_pdf_text'):
        eval_models = ["gemma2-9b-it", "llama3-8b-8192", "llama3-70b-8192"]
        st.subheader("Quick Summarization")
        summarize_model = st.selectbox("Select model:", eval_models, key="single_sum_model")
        if st.button("Summarize Full Text"):
            with st.spinner("Summarizing..."):
                result_dict = summarize_text_map_reduce(groq_client, st.session_state.eval_pdf_text, summarize_model, "the entire document")
                st.session_state.eval_single_summary = result_dict["final_summary"]
        if st.session_state.eval_single_summary:
            st.info(st.session_state.eval_single_summary)
        st.divider()
        st.header("Ask & Compare Model Answers")
        question = st.text_input("Enter a question:", key="compare_qa_input")
        if st.button("Compare Answers", disabled=not question):
            with st.spinner("Generating answers and comparison report..."):
                report = compare_model_answers(groq_client, st.session_state.eval_pdf_text, question, eval_models, api_ninja_key)
                st.session_state.compare_qa_results = report
            st.success("Comparison complete!")
        if st.session_state.get('compare_qa_results'):
            results = st.session_state.compare_qa_results
            st.subheader("Consensus Answer")
            st.markdown(f"The answer with the highest agreement among all models was from **{results['best_model']}**.")
            st.info(results['best_answer'])
            st.subheader("Model Consensus Scores")
            st.dataframe(results['avg_scores_df'].style.format({'Average Similarity Score': '{:.2%}'}), use_container_width=True)
            st.pyplot(results['barchart_fig'])
        st.divider()
        st.header("Comprehensive Summary Evaluation")
        eval_method = st.radio("Choose your evaluation method:", ("Reference-Free Consensus", "Compare to Golden Reference"), key="eval_method_selector", horizontal=True)
        if eval_method == "Reference-Free Consensus":
            st.info("This tool compares all models against each other to find the 'best' summary without needing a pre-written reference.")
            if st.button("Generate & Compare All Summaries (Consensus)"):
                with st.spinner("Generating summaries and calculating consensus..."):
                    summaries = {}
                    for model in eval_models:
                        result_dict = summarize_text_map_reduce(
                            groq_client, st.session_state.eval_pdf_text, model, f"summary from `{model}`"
                        )
                        summaries[model] = result_dict["final_summary"]
                    scores = {(m1, m2): get_similarity_score(summaries[m1], summaries[m2], api_ninja_key) for m1, m2 in itertools.combinations(eval_models, 2)}
                    consensus_scores = {model: sum(s for p, s in scores.items() if model in p) / (len(eval_models) - 1) for model in eval_models}
                    best_model_name = max(consensus_scores, key=consensus_scores.get)
                    best_summary = summaries[best_model_name]
                    graph_fig = create_consensus_graph(summaries, api_ninja_key, best_model_name)
                    consensus_df = pd.DataFrame(list(consensus_scores.items()), columns=['Model', 'Avg. Consensus Score']).sort_values(by='Avg. Consensus Score', ascending=False)
                    st.session_state.consensus_results = {"summaries": summaries, "best_model": best_model_name, "best_summary": best_summary, "graph_fig": graph_fig, "consensus_df": consensus_df}
                st.success("Consensus analysis complete!")
            if st.session_state.consensus_results:
                res = st.session_state.consensus_results
                st.subheader("🏆 Best Consensus Summary")
                st.markdown(f"The summary from **`{res['best_model']}`** had the highest agreement with all other models.")
                st.info(res['best_summary'])
                st.subheader("📊 Consensus Scores & Graph")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(res['consensus_df'].style.format({'Avg. Consensus Score': '{:.2%}'}), use_container_width=True)
                with col2:
                    st.pyplot(res["graph_fig"])
                with st.expander("View all generated summaries"):
                    for model, summary in res['summaries'].items():
                        st.markdown(f"**`{model}`:** {summary}")

        elif eval_method == "Compare to Golden Reference":
            st.info("Create or upload a 'perfect' reference summary, then evaluate all models against it using only ROUGE scores.")
            st.subheader("Step 1: Create Your Reference Summary")
            uploaded_ref = st.file_uploader("Upload a Reference (.txt)", type="txt", key="ref_upload")
            if uploaded_ref:
                st.session_state.reference_text = uploaded_ref.read().decode("utf-8")
            st.session_state.reference_text = st.text_area("Paste or edit your reference summary here:", value=st.session_state.reference_text, height=250, key="ref_text_area")
            st.subheader("Step 2: Evaluate Models Against Your Reference")
            if st.button("Evaluate with ROUGE", disabled=not st.session_state.reference_text.strip()):
                with st.spinner("Generating summaries and calculating ROUGE scores..."):
                    ref_sum = st.session_state.reference_text
                    summaries = {}
                    for model in eval_models:
                        result_dict = summarize_text_map_reduce(
                            groq_client, st.session_state.eval_pdf_text, model, f"summary from `{model}`"
                        )
                        summaries[model] = result_dict["final_summary"]
                    rouge_df, rouge_fig = compute_and_plot_rouge(summaries, ref_sum)
                    best_rouge_model = rouge_df['rougeL'].idxmax()
                    st.session_state.reference_eval_results = {
                        "summaries": summaries,
                        "rouge_df": rouge_df,
                        "rouge_fig": rouge_fig,
                        "best_rouge_model": best_rouge_model,
                    }
                st.success("ROUGE evaluation complete!")
            if st.session_state.reference_eval_results:
                res = st.session_state.reference_eval_results
                st.subheader("🏆 Best Model (by ROUGE-L Score)")
                st.markdown(f"Based on the ROUGE-L f-measure score, the best model is **`{res['best_rouge_model']}`**.")
                st.subheader("📊 ROUGE Scores")
                st.dataframe(res["rouge_df"].style.format({'rouge1': '{:.4f}', 'rouge2': '{:.4f}', 'rougeL': '{:.4f}'}))
                st.pyplot(res["rouge_fig"])
                # --- THIS IS THE FINAL, MISSING PART ---
                with st.expander("View all generated summaries"):
                    for model, summary in res['summaries'].items():
                        st.markdown(f"**`{model}`:** {summary}")
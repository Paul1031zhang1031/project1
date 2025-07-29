import fitz
import streamlit as st
import time
from chat import num_tokens_from_string

# --- PDF Extraction ---
# This function is used by app.py
def ext_text_from_pdf(file):
    """Extracts all text from a given PDF file."""
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

# --- Question Answering ---
# This function is used by e_qa.py
def ask_question(client, pdf_text: str, question: str, model_name: str):
    """Asks a question to a model based on a given context."""
    if not client:
        st.error(f"API Client not provided to ask_question for model `{model_name}`.")
        return "⚠️ Unable to generate answer due to an internal configuration error."

    prompt = f"Context:\n{pdf_text}\n\nQuestion: {question}"
    token_count = num_tokens_from_string(prompt, model_name)
    st.info(f"🔢 Tokens in request: {token_count}")

    try:
        time.sleep(2) # To respect potential rate limits
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Answer the question based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Error during question answering with model `{model_name}`: {e}")
        return "⚠️ Unable to generate answer due to token limit or API error."

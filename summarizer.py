import fitz
import os
from groq import Groq
from PIL import Image


os.environ["GROQ_API_KEY"] = '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'
#client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ext_text_from_pdf(file):
    # file can be path or file-like object
    # If it's an UploadedFile, convert to bytes buffer
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

def sum_text(client,text,model_name):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            model=model_name,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "The text could not be processed."

def ask_question(client,context, question,model_name):
    try:
        answer_response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"context: {context} Question: {question}"}
            ],
            model=model_name,
        )
        return answer_response.choices[0].message.content
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "An error occurred."



├── app.py              # Streamlit app (combined)
├── chat.py             # deal with inner part of big file content 
├── summarizer.py       # for evaluation-based summarization and Q&A.
├── e_rouge.py          # summary evaluation by hugging face evaluation 
├── e_similarity.py     # summary evaluation by similarity
├── e_qa.py             # qa evaluation by similarity
└── .streamlit/
    └── secrets.toml    # To store API keys securely.

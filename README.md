# AI-Powered Text Analysis Toolkit

This project is a multi-functional Streamlit application designed for advanced text analysis. It provides tools for text summarization, interactive chat, and a suite of evaluation metrics to measure model performance, including ROUGE, QA accuracy, and semantic similarity.

## Features

- **Interactive Web UI:** Built with Streamlit for a user-friendly experience.  
- **Text Summarization:** Generate concise summaries of long documents.  
- **Chat Interface:** A module for implementing conversational AI.  
- **Comprehensive Evaluation Suite:**  
  - **ROUGE Scores** (`e_rouge.py`): Evaluate summary quality against a reference.  
  - **QA Evaluation** (`e_qa.py`): Assess Question-Answering model performance.  
  - **Similarity Scores** (`e_similarity.py`): Measure semantic similarity between texts without references.

## Project Structure
├── app.py              # Streamlit app (combined)
├── chat.py             # deal with inner part of big file content 
├── summarizer.py       # for evaluation-based summarization and Q&A.
├── e_rouge.py          # summary evaluation by hugging face evaluation 
├── e_similarity.py     # summary evaluation by similarity
├── e_qa.py             # qa evaluation by similarity
├── requirements.txt
├── README.md       
├── .gitignore       
└── LICENSE          
├── data for test/
└── .streamlit/
    └── secrets.toml    # To store API keys securely.

# AI-Powered Text Analysis Toolkit

This project is a multi-functional Streamlit application designed for advanced text analysis. It provides tools for text summarization, interactive chat, and a suite of evaluation metrics to measure model performance, including ROUGE, QA accuracy, and semantic similarity.

## Features

-   **Interactive Web UI**: Built with Streamlit for a user-friendly experience.
-   **Text Summarization**: Generate concise summaries of long documents.
-   **Chat Interface**: A module for implementing conversational AI.
-   **Comprehensive Evaluation Suite**:
    -   **ROUGE Scores** (`e_rouge.py`): Evaluate summary quality against a reference.
    -   **QA Evaluation** (`e_qa.py`): Assess the performance of Question-Answering models.
    -   **Similarity Scores** (`e_similarity.py`): Measure the semantic similarity between texts without reference.

## Project Structure

A brief overview of the key files in this project:

-   `app.py`: The main entry point to launch the Streamlit web application.
-   `summarizer.py`: Contains the core logic for the text summarization feature.
-   `chat.py`: Implements the interactive chat functionality.
-   `e_*.py`: A collection of scripts for evaluating model outputs.
-   `data for test/`: Directory containing sample data for testing the application's features.
-   `requirements.txt`: A list of all necessary Python dependencies.

## Setup and Installation
Follow these steps to get the project running on your local machine.
1. Clone the Repository
git clone https://github.com/Paul1031zhang1031/project1/
cd project1
2.(Optional) Create and activate a virtual environment:
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
# On Windows
python -m venv venv
.\venv\Scripts\activate
3.Install dependencies:
pip install -r requirements.txt
4.Run the Streamlit app:streamlit run app.py #Your web browser should open automatically with the running app.

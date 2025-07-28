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
<img width="2507" height="533" alt="1753667217063" src="https://github.com/user-attachments/assets/6b1a2cc2-4a9d-4d4e-ad78-63e0059f760f" />

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

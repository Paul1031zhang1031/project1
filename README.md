This project is a powerful, interactive Streamlit application designed for both deep analysis of large documents and comprehensive evaluation of Large Language Models (LLMs). Powered by the high-speed Groq API, it offers a dual-workflow interface to tackle complex text-based tasks.
The Thematic Q&A workflow allows users to parse large PDFs via their Table of Contents for focused summarization and question-answering. The Model Evaluation workflow provides a sophisticated suite of tools to compare LLM performance, featuring a unique, flexible system that supports both reference-free consensus analysis and traditional reference-based evaluation.
Key Features
Dual-Workflow Interface: A clean UI that separates the application into two distinct modes: large document interaction and model evaluation.
Intelligent Large Document Handling:
Parses PDFs based on a user-provided Table of Contents to create logical, chapter-based sections.
Automatically uses a Map-Reduce strategy to summarize sections that exceed the LLM's context window, preventing errors and ensuring complete analysis.
Connected Analysis Tools: In the Thematic Q&A workflow, the summarizer and Q&A chat are seamlessly linked to a single, user-selected document section for an intuitive experience.
Flexible Evaluation Suite: The user can choose the best evaluation method for their needs:
Reference-Free Consensus: Generates summaries from all models, calculates pairwise similarities, and identifies the "best" summary based on the highest average agreement with all other models.
Compare to Golden Reference: Allows the user to upload or write a "perfect" reference summary and evaluate models against it using industry-standard ROUGE scores.
Advanced Visualizations:
Generates similarity graphs (digraphs) using NetworkX to visually represent the "closeness" of model outputs, highlighting clusters and outliers.
Plots ROUGE scores for easy comparison.
Robust Q&A: For large document sections that exceed the context window, the Q&A tool uses an intelligent LLM-powered distillation process. It first prompts the model to extract only the most relevant information related to the question, then uses that focused context to generate a final, accurate answer.

Project Structure
The project is organized into modular Python scripts, each with a specific responsibility.
project1/
- .streamlit/             # Configuration folder for Streamlit.
-  - secrets.toml         # Securely stores API keys for Groq and API Ninjas.
- app.py                  # The main Streamlit application file; handles UI and state.
- chat.py                 # Core logic engine for summarization and Q&A.
- summarizer.py           # Utilities for PDF text extraction and helper functions.
- e_qa.py                 # Module for comparing Q&A answers.
- e_rouge.py              # Module for calculating and plotting ROUGE scores.
- e_similarity.py         # Utility for getting semantic similarity scores.
- e_graph.py              # Generates the NetworkX similarity and consensus graphs.
- README.md               # This documentation file.
- requirements.txt        # A list of all necessary Python dependencies.
  

Setup and Installation
Follow these steps to get the project running on your local machine.
1. Clone the Repository
Generated bash
git clone https://github.com/Paul1031zhang1031/project1/
cd project1

2. (Recommended and optional) Create and Activate a Virtual Environment
Generated bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt


4. Add API Keys
This application requires API keys for Groq (for LLM inference) and API Ninjas (for similarity scores).
Create a folder named .streamlit in the root of your project directory.
Inside .streamlit, create a file named secrets.toml.
Add your keys to the secrets.toml file in the following format:
Generated toml
# .streamlit/secrets.toml

GROQ_API_KEY = "your_groq_api_key_here"
API_NINJA_KEY = "your_api_ninjas_key_here"

5. Run the Streamlit App
streamlit run app.py
Your web browser should open automatically with the running application.

How to Use
Thematic Document Q&A
Select the "Thematic Document Q&A" workflow.
In the sidebar, upload a large PDF and paste its Table of Contents.
Click "Process Document."
Use the main dropdown to select the section you want to analyze.
Click "Summarize this Section" to get a summary or use the chat box to ask specific questions about the selected section.
Model Evaluation & Short Doc Analysis
Select the "Model Evaluation" workflow.
Upload a shorter PDF (1-20 pages).
Use the "Quick Summarization" or "Ask & Compare Model Answers" tools for fast analysis.
For a deep dive, navigate to the "Comprehensive Summary Evaluation" section.
Choose "Reference-Free Consensus" to see how models compare to each other and find the best "consensus" answer.
Choose "Compare to Golden Reference" to provide your own perfect summary and score models against it using ROUGE.

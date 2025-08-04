# evaluation.py

import streamlit as st
import itertools
import pandas as pd
import requests
import time
from datetime import datetime
from pathlib import Path
import shutil

# Import the core AI functions from your chat module
from chat import get_summary, get_qa_answer

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_similarity_score(text1: str, text2: str) -> float:
    """Calculates semantic similarity using the API Ninjas service."""
    try:
        api_key = st.secrets["API_NINJA_KEY"]
    except Exception:
        st.error("API_NINJA_KEY not found. Cannot calculate similarity.")
        return 0.0
    api_url = 'https://api.api-ninjas.com/v1/textsimilarity'
    headers = {'X-Api-Key': api_key}
    body = {'text_1': text1[:4900], 'text_2': text2[:4900]}
    try:
        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        return response.json().get('similarity', 0.0)
    except requests.exceptions.RequestException:
        return 0.0

# ==============================================================================
# MAIN EVALUATION FUNCTION
# ==============================================================================

def run_consensus_evaluation(client, models, task_type, context, prompt):
    """
    Gets results from all models, performs consensus evaluation, saves a report,
    and returns a dictionary containing only the best result.
    """
    
    # 1. Get results from all models
    results = {}
    for model in models:
        if task_type == 'qa':
            result = get_qa_answer(client, prompt, context, model)
        elif task_type == 'summary':
            result = get_summary(client, context, model, f"the theme '{prompt}'")
        results[model] = result
        time.sleep(10) # Delay to respect API rate limits
 
    # 2. Perform pairwise consensus evaluation
    scores = {}
    valid_results = {m: r for m, r in results.items() if isinstance(r, str) and not r.startswith("An error")}
    
    if len(valid_results) < 2:
        return {"best_result": next(iter(valid_results.values()), "Could not generate a valid answer from any model.")}

    for m1, m2 in itertools.combinations(valid_results.keys(), 2):
        score = get_similarity_score(valid_results[m1], valid_results[m2])
        scores[(m1, m2)] = score
        time.sleep(1.1) # Respect API Ninjas rate limit

    if not scores:
        return {"best_result": next(iter(valid_results.values()))}

    # 3. Calculate average score for each model
    avg_scores = {model: sum(s for p, s in scores.items() if model in p) / (len(valid_results) - 1) for model in valid_results.keys()}
    
    # 4. Find the best model and result
    best_model_name = max(avg_scores, key=avg_scores.get)
    best_result = results[best_model_name]

    # 5. Create the similarity matrix
    sim_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for (m1, m2), score in scores.items():
        sim_matrix.loc[m1, m2] = score
        sim_matrix.loc[m2, m1] = score
    for model in models:
        sim_matrix.loc[model, model] = 1.0
    matrix_string = sim_matrix.to_string(float_format="%.4f")

    # 6. Create and save the report files
    log_dir = Path("logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = log_dir / f"report_{task_type}_{timestamp}.txt"
    matrix_html_filename = log_dir / f"matrix_{task_type}_{timestamp}.html"
    
    report_content = f"--- Consensus Evaluation Report ---\n"
    report_content += f"Timestamp: {timestamp}\nTask Type: {task_type.upper()}\nPrompt/Theme: {prompt}\n\n"
    report_content += f"--- BEST RESULT (from {best_model_name}) ---\n{best_result}\n\n"
    report_content += f"--- Consensus Scores ---\n"
    for model, score in sorted(avg_scores.items(), key=lambda item: item[1], reverse=True):
        report_content += f"- {model}: {score:.4f}\n"
    report_content += f"\n--- Pairwise Similarity Matrix ---\n{matrix_string}\n"
    report_content += f"\n--- All Model Outputs ---\n"
    for model, output in results.items():
        report_content += f"\n--- Output from {model} ---\n{output}\n"

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # Save the styled matrix 
    
    styled_matrix = sim_matrix.style.format("{:.4f}")  
    styled_matrix.to_html(matrix_html_filename)

    print(f"Evaluation complete. Report and matrix saved to the '{log_dir.name}' folder.")
    
    # 7. Create the final, simplified dictionary to return to the app
    evaluation_report = {
        "best_result": best_result
    }

    return evaluation_report
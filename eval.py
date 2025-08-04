import streamlit as st
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import requests
import time
from datetime import datetime
import os
from pathlib import Path
import shutil


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

def create_consensus_graph(summaries: dict, best_model_name: str, scores: dict, avg_scores: dict):
    """Creates an improved reference-free consensus graph of models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    G = nx.Graph()
    for model_name in summaries.keys(): G.add_node(model_name)
    for (m1, m2), score in scores.items():
        if score > 0.1: G.add_edge(m1, m2, weight=score, label=f'{score:.2f}')
    min_size, max_size = 1500, 5000
    min_score, max_score = (min(avg_scores.values()), max(avg_scores.values())) if avg_scores else (0,1)
    node_sizes = [min_size + (avg_scores.get(node, 0) - min_score) * (max_size - min_size) / (max_score - min_score) if max_score > min_score else min_size for node in G.nodes()]
    labels_with_scores = {model: f"{model}\n(Score: {avg_scores.get(model, 0):.3f})" for model in G.nodes()}
    pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    node_colors = ['#ff9999' if node == best_model_name else '#66b3ff' for node in G.nodes()]
    edge_weights = [d['weight'] * 6 for u, v, d in G.edges(data=True)]
    nx.draw_networkx(G, pos, labels=labels_with_scores, node_size=node_sizes, node_color=node_colors, width=edge_weights, font_size=10, font_weight='bold', ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8, ax=ax)
    ax.set_title('Model Consensus Graph (Node Size = Avg. Score)', size=16)
    plt.tight_layout()
    return fig

# ==============================================================================
# MAIN EVALUATION FUNCTION
# ==============================================================================

def run_consensus_evaluation(client, models, task_type, context, prompt):
    """
    Gets results from all models, performs consensus evaluation, saves a report,
    and returns a full report dictionary.
    """
    
    # 1. Get results from all models
    results = {}
    for model in models:
        if task_type == 'qa':
            result = get_qa_answer(client, prompt, context, model)
        elif task_type == 'summary':
            result = get_summary(client, context, model, f"the theme '{prompt}'")
        results[model] = result
        time.sleep(10)
 
    # 2. Perform pairwise consensus evaluation
    scores = {}
    for m1, m2 in itertools.combinations(models, 2):
        score = get_similarity_score(results[m1], results[m2])
        scores[(m1, m2)] = score
        time.sleep(1.1) # Respect API Ninjas rate limit

    # 3. Calculate average score for each model
    avg_scores = {model: sum(s for p, s in scores.items() if model in p) / (len(models) - 1) for model in models}
    
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
    # The log directory is created by app_v4.py at startup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = log_dir / f"report_{task_type}_{timestamp}.txt"
    graph_filename = log_dir / f"graph_{task_type}_{timestamp}.png"
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
    
    # Create the graph figure
    graph_fig = create_consensus_graph(results, best_model_name, scores, avg_scores)
    graph_fig.savefig(graph_filename)
    
    styled_matrix = sim_matrix.style.background_gradient(cmap='viridis', axis=None).format("{:.4f}")
    styled_matrix.to_html(matrix_html_filename)

    print(f"Evaluation complete. Results saved to the '{log_dir.name}' folder.")
    
    # 7. Create the final report dictionary to return to the app
    evaluation_report = {
        "best_model": best_model_name,
        "best_result": best_result,
        "avg_scores_df": avg_scores,
        "graph_fig": graph_fig,
        "all_results": results,
        "similarity_matrix": sim_matrix
    }

    return evaluation_report
# qa_comparator.py 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time

#  Q&A function from the summarizer module
from summarizer import ask_question as eval_ask_question

def get_api_ninja_similarity(text1: str, text2: str, api_key: str):
    #Helper function to call the API Ninjas Similarity endpoint.
    api_url = 'https://api.api-ninjas.com/v1/textsimilarity'
    headers = {'X-Api-Key': api_key}
    payload = {'text_1': text1, 'text_2': text2}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("similarity", 0.0)
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return 0.0

def compare_model_answers(client, pdf_text: str, question: str, models: list, api_ninja_key: str):
   
    #Generates answers, compares them, calculates consensus, and returns a rich report dictionary.
    
    # 1. Get answers from all models
    answers = {model: eval_ask_question(client, pdf_text, question, model) for model in models}
    model_names = list(answers.keys())
    answer_texts = list(answers.values())
    num_models = len(model_names)

    # 2. Build the similarity matrix using pairwise API calls
    sim_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    for i in range(num_models):
        for j in range(i, num_models):
            if i == j:
                score = 1.0
            else:
                score = get_api_ninja_similarity(answer_texts[i], answer_texts[j], api_ninja_key)
                time.sleep(1.1)
            sim_matrix.iloc[i, j] = score
            sim_matrix.iloc[j, i] = score

    # 3. NEW: Calculate the average similarity for each answer
    avg_scores = []
    for i in range(num_models):
        avg_similarity = (sim_matrix.iloc[i].sum() - 1.0) / (num_models - 1) if num_models > 1 else 1.0
        avg_scores.append({
            "Model": model_names[i],
            "Average Similarity Score": avg_similarity
        })
    avg_scores_df = pd.DataFrame(avg_scores).sort_values(by="Average Similarity Score", ascending=False).reset_index(drop=True)

    # 4. NEW: Identify the best/"consensus" answer
    best_model_name = avg_scores_df.iloc[0]["Model"]
    best_answer_text = answers[best_model_name]

    # 5. Create the visualizations
    # Heatmap (as before)
    heatmap_fig, ax1 = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(sim_matrix, annot=True, cmap='viridis', fmt='.2f', ax=ax1)
    ax1.set_title('Pairwise Answer Similarity Matrix', fontsize=14)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax1.get_yticklabels(), rotation=0)
    heatmap_fig.tight_layout()

    # NEW: Bar chart for average scores
    barchart_fig, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Model", y="Average Similarity Score", data=avg_scores_df, ax=ax2, palette="coolwarm",hue="Model", legend=False)
    ax2.set_title('Model Consensus Score', fontsize=14)
    ax2.set_ylabel('Average Similarity to Other Answers')
    ax2.set_xlabel('')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    barchart_fig.tight_layout()

    # 6. Return all results in a single dictionary
    return {
        "answers": answers,
        "best_model": best_model_name,
        "best_answer": best_answer_text,
        "avg_scores_df": avg_scores_df,
        "heatmap_fig": heatmap_fig,
        "barchart_fig": barchart_fig
    }
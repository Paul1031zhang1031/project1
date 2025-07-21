import requests
import pandas as pd

def calculate_similarity(text1: str, text2: str, api_key: str):
    api_url = 'https://api.api-ninjas.com/v1/textsimilarity'
    headers = {'X-Api-Key': api_key}
    body = {'text_1': text1, 'text_2': text2}
    try:
        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        return result.get('similarity', 0.0)
    except Exception as e:
        print(f"Similarity API error: {e}")
        return 0.0

def create_similarity_scores_against_reference(summaries: dict, reference: str, api_key: str):
    scores = {}
    for model_name, summary in summaries.items():
        if not summary or summary.startswith("Error:"):
            scores[model_name] = 0.0
        else:
            sim = calculate_similarity(summary, reference, api_key)
            scores[model_name] = sim
    return pd.DataFrame.from_dict(scores, orient='index', columns=['Similarity to Reference'])

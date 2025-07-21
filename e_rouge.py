import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_rouge_scores(summaries: dict, reference: str) -> pd.DataFrame:
    """
    Compute ROUGE scores for multiple summaries against a reference.

    Returns a DataFrame indexed by model names with rouge1, rouge2, rougeL scores.
    """
    rouge = evaluate.load("rouge")

    rouge_scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }
    model_names = []

    for model_name, summary in summaries.items():
        if not summary or summary.startswith("Error:"):
            continue
        result = rouge.compute(predictions=[summary], references=[reference])
        model_names.append(model_name)
        rouge_scores["rouge1"].append(result["rouge1"])
        rouge_scores["rouge2"].append(result["rouge2"])
        rouge_scores["rougeL"].append(result["rougeL"])

    df = pd.DataFrame(rouge_scores, index=model_names)
    return df

def plot_rouge_scores(rouge_df: pd.DataFrame):
    """
    Given a DataFrame of ROUGE scores, plot a grouped bar chart and return the figure.
    """
    x = np.arange(len(rouge_df.index))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, rouge_df["rouge1"], width, label="ROUGE-1")
    ax.bar(x, rouge_df["rouge2"], width, label="ROUGE-2")
    ax.bar(x + width, rouge_df["rougeL"], width, label="ROUGE-L")

    ax.set_xticks(x)
    ax.set_xticklabels(rouge_df.index)
    ax.set_ylabel("ROUGE Score")
    ax.set_title("ROUGE Scores of Summaries by Model")
    ax.legend()
    fig.tight_layout()

    return fig

def compute_and_plot_rouge(summaries: dict, reference: str):
    """
    Compute ROUGE scores and plot results.
    Returns (DataFrame, matplotlib.figure.Figure)
    """
    df = compute_rouge_scores(summaries, reference)
    fig = plot_rouge_scores(df)
    return df, fig

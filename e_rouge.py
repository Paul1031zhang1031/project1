import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Compute ROUGE-L scores for multiple summaries against a reference.
def compute_rouge_scores(summaries: dict, reference: str) -> pd.DataFrame:
   

    rouge = evaluate.load("rouge")
    scores = []
    model_names = []

    for model_name, summary in summaries.items():
        if not summary or summary.startswith("Error:"):
            continue
        result = rouge.compute(predictions=[summary], references=[reference])
        model_names.append(model_name)
        scores.append(result["rougeL"])

    return pd.DataFrame({"rougeL": scores}, index=model_names)

def plot_rouge_scores(rouge_df: pd.DataFrame):
    """
    Plot a bar chart of ROUGE-L scores.

    Args:
        rouge_df: DataFrame with 'rougeL' scores indexed by model name

    Returns:
        A matplotlib Figure object
    """
    x = np.arange(len(rouge_df))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, rouge_df["rougeL"], color="skyblue")

    ax.set_xticks(x)
    ax.set_xticklabels(rouge_df.index, rotation=45, ha="right")
    ax.set_ylabel("ROUGE-L Score")
    ax.set_title("ROUGE-L Scores by Model")
    fig.tight_layout()

    return fig

def compute_and_plot_rouge(summaries: dict, reference: str):
    """
    Compute and plot ROUGE-L scores for model summaries.

    Returns:
        Tuple of (DataFrame, matplotlib Figure)
    """
    df = compute_rouge_scores(summaries, reference)
    fig = plot_rouge_scores(df)
    return df, fig

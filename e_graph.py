
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from e_similarity import get_similarity_score

def create_consensus_graph(summaries: dict, api_ninja_key: str, best_model_name: str):
    """Creates a reference-free consensus graph of models."""
    node_names = list(summaries.keys())
    if len(node_names) < 2: return None
    G = nx.Graph()
    G.add_nodes_from(node_names)

    for node1, node2 in itertools.combinations(node_names, 2):
        score = get_similarity_score(summaries[node1], summaries[node2], api_ninja_key)
        if score > 0.5: G.add_edge(node1, node2, weight=score)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    weights = [G[u][v]['weight'] * 6 for u, v in G.edges()]
    node_colors = ['#ff9999' if node == best_model_name else '#66b3ff' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    ax.set_title('Model Consensus Graph (Reference-Free)', size=18)
    plt.axis('off')
    return fig

def create_reference_graph(summaries: dict, reference_summary: str, api_ninja_key: str):
    """Creates a graph showing each model's similarity to a central reference."""
    all_nodes = list(summaries.keys()) + ["Reference"]
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    
    for model_name, model_summary in summaries.items():
        score = get_similarity_score(model_summary, reference_summary, api_ninja_key)
        if score > 0.1: G.add_edge(model_name, "Reference", weight=score)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    weights = [G[u][v]['weight'] * 6 for u, v in G.edges()]
    node_colors = ['#ff9999' if node == 'Reference' else '#66b3ff' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    ax.set_title('Model Similarity to Golden Reference', size=18)
    plt.axis('off')
    return fig
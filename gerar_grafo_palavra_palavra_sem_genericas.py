
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ARQ_NODES = "nodes_word.csv"
ARQ_EDGES = "edges_word.csv"

PASTA_SAIDA = "03_palavras_sem_genericas"
SAIDA_IMG = "rede_palavra_palavra_sem_genericas.png"
SAIDA_STATS = "estatisticas_palavra_palavra_sem_genericas.csv"

FIGSIZE = (18, 12)
DPI = 220
ALPHA_ARESTAS = 0.18
MOSTRAR_LABELS = True
FONTE_LABEL = 9
SEED_FALLBACK = 7

NODE_SIZE_MIN = 80
NODE_SIZE_MAX = 3600
EDGE_W_MIN = 0.4
EDGE_W_MAX = 5.0

FA2_PARAMS = dict(
    outboundAttractionDistribution=True,
    linLogMode=False,
    adjustSizes=True,
    edgeWeightInfluence=1.0,
    jitterTolerance=1.0,
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    scalingRatio=10.0,
    strongGravityMode=False,
    gravity=1.0,
    verbose=False
)
FA2_ITER = 2500

def scale(values, vmin, vmax):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.max() == values.min():
        return np.full_like(values, (vmin + vmax) / 2, dtype=float)
    return vmin + (vmax - vmin) * (values - values.min()) / (values.max() - values.min())

def try_forceatlas2_layout(G):
    try:
        from fa2 import ForceAtlas2  # pip install fa2
        fa2 = ForceAtlas2(**FA2_PARAMS)
        pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=FA2_ITER)
        return pos, "forceatlas2"
    except Exception:
        pos = nx.spring_layout(G, seed=SEED_FALLBACK, k=0.9)
        return pos, "spring_layout_fallback"

def compute_modularity_classes(G):
    try:
        import community as community_louvain
        part = community_louvain.best_partition(G, weight="weight")
        return part, "louvain"
    except Exception:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        part = {}
        for i, comm in enumerate(communities):
            for n in comm:
                part[n] = i
        return part, "greedy_modularity_fallback"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    nodes_path = os.path.join(script_dir, ARQ_NODES)
    edges_path = os.path.join(script_dir, ARQ_EDGES)

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Não encontrei: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Não encontrei: {edges_path}")


    outdir = os.path.join(script_dir, PASTA_SAIDA)
    os.makedirs(outdir, exist_ok=True)

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    if "Id" not in nodes.columns or "Frequency" not in nodes.columns:
        raise ValueError("nodes_word.csv precisa ter colunas 'Id' e 'Frequency'")
    for c in ["Source", "Target", "Weight"]:
        if c not in edges.columns:
            raise ValueError(f"edges_word.csv precisa ter a coluna '{c}'")

   
    G = nx.Graph()

    for _, r in nodes.iterrows():
        node_id = r["Id"]
        freq = int(r["Frequency"]) if not pd.isna(r["Frequency"]) else 0
        G.add_node(node_id, frequency=freq)

    for _, r in edges.iterrows():
        u = r["Source"]
        v = r["Target"]
        w = int(r["Weight"]) if not pd.isna(r["Weight"]) else 1
        if u not in G:
            G.add_node(u, frequency=0)
        if v not in G:
            G.add_node(v, frequency=0)
        G.add_edge(u, v, weight=w)


    degree = dict(G.degree())
    wdegree = dict(G.degree(weight="weight"))
    betw = nx.betweenness_centrality(G, weight="weight", normalized=True)

    part, part_method = compute_modularity_classes(G)
    nx.set_node_attributes(G, part, "modularity_class")

 
    pos, layout_method = try_forceatlas2_layout(G)

    node_list = list(G.nodes())
    edge_list = list(G.edges())

    node_sizes = scale([wdegree[n] for n in node_list], NODE_SIZE_MIN, NODE_SIZE_MAX)
    edge_widths = scale([G[u][v].get("weight", 1) for u, v in edge_list], EDGE_W_MIN, EDGE_W_MAX)

    classes = [part.get(n, 0) for n in node_list]
    unique_classes = sorted(set(classes))
    cmap = plt.get_cmap("tab20")
    class_to_color = {c: cmap(i % 20) for i, c in enumerate(unique_classes)}
    node_colors = [class_to_color[c] for c in classes]

    labels = {n: n for n in node_list} if MOSTRAR_LABELS else {}

    plt.figure(figsize=FIGSIZE)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=ALPHA_ARESTAS)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.95)
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=FONTE_LABEL)

    plt.title(f"Rede Palavra × Palavra (Sem genéricas)")
    plt.axis("off")
    plt.tight_layout()

    out_img = os.path.join(outdir, SAIDA_IMG)
    plt.savefig(out_img, dpi=DPI, bbox_inches="tight")
    plt.close()

    stats = pd.DataFrame({
        "node": node_list,
        "frequency": [G.nodes[n].get("frequency", 0) for n in node_list],
        "degree": [degree[n] for n in node_list],
        "weighted_degree": [wdegree[n] for n in node_list],
        "betweenness": [betw[n] for n in node_list],
        "modularity_class": [part.get(n, 0) for n in node_list],
    }).sort_values(["weighted_degree", "degree", "frequency"], ascending=False).reset_index(drop=True)

    out_stats = os.path.join(outdir, SAIDA_STATS)
    stats.to_csv(out_stats, index=False, encoding="utf-8")

    print("[OK] Gerado (SEM genéricas):")
    print(" -", out_img)
    print(" -", out_stats)

if __name__ == "__main__":
    main()

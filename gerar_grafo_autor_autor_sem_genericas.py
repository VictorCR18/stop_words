import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

PASTA_SAIDA = "04_autor_autor_sem_genericas"

ARQ_NODES = "nodes_author_author.csv"
ARQ_EDGES = "edges_author_author.csv"

SAIDA_IMG = "rede_autor_autor.png"
SAIDA_STATS = "estatisticas_autor_autor.csv"

FIGSIZE = (20, 14)
DPI = 220
ALPHA_ARESTAS = 0.18

MOSTRAR_LABELS = True
FONTE_LABEL = 8

SEED_FALLBACK = 7

NODE_SIZE_MIN = 80
NODE_SIZE_MAX = 4200
EDGE_W_MIN = 0.3
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
        from fa2 import ForceAtlas2
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

    outdir = os.path.join(script_dir, PASTA_SAIDA)
    os.makedirs(outdir, exist_ok=True)

    nodes_path = os.path.join(outdir, ARQ_NODES)
    edges_path = os.path.join(outdir, ARQ_EDGES)

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Não encontrei: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Não encontrei: {edges_path}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    G = nx.Graph()

    for _, r in nodes.iterrows():
        node_id = r["Id"]
        freq = int(r["Frequency"]) if "Frequency" in nodes.columns and not pd.isna(r["Frequency"]) else 0
        G.add_node(node_id, frequency=freq)

    for _, r in edges.iterrows():
        u = r["Source"]
        v = r["Target"]
        w = float(r["Weight"]) if not pd.isna(r["Weight"]) else 0.0
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
    edge_widths = scale([G[u][v].get("weight", 0.0) for u, v in edge_list], EDGE_W_MIN, EDGE_W_MAX)

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

    plt.title(f"Rede Autor × Autor (Sem genéricas)")
    plt.axis("off")
    plt.tight_layout()

    out_img = os.path.join(outdir, SAIDA_IMG)
    plt.savefig(out_img, dpi=DPI, bbox_inches="tight")
    plt.close()

    stats = pd.DataFrame({
        "node": node_list,
        "frequency_posts": [G.nodes[n].get("frequency", 0) for n in node_list],
        "degree": [degree[n] for n in node_list],
        "weighted_degree": [wdegree[n] for n in node_list],
        "betweenness": [betw[n] for n in node_list],
        "modularity_class": [part.get(n, 0) for n in node_list],
    }).sort_values(["weighted_degree", "degree", "frequency_posts"], ascending=False).reset_index(drop=True)

    out_stats = os.path.join(outdir, SAIDA_STATS)
    stats.to_csv(out_stats, index=False, encoding="utf-8")

    print("[OK] Gerado dentro de 04_gephi_autor_autor_sem_genericas:")
    print(" -", out_img)
    print(" -", out_stats)

if __name__ == "__main__":
    main()

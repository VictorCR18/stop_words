

import os
import re
import itertools
from collections import Counter
import pandas as pd

PASTA_ENTRADA = "00_dados_limpos"
ARQUIVO_ENTRADA = "cleaned_posts.csv"

GERAR_VERSAO_COMPLETA = True
GERAR_VERSAO_SEM_GENERICAS = True


MIN_FREQ_HASHTAG = 3
MIN_PESO_ARESTA = 2

HASHTAGS_GENERICAS = {"#bbb", "#bbb26"}

HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)

def extract_hashtags(text: str) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    return [t.lower() for t in HASHTAG_RE.findall(text)]

def build_edges(tags_per_post: list[list[str]]) -> Counter:
    edges = Counter()
    for tags in tags_per_post:
        uniq = sorted(set(tags))
        if len(uniq) < 2:
            continue
        for u, v in itertools.combinations(uniq, 2):
            edges[(u, v)] += 1
    return edges

def export_gephi(nodes: pd.DataFrame, edges: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    nodes_path = os.path.join(outdir, "nodes_hashtag.csv")
    edges_path = os.path.join(outdir, "edges_hashtag.csv")
    nodes.to_csv(nodes_path, index=False, encoding="utf-8")
    edges.to_csv(edges_path, index=False, encoding="utf-8")
    print(f"[OK] Exportado: {outdir}")
    print(f"     nós: {nodes_path} ({len(nodes)})")
    print(f"     arestas: {edges_path} ({len(edges)})")

def gerar_rede(df: pd.DataFrame, remover_genericas: bool, outdir: str):
    
    col_texto = "text" if "text" in df.columns else "text_clean"
    df["hashtags"] = df[col_texto].apply(extract_hashtags)

    if remover_genericas:
        df["hashtags"] = df["hashtags"].apply(lambda tags: [t for t in tags if t not in HASHTAGS_GENERICAS])


    counts = Counter([h for tags in df["hashtags"] for h in tags])
    counts = Counter({k: v for k, v in counts.items() if v >= MIN_FREQ_HASHTAG})

    tags_per_post = [[t for t in tags if t in counts] for tags in df["hashtags"]]

    
    edge_counts = build_edges(tags_per_post)
    edge_counts = Counter({k: v for k, v in edge_counts.items() if v >= MIN_PESO_ARESTA})

    nodes = pd.DataFrame(
        [{"Id": k, "Label": k, "Type": "hashtag", "Frequency": int(v)} for k, v in counts.items()]
    ).sort_values("Frequency", ascending=False)

    edges = pd.DataFrame(
        [{"Source": u, "Target": v, "Weight": int(w)} for (u, v), w in edge_counts.items()]
    ).sort_values("Weight", ascending=False)

    export_gephi(nodes, edges, outdir)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, PASTA_ENTRADA, ARQUIVO_ENTRADA)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Não encontrei: {input_path}")

    df = pd.read_csv(input_path)

    if GERAR_VERSAO_COMPLETA:
        outdir = os.path.join(script_dir, "01_gephi_hashtags")
        gerar_rede(df.copy(), remover_genericas=False, outdir=outdir)

    if GERAR_VERSAO_SEM_GENERICAS:
        outdir = os.path.join(script_dir, "01_gephi_hashtags_sem_genericas")
        gerar_rede(df.copy(), remover_genericas=True, outdir=outdir)

if __name__ == "__main__":
    main()

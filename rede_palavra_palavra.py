import os
import itertools
from collections import Counter
import pandas as pd

PASTA_ENTRADA = "00_dados_limpos"
ARQUIVO_ENTRADA = "cleaned_posts.csv"

GERAR_VERSAO_COMPLETA = True
GERAR_VERSAO_SEM_GENERICAS = True


MIN_FREQ_PALAVRA = 5     
MIN_PESO_ARESTA = 3      
MAX_NOS = 300          


TERMOS_GENERICOS = {
    "bbb", "bbb26", "redebbb", "globo", "bigday", "big", "day"
}

def build_edges(words_per_post: list[list[str]]) -> Counter:
    edges = Counter()
    for ws in words_per_post:
        uniq = sorted(set(ws))
        if len(uniq) < 2:
            continue
        for u, v in itertools.combinations(uniq, 2):
            edges[(u, v)] += 1
    return edges

def export_gephi(nodes: pd.DataFrame, edges: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    nodes_path = os.path.join(outdir, "nodes_word.csv")
    edges_path = os.path.join(outdir, "edges_word.csv")
    nodes.to_csv(nodes_path, index=False, encoding="utf-8")
    edges.to_csv(edges_path, index=False, encoding="utf-8")
    print(f"[OK] Exportado: {outdir}")
    print(f"     nós: {nodes_path} ({len(nodes)})")
    print(f"     arestas: {edges_path} ({len(edges)})")

def gerar_rede(df: pd.DataFrame, remover_genericas: bool, outdir: str):
    
    if "tokens_str" in df.columns:
        serie_tokens = df["tokens_str"].fillna("")
    elif "text_clean" in df.columns:
        
        serie_tokens = df["text_clean"].fillna("")
    else:
        raise ValueError("CSV precisa ter 'tokens_str' (recomendado) ou 'text_clean'.")

    words_per_post = []
    for s in serie_tokens:
        ws = [w for w in str(s).split() if w]
        if remover_genericas:
            ws = [w for w in ws if w not in TERMOS_GENERICOS]
        words_per_post.append(ws)

    freq = Counter()
    for ws in words_per_post:
        for w in set(ws):
            freq[w] += 1

    freq = Counter({k: v for k, v in freq.items() if v >= MIN_FREQ_PALAVRA})

    if MAX_NOS and len(freq) > MAX_NOS:
        top = dict(freq.most_common(MAX_NOS))
        freq = Counter(top)

    words_per_post = [[w for w in ws if w in freq] for ws in words_per_post]


    edge_counts = build_edges(words_per_post)
    edge_counts = Counter({k: v for k, v in edge_counts.items() if v >= MIN_PESO_ARESTA})

    nodes = pd.DataFrame(
        [{"Id": w, "Label": w, "Type": "word", "Frequency": int(c)} for w, c in freq.items()]
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
        outdir = os.path.join(script_dir, "03_gephi_palavras")
        gerar_rede(df.copy(), remover_genericas=False, outdir=outdir)

    if GERAR_VERSAO_SEM_GENERICAS:
        outdir = os.path.join(script_dir, "03_gephi_palavras_sem_genericas")
        gerar_rede(df.copy(), remover_genericas=True, outdir=outdir)

if __name__ == "__main__":
    main()

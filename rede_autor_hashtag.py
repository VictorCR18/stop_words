import os
import re
from collections import Counter
import pandas as pd

PASTA_ENTRADA = "00_dados_limpos"
ARQUIVO_ENTRADA = "cleaned_posts.csv"


GERAR_VERSAO_COMPLETA = True
GERAR_VERSAO_SEM_GENERICAS = True


MIN_FREQ_HASHTAG = 3       
MIN_PESO_ARESTA = 2         
MIN_POSTS_AUTOR = 3         

HASHTAGS_GENERICAS = {"#bbb", "#bbb26"} 

HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)

def extract_hashtags(text: str) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    return [t.lower() for t in HASHTAG_RE.findall(text)]

def export_gephi(nodes: pd.DataFrame, edges: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    nodes_path = os.path.join(outdir, "nodes_author_hashtag.csv")
    edges_path = os.path.join(outdir, "edges_author_hashtag.csv")
    nodes.to_csv(nodes_path, index=False, encoding="utf-8")
    edges.to_csv(edges_path, index=False, encoding="utf-8")
    print(f"[OK] Exportado: {outdir}")
    print(f"     nós: {nodes_path} ({len(nodes)})")
    print(f"     arestas: {edges_path} ({len(edges)})")

def gerar_rede(df: pd.DataFrame, remover_genericas: bool, outdir: str):
    
    col_texto = "text" if "text" in df.columns else "text_clean"

    if "author_handle" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'author_handle'.")

    
    df["hashtags"] = df[col_texto].apply(extract_hashtags)

    
    if remover_genericas:
        df["hashtags"] = df["hashtags"].apply(lambda tags: [t for t in tags if t not in HASHTAGS_GENERICAS])

   
    posts_por_autor = df.groupby("author_handle").size()
    autores_validos = set(posts_por_autor[posts_por_autor >= MIN_POSTS_AUTOR].index)
    df = df[df["author_handle"].isin(autores_validos)].copy()


    freq_hashtag = Counter([h for tags in df["hashtags"] for h in tags])
    freq_hashtag = Counter({k: v for k, v in freq_hashtag.items() if v >= MIN_FREQ_HASHTAG})

    df["hashtags"] = df["hashtags"].apply(lambda tags: [t for t in tags if t in freq_hashtag])

    edge_counts = Counter()
    for _, row in df.iterrows():
        autor = row["author_handle"]
       
        for tag in set(row["hashtags"]):
            edge_counts[(autor, tag)] += 1

    edge_counts = Counter({k: v for k, v in edge_counts.items() if v >= MIN_PESO_ARESTA})

    edges = pd.DataFrame(
        [{"Source": a, "Target": t, "Weight": int(w)} for (a, t), w in edge_counts.items()]
    ).sort_values("Weight", ascending=False)

    posts_por_autor = df.groupby("author_handle").size().to_dict()

    nodes_autores = pd.DataFrame(
        [{"Id": a, "Label": a, "Type": "author", "Frequency": int(n)} for a, n in posts_por_autor.items()]
    )

    nodes_tags = pd.DataFrame(
        [{"Id": t, "Label": t, "Type": "hashtag", "Frequency": int(n)} for t, n in freq_hashtag.items()]
    )

    nodes = pd.concat([nodes_autores, nodes_tags], ignore_index=True)

    export_gephi(nodes, edges, outdir)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, PASTA_ENTRADA, ARQUIVO_ENTRADA)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Não encontrei: {input_path}")

    df = pd.read_csv(input_path)

    if GERAR_VERSAO_COMPLETA:
        outdir = os.path.join(script_dir, "02_gephi_autor_hashtag")
        gerar_rede(df.copy(), remover_genericas=False, outdir=outdir)

    if GERAR_VERSAO_SEM_GENERICAS:
        outdir = os.path.join(script_dir, "02_gephi_autor_hashtag_sem_genericas")
        gerar_rede(df.copy(), remover_genericas=True, outdir=outdir)

if __name__ == "__main__":
    main()

import os
import re
import itertools
import pandas as pd

PASTA_ENTRADA = "00_dados_limpos"
ARQUIVO_ENTRADA = "cleaned_posts.csv"

PASTA_SAIDA = "04_gephi_autor_autor_sem_genericas"

HASHTAGS_GENERICAS = {"#bbb", "#bbb26"}


MIN_POSTS_AUTOR = 3           
MIN_HASHTAGS_UNICAS = 2       
MIN_SHARED_HASHTAGS = 2       
MIN_JACCARD = 0.08           

HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)

def extract_hashtags(text: str) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    return [t.lower() for t in HASHTAG_RE.findall(text)]

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, PASTA_ENTRADA, ARQUIVO_ENTRADA)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Não encontrei: {input_path}")

    outdir = os.path.join(script_dir, PASTA_SAIDA)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_path)

    if "author_handle" not in df.columns:
        raise ValueError("O CSV precisa ter a coluna 'author_handle'.")
    if "text_clean" not in df.columns and "text" not in df.columns:
        raise ValueError("O CSV precisa ter 'text_clean' ou 'text'.")

    col_texto = "text_clean" if "text_clean" in df.columns else "text"
    df["hashtags"] = df[col_texto].apply(extract_hashtags)

    
    df["hashtags"] = df["hashtags"].apply(lambda tags: [t for t in tags if t not in HASHTAGS_GENERICAS])

   
    posts_por_autor = df.groupby("author_handle").size().to_dict()

    hashtags_por_autor = {}
    for autor, grp in df.groupby("author_handle"):
        s = set()
        for tags in grp["hashtags"]:
            s.update(tags)
        hashtags_por_autor[autor] = s

    autores_validos = []
    for autor, n_posts in posts_por_autor.items():
        if n_posts < MIN_POSTS_AUTOR:
            continue
        if len(hashtags_por_autor.get(autor, set())) < MIN_HASHTAGS_UNICAS:
            continue
        autores_validos.append(autor)

    autores_validos = sorted(autores_validos)
    hashtags_por_autor = {a: hashtags_por_autor[a] for a in autores_validos}

    edges_rows = []
    for a, b in itertools.combinations(autores_validos, 2):
        ha = hashtags_por_autor[a]
        hb = hashtags_por_autor[b]
        shared = len(ha & hb)
        if shared < MIN_SHARED_HASHTAGS:
            continue
        jac = jaccard(ha, hb)
        if jac < MIN_JACCARD:
            continue
        edges_rows.append({
            "Source": a,
            "Target": b,
            "Weight": round(float(jac), 6),  
            "Shared": int(shared),
        })

    edges = pd.DataFrame(edges_rows).sort_values(["Weight", "Shared"], ascending=False)

    nodes_rows = []
    for a in autores_validos:
        nodes_rows.append({
            "Id": a,
            "Label": a,
            "Type": "author",
            "Frequency": int(posts_por_autor.get(a, 0)),          
            "UniqueHashtags": int(len(hashtags_por_autor.get(a, set()))),
        })
    nodes = pd.DataFrame(nodes_rows).sort_values(["Frequency", "UniqueHashtags"], ascending=False)

    nodes_path = os.path.join(outdir, "nodes_author_author.csv")
    edges_path = os.path.join(outdir, "edges_author_author.csv")

    nodes.to_csv(nodes_path, index=False, encoding="utf-8")
    edges.to_csv(edges_path, index=False, encoding="utf-8")

    print("[OK] Autor×Autor (SEM genéricas) gerado em:")
    print(" -", nodes_path)
    print(" -", edges_path)
    print(f"[INFO] Nós: {len(nodes)} | Arestas: {len(edges)}")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

PASTA_ENTRADA = "00_dados_limpos"
ARQUIVO_ENTRADA = "cleaned_posts.csv"

PASTA_SAIDA = "04_nuvem_palavras"
ARQ_SAIDA = "nuvem_palavras.png"

REMOVER_GENERICAS = True
GENERICAS = {
    "bbb", "bbb26", "#bbb", "#bbb26"
}


LARGURA = 1600
ALTURA = 900
FUNDO_BRANCO = True

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(script_dir, PASTA_ENTRADA, ARQUIVO_ENTRADA)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Não encontrei: {input_path}")

    outdir = os.path.join(script_dir, PASTA_SAIDA)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_path)

    if "tokens_str" in df.columns:
        textos = df["tokens_str"].fillna("").astype(str).tolist()
    elif "text_clean" in df.columns:
        textos = df["text_clean"].fillna("").astype(str).tolist()
    elif "text" in df.columns:
        textos = df["text"].fillna("").astype(str).tolist()
    else:
        raise ValueError("O CSV precisa ter uma coluna 'tokens_str' ou 'text_clean' ou 'text'.")

    texto_unico = " ".join(textos)

    if REMOVER_GENERICAS and GENERICAS:
        palavras = texto_unico.split()
        palavras = [p for p in palavras if p.lower() not in GENERICAS]
        texto_unico = " ".join(palavras)

    wc = WordCloud(
        width=LARGURA,
        height=ALTURA,
        background_color="white" if FUNDO_BRANCO else None,
        collocations=False, 
    ).generate(texto_unico)

    out_img = os.path.join(outdir, ARQ_SAIDA)
    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_img, dpi=220, bbox_inches="tight")
    plt.close()

    print("[OK] Nuvem de palavras gerada em:")
    print(" -", out_img)

if __name__ == "__main__":
    main()

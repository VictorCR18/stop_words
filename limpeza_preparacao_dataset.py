
import os
import re
import pandas as pd

ARQUIVO_ENTRADA = "posts-bbb26.csv"     
PASTA_SAIDA = "00_dados_limpos"        

APLICAR_LOWERCASE = True
FILTRAR_AGREGADORES = True
REMOVER_VAZIOS = True
GERAR_TOKENS = True                    

AUTORES_AGREGADORES = {"nowbreezing.ntw.app", "hourlybreezing.ntw.app"}

BLACKLIST_TOKENS = {"bbb", "bbb26", "redebbb"} 
MIN_LEN_TOKEN = 3


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w#@À-ÖØ-öø-ÿ\s]", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
TRENDING_RE = re.compile(r"trending words", flags=re.IGNORECASE)

STOPWORDS_PT = {
    "a","à","agora","ai","aí","ainda","além","algo","algum","alguma","algumas","alguns","ao","aos",
    "apenas","aqui","as","até","bem","boa","boas","bom","bons","cada","cadê","cê","cem","certo","como",
    "com","contra","da","das","de","dela","dele","deles","delas","demais","depois","desde","dessa",
    "desse","deste","desta","disso","disto","do","dos","e","é","ela","ele","eles","elas","em","era",
    "eram","essa","esse","esta","está","estão","estava","estavam","este","estes","estas","eu","foi",
    "foram","há","isso","isto","já","lá","lhe","lhes","mais","mas","me","mesmo","meu","minha","meus",
    "minhas","muita","muitas","muito","muitos","na","nas","não","nem","nessa","nesse","nesta","neste",
    "no","nos","nós","nossa","nosso","nossas","nossos","num","numa","o","os","ou","para","pela","pelas",
    "pelo","pelos","per","por","pra","pro","pros","pras","qual","quando","que","quem","se","sem","seu",
    "sua","seus","suas","só","sobre","também","tão","tem","têm","tinha","tinham","toda","todas","todo",
    "todos","um","uma","umas","uns","vai","vão","vc","vcs","você","vocês","tô","tá","tava","tavam",
    "rs","kkk","kkkk","kk","pq","porque","porquê","por que","p","q"
}

def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = URL_RE.sub(" ", s)
    s = NON_WORD_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s.lower() if APLICAR_LOWERCASE else s

def tokenizar(texto_limpo: str):
    if not isinstance(texto_limpo, str):
        return []
    toks = [t for t in texto_limpo.split() if len(t) >= MIN_LEN_TOKEN]
    toks = [t for t in toks if not t.startswith("#") and not t.startswith("@")]
    toks = [t for t in toks if t not in STOPWORDS_PT and t not in BLACKLIST_TOKENS]
    return toks

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, ARQUIVO_ENTRADA)
    outdir_path = os.path.join(script_dir, PASTA_SAIDA)
    os.makedirs(outdir_path, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Não encontrei o arquivo: {input_path}")

    df = pd.read_csv(input_path)

    
    if "indexed_at" in df.columns:
        df["indexed_at"] = pd.to_datetime(df["indexed_at"], errors="coerce", utc=True)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

   
    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].apply(clean_text)

    
    if FILTRAR_AGREGADORES:
        if "author_handle" in df.columns:
            df = df[~df["author_handle"].isin(AUTORES_AGREGADORES)].copy()
        df = df[~df["text"].str.contains(TRENDING_RE, na=False)].copy()


    if REMOVER_VAZIOS:
        df = df[df["text_clean"].str.len() > 0].copy()

    
    if GERAR_TOKENS:
        df["tokens"] = df["text_clean"].apply(tokenizar)
        df["tokens_str"] = df["tokens"].apply(lambda lst: " ".join(lst))

    out_path = os.path.join(outdir_path, "cleaned_posts.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Dataset limpo salvo em: {out_path} ({len(df)} linhas)")

if __name__ == "__main__":
    main()

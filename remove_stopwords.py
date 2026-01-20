import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Ler o arquivo unificado
df = pd.read_csv("posts-bbb26-rivalidades-filtrados.csv")

# --- NOVO BLOCO: COLAGEM DE NOMES ---
# Isso transforma "ana paula" em "anapaula" para o grafo entender que é uma pessoa só
print("A unificar nomes compostos...")
df["text"] = df["text"].astype(str).str.lower()  # Passa tudo para minúsculo primeiro

# Ana Paula Renault
df["text"] = df["text"].str.replace("ana paula", "anapaula", regex=False)
df["text"] = df["text"].str.replace("ana renault", "anapaula", regex=False) # Caso chamem assim

# Edilson Capetinha
df["text"] = df["text"].str.replace("edilson capetinha", "edilson", regex=False)
df["text"] = df["text"].str.replace("capetinha", "edilson", regex=False)

# Babu Santana (muita gente chama só de Babu, mas previne "babu santana")
df["text"] = df["text"].str.replace("babu santana", "babu", regex=False)
# ------------------------------------

# Configurar stop words padrão (português)
stop_words = set(stopwords.words('portuguese'))

# Lista Negra do BBB (Mantida igual à sua)
custom_stops = {
    # --- Termos Genéricos ---
    'bbb', 'bbb26', 'redebbb', 'reality', 'globo', 'gshow', 'boninho', 
    'votar', 'voto', 'paredao', 'paredão', 'assistir', 'payperview', 'ppv',
    'globoplay', 'multishow', 'programa', 'edição', 'participante', 'brother', 
    'sister', 'casa', 'jogo', 'jogar', 'confessionário', 'prova', 'lider', 
    'anjo', 'monstro', 'big', 'brother', 'brasil', 'dummy', 'estalo',
    'team', 'fora', 'torcida', 'torcer', 'fandom', 'adm', 'adms', 'perfil',

    # --- Verbos e Ações ---
    'fazer', 'pode', 'disse', 'sobre', 'então', 'coisa', 'falar', 'falou',
    'ser', 'ter', 'ir', 'ver', 'estar', 'ficar', 'dar', 'haver', 'achar',
    'tá', 'ta', 'tava', 'tô', 'to', 'vai', 'foi', 'era', 'é', 'tem', 'tinha',
    'quer', 'queria', 'sabe', 'saber', 'dizer', 'olha', 'olhar', 'vem', 'vamos',
    'sair', 'saiu', 'entrar', 'entrou', 'ganhar', 'perder', 'botar', 'tirar',

    # --- Gírias ---
    'vc', 'vcs', 'pq', 'so', 'só', 'ne', 'né', 'ai', 'aí', 'la', 'lá', 
    'pra', 'pro', 'mto', 'mt', 'tbm', 'tb', 'td', 'n', 'q', 'que', 
    'kkk', 'kkkk', 'kkkkk', 'kkkkkk', 'haha', 'hahaha', 'rs', 'aff', 'mds', 
    'pqp', 'vt', 'off', 'on', 'pov', 'fic', 'flop', 'hype',

    # --- Tempo/Lugar/Pronomes ---
    'hoje', 'agora', 'ontem', 'amanhã', 'dia', 'noite', 'semana', 'ano',
    'sempre', 'nunca', 'antes', 'depois', 'logo', 'ainda', 'já', 'ja',
    'aqui', 'ali', 'lugar', 'momento', 'hora', 'vez', 'vezes',
    'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas', 'meu', 'teu', 
    'seu', 'nosso', 'minha', 'sua', 'disso', 'daquilo', 'nesse', 'nessa', 
    'esse', 'essa', 'isso', 'aquilo', 'este', 'esta', 'quem', 'qual', 'onde',
    'gente', 'pessoa', 'pessoal', 'galera', 'mundo', 'todo', 'toda', 'todos'
}
stop_words.update(custom_stops)

# Remover stop words e limpar o texto
print("A limpar os textos... (isto pode demorar alguns segundos)")
df["text_clean"] = df["text"].apply(lambda post: " ".join(
    [w for w in word_tokenize(post) if w.isalnum() and w not in stop_words]
))

# Salvar o arquivo final
df.to_csv("posts-bbb26-limpos.csv", index=False)
print("Limpeza concluída! Arquivo 'posts-bbb26-limpos.csv' criado.")
import pandas as pd
import string

# Lista personalizada de stop words
stop_words = {
    "a", "à", "agora", "ainda", "alguém", "algum", "alguma", "algumas", "alguns",
    "ampla", "amplas", "amplo", "amplos", "ante", "antes", "ao", "aos", "após",
    "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", "através",
    "cada", "coisa", "coisas", "com", "como", "contra", "contudo", "da", "daquele",
    "daqueles", "das", "de", "dela", "delas", "dele", "deles", "depois", "dessa",
    "dessas", "desse", "desses", "desta", "destas", "deste", "destes", "deve", 
    "devem", "devendo", "dever", "deverá", "deverão", "deveria", "deveriam", "devia", 
    "deviam", "disse", "disso", "disto", "dito", "diz", "dizem", "do", "dos", "e", "é",
    "ela", "elas", "ele", "eles", "em", "enquanto", "entre", "era", "essa", "essas",
    "esse", "esses", "esta", "está", "estamos", "estão", "estas", "estava", "estavam",
    "estávamos", "este", "estes", "estou", "eu", "fazendo", "fazer", "foi", "for",
    "foram", "fosse", "fossem", "grande", "grandes", "há", "isso", "isto", "já", "la",
    "lá", "lhe", "lhes", "lo", "mas", "me", "mesma", "mesmas", "mesmo", "mesmos",
    "meu", "meus", "minha", "minhas", "muita", "muitas", "muito", "muitos", "na", "não",
    "nas", "nem", "nenhum", "nessa", "nessas", "nesta", "nestas", "ninguém", "no", "nos",
    "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "nunca", "o", "os", "ou",
    "outra", "outras", "outro", "outros", "para", "pela", "pelas", "pelo", "pelos",
    "pode", "pois", "por", "porém", "porque", "posso", "primeiro", "primeiros",
    "próprio", "próprios", "quais", "qual", "quando", "quanto", "quantos", "que",
    "quem", "são", "se", "seja", "sem", "sempre", "será", "seu", "seus", "só",
    "sob", "sobre", "sua", "suas", "também", "te", "tem", "tendo", "ter", "teu",
    "teus", "toda", "todas", "todo", "todos", "tu", "tua", "tuas", "tudo",
    "um", "uma", "umas", "uns", "vendo", "ver", "vez", "vir", "vos", "vós"
}

def remove_stop_words(text):
    """Remove stop words e pontuações de um texto."""
    if isinstance(text, str):  # Verifica se é uma string
        # Remove caracteres especiais (*, ", ', etc.)
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        return " ".join(cleaned_words)
    return text  # Retorna o original se não for string

input_csv = "search_results_pix sonegação_2025_02_12.csv"
output_csv = "pix_sonegação_dados_sem_stopwords.csv"

df = pd.read_csv(input_csv)

df = df.applymap(remove_stop_words)

df.to_csv(output_csv, index=False)

print(f"Processo concluído! Arquivo salvo como: {output_csv}")
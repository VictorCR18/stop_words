import pandas as pd

def remove_duplicates(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df_unique = df.drop_duplicates()
    df_unique.to_csv(output_csv, index=False)

input_csv = "posts-pix.csv"
output_csv = "posts-limpos.csv"
remove_duplicates(input_csv, output_csv)
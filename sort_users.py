import pandas as pd

df = pd.read_csv('metricasusuarios.csv')

df['total'] = df['reply_count'] + df['repost_count'] + df['like_count']
df_sorted = df.sort_values(by='total', ascending=False)
df_sorted.to_csv('metricasusuarios_sorted.csv', index=False)
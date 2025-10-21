import pandas as pd

df = pd.read_csv('Harvard Fees Dataset.csv')
pivot_df = df.pivot(index='academic.year', columns='school', values='cost')
pivot_df.reset_index(inplace=True)
pivot_df.to_csv('dataset.csv', index=False)
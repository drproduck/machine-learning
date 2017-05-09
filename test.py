import pandas as pd

df = pd.read_csv('/home/drproduck/table.csv')
print(df.head())
df = df.sort(columns=['Date'])
print(df.head())

df.to_csv('/home/drproduck/stock1.csv')
print(df.columns)


import pandas as pd


df = pd.read_pickle("oracle_rf_ALL_DATA.pkl")

print(df.columns())
print(df.head())
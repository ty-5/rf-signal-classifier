import pandas as pd


df = pd.read_pickle(r"oracle_rf_ALL_DATA.pkl")

print(df.size, df.shape)
print(df.info(verbose=True))
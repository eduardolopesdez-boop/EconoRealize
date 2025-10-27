import pandas as pd

def merge_bases(df_interna, df_bcb):
    df_interna["data"] = pd.to_datetime(df_interna["data"])
    df_bcb["data"] = pd.to_datetime(df_bcb["data"])
    merged = pd.merge(df_interna, df_bcb, on="data", how="left")
    merged = merged.sort_values("data").reset_index(drop=True)
    return merged

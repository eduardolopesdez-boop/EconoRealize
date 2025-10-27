import pandas as pd
import numpy as np

def formatar_valor(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def limpar_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates().dropna(how="all")
    return df

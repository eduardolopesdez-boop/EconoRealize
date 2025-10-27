def calcular_correlação(df, col1, col2):
    return df[[col1, col2]].corr().iloc[0, 1]

def rolling_corr(df, col1, col2, janela=6):
    return df[[col1, col2]].rolling(janela).corr().iloc[0::2, 1].reset_index(drop=True)

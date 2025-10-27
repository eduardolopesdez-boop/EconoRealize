# backend/forecast_models.py
import pandas as pd
import numpy as np
import statsmodels.api as sm


def regressao_linear(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    """
    Ajusta um OLS robusto:
    - conserva apenas colunas existentes
    - força numérico e remove NaN
    - adiciona intercepto ('const')
    """
    x_cols = [c for c in x_cols if c in df.columns]
    work = df.copy()

    for c in [y_col] + x_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[y_col] + x_cols)
    X = sm.add_constant(work[x_cols], has_constant="add")
    y = work[y_col]

    modelo = sm.OLS(y, X).fit()
    # guardo os nomes do design para uso nas previsões
    modelo._x_names = list(modelo.params.index)  # ex.: ['const', 'selic_mensal', ...]
    return modelo


def gerar_cenarios(modelo, df_ref: pd.DataFrame, nome_variavel_alvo: str) -> pd.DataFrame:
    """
    Gera projeções de cenários para modelos OLS com múltiplas variáveis.
    Mantém todas as regressoras nos últimos valores disponíveis (ou 0/média),
    alterando apenas `nome_variavel_alvo`.
    """
    cols_modelo = list(modelo.params.index)  # mesma ordem do treino (inclui 'const')

    if nome_variavel_alvo not in cols_modelo:
        raise ValueError(
            f"Variável '{nome_variavel_alvo}' não está no modelo. "
            f"Variáveis do modelo: {cols_modelo}"
        )

    # linha-base: const=1 e demais = último valor disponível no df_ref (fallback: 0.0)
    base = {}
    for c in cols_modelo:
        if c == "const":
            base[c] = 1.0
        else:
            if df_ref is not None and c in df_ref and df_ref[c].notna().any():
                base[c] = float(df_ref[c].dropna().iloc[-1])
            else:
                # fallback conservador quando a coluna não existe no df_ref
                base[c] = float(df_ref[c].mean()) if (df_ref is not None and c in df_ref) else 0.0

    valor_atual = float(base.get(nome_variavel_alvo, 0.0))

    def _pred(valor):
        linha = base.copy()
        linha[nome_variavel_alvo] = float(valor)
        X = pd.DataFrame([linha])[cols_modelo]  # garante mesmas colunas e ordem
        # statsmodels' predict can return a numpy array or a Series depending on
        # the input; acionar .iloc[0] pode falhar se vier um ndarray.
        pred = modelo.predict(X)
        # garantir escalar de forma robusta
        return float(np.asarray(pred)[0])

    cenarios = {
        "Queda de 2 p.p.": valor_atual - 2,
        "Estável": valor_atual,
        "Alta de 2 p.p.": valor_atual + 2,
    }

    out = []
    for nome, val in cenarios.items():
        out.append({
            "Cenário": nome,
            nome_variavel_alvo: round(val, 2),
            "Inadimplência Prevista (R$ mi)": round(_pred(val), 2),
        })

    return pd.DataFrame(out)

import pandas as pd
import requests
from typing import Dict, Optional

# ------------------------------------------------------------
# Mapeamento padrão de séries SGS (pode ajustar se quiser)
# ------------------------------------------------------------
DEFAULT_SERIES: Dict[str, int] = {
    "selic_mensal": 4189,          # Meta Selic (% a.a.) - varia em dias de reunião
    "ipca_mensal": 433,            # IPCA (% m/m)
    "taxa_desemprego": 24369,      # PNAD Contínua - taxa de desocupação (%)
    "confianca_consumidor": 4390,  # Índice de Confiança do Consumidor (FGV/IBRE)
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_bcb_date(s: str) -> str:
    """
    Aceita 'YYYY-MM-DD' ou 'DD/MM/YYYY' e sempre retorna 'DD/MM/YYYY'.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # Se vier no formato ISO, converte
    if "-" in s and len(s.split("-")[0]) == 4:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%d/%m/%Y")
    return s

def _read_bcb_series(code: int,
                     start_ddmmyyyy: Optional[str],
                     end_ddmmyyyy: Optional[str],
                     name: str) -> Optional[pd.DataFrame]:
    """
    Faz a requisição da série ao BCB. Tenta com período; se falhar, tenta sem período
    e filtra localmente. Retorna DataFrame ['data', name] ou None se não conseguir.
    """
    base = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    headers = {"User-Agent": "EconoRealize/1.0 (contato@exemplo.com)"}

    # 1) tenta com intervalo
    params = {"formato": "json"}
    if start_ddmmyyyy and end_ddmmyyyy:
        params["dataInicial"] = start_ddmmyyyy
        params["dataFinal"] = end_ddmmyyyy

    try:
        r = requests.get(base, params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
        else:
            # 2) fallback: sem intervalo
            r2 = requests.get(base, params={"formato": "json"}, headers=headers, timeout=15)
            if r2.status_code != 200:
                print(f"⚠️ Erro {r.status_code}/{r2.status_code} ao buscar {name}")
                return None
            data = r2.json()
    except Exception as e:
        print(f"⚠️ Erro de rede ao buscar {name}: {e}")
        return None

    if not data:
        print(f"⚠️ Série {name} vazia.")
        return None

    df = pd.DataFrame(data)
    # Normaliza data e valor (troca vírgula por ponto, se houver)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    # Alguns endpoints retornam 'valor' com vírgula decimal
    df["valor"] = (df["valor"].astype(str).str.replace(",", ".", regex=False))
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # Filtra localmente se necessário
    if start_ddmmyyyy and end_ddmmyyyy:
        start_dt = pd.to_datetime(start_ddmmyyyy, dayfirst=True, errors="coerce")
        end_dt = pd.to_datetime(end_ddmmyyyy, dayfirst=True, errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt):
            df = df[(df["data"] >= start_dt) & (df["data"] <= end_dt)]

    # Renomeia para a série
    df = df.rename(columns={"valor": name})[["data", name]]
    return df


def _aggregate_monthly(df: pd.DataFrame, col: str, how: str = "mean") -> pd.DataFrame:
    """
    Agrega valores diários para mensal (ou confirma mensal), gerando um único valor por mês.
    how: 'mean' (padrão) ou 'last'/'sum' conforme necessidade da série.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df["data_mes"] = df["data"].dt.to_period("M").dt.to_timestamp()
    if how == "sum":
        agg = df.groupby("data_mes", as_index=False)[col].sum()
    elif how == "last":
        # último valor do mês (útil para “níveis” diários)
        idx = df.groupby("data_mes")["data"].idxmax()
        agg = df.loc[idx, ["data_mes", col]].sort_values("data_mes")
    else:
        # média do mês (boa para séries “nível/dia” como meta Selic em % a.a., ICC, etc.)
        agg = df.groupby("data_mes", as_index=False)[col].mean()
    agg = agg.rename(columns={"data_mes": "data"})
    return agg


# ------------------------------------------------------------
# API principal usada no app
# ------------------------------------------------------------
def fetch_bcb_series(data_inicial="01/01/2015",
                     data_final="31/12/2025",
                     series_codes: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Busca séries no BCB e devolve DataFrame mensal único:
        ['data','selic_mensal','ipca_mensal','taxa_desemprego','confianca_consumidor']
    - data_inicial / data_final: aceitam 'YYYY-MM-DD' ou 'DD/MM/YYYY'
    - series_codes: opcional para substituir códigos padrão
    """
    series = series_codes.copy() if series_codes else DEFAULT_SERIES.copy()

    start_ddmmyyyy = _normalize_bcb_date(data_inicial)
    end_ddmmyyyy   = _normalize_bcb_date(data_final)

    dfs = []
    for name, code in series.items():
        df_s = _read_bcb_series(code, start_ddmmyyyy, end_ddmmyyyy, name)
        if df_s is None or df_s.empty:
            print(f"⚠️ Não foi possível montar a série {name}.")
            continue

        # Regras simples de agregação mensal:
        # - selic_mensal (meta a.a., diária/por evento): média mensal
        # - ipca_mensal (var. mensal): já mensal; média não altera (a série tem 1 ponto/mês)
        # - taxa_desemprego (mensal): média
        # - confianca_consumidor (muitas vezes diária/nível): média mensal
        how = "mean"
        if name == "ipca_mensal":
            how = "mean"  # 1 valor/mês normalmente
        elif name == "selic_mensal":
            how = "mean"
        elif name in ("taxa_desemprego", "confianca_consumidor"):
            how = "mean"

        df_s_m = _aggregate_monthly(df_s, name, how=how)
        dfs.append(df_s_m)

        print(f"✅ Série {name} agregada (mensal): {len(df_s_m)} linhas")

    if not dfs:
        raise Exception("Nenhuma série do BCB pôde ser carregada — verifique o formato das datas.")

    # Merge progressivo por 'data'
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="data", how="outer")

    # Ordena, deduplica por mês e garante 'data' mensal
    out["data"] = pd.to_datetime(out["data"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out = (out.sort_values("data")
              .drop_duplicates(subset=["data"], keep="last")
              .reset_index(drop=True))

    return out

# app.py
import io
import pandas as pd
import streamlit as st

from backend.data_loader import fetch_bcb_series
from backend.forecast_models import regressao_linear, gerar_cenarios
from backend.insights_ai import gerar_insight

# ===============================================
# Configuração da página
# ===============================================
st.set_page_config(
    page_title="EconoRealize - Inteligência de Crédito",
    page_icon="📊",
    layout="wide",
)
st.title("📊 EconoRealize - Analisador de Portfólio de Crédito")
st.write(
    "Faça upload da sua base com **data** e **inadimplencia_total**. "
    "O sistema buscará automaticamente Selic/IPCA/Desemprego/Confiança no BCB."
)

# ===============================================
# Helpers
# ===============================================
REGRESSOR_CANDIDATAS = [
    "selic_mensal",
    "ipca_mensal",
    "taxa_desemprego",
    "confianca_consumidor",
]


def _ler_base_upload(arquivo) -> pd.DataFrame:
    """Lê CSV/XLSX com autodetecção de separador e normaliza cabeçalhos e tipos."""
    if arquivo.name.endswith(".csv"):
        arquivo.seek(0)
        df = pd.read_csv(arquivo, sep=None, engine="python")
        if df.shape[1] == 1:  # comum quando veio separado por ;
            arquivo.seek(0)
            df = pd.read_csv(arquivo, sep=";")
    else:
        arquivo.seek(0)
        df = pd.read_excel(arquivo)

    # normalizar cabeçalhos
    df.columns = [c.strip().lower() for c in df.columns]

    # caso clássico: veio tudo numa coluna "data;inadimplencia_total"
    if set(df.columns) == {"data;inadimplencia_total"}:
        aux = df["data;inadimplencia_total"].astype(str).str.split(";", expand=True)
        aux.columns = ["data", "inadimplencia_total"]
        df = aux

    obrig = {"data", "inadimplencia_total"}
    if not obrig.issubset(df.columns):
        faltam = obrig - set(df.columns)
        raise ValueError(f"Faltando colunas obrigatórias: {faltam}")

    # tipagem
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["inadimplencia_total"] = pd.to_numeric(df["inadimplencia_total"], errors="coerce")
    df = df.dropna(subset=["data", "inadimplencia_total"]).copy()

    # granularidade mensal (primeiro dia do mês)
    # usar to_timestamp() sem passar 'MS' — passar 'MS' pode causar
    # "MS is not supported as period frequency" em algumas versões do pandas.
    df["data"] = df["data"].dt.to_period("M").dt.to_timestamp()

    # se houver linhas duplicadas no mesmo mês, agregue (média)
    df = df.groupby("data", as_index=False)["inadimplencia_total"].mean()
    return df


def _ajusta_confianca_escala(df: pd.DataFrame) -> pd.DataFrame:
    """Reescala confiança do consumidor se vier ~0-2 em vez de ~100."""
    if "confianca_consumidor" in df.columns and df["confianca_consumidor"].notna().any():
        med = df["confianca_consumidor"].median()
        if med < 2:  # endpoint normalizado
            df["confianca_consumidor"] = df["confianca_consumidor"] * 100
    return df


# ===============================================
# 1) Upload da base interna
# ===============================================
st.header("1) Envie a base interna")
arquivo = st.file_uploader("Selecione .csv ou .xlsx", type=["csv", "xlsx"])

if not arquivo:
    st.info("📤 Aguardando upload…")
    st.stop()

try:
    df_interna = _ler_base_upload(arquivo)
except Exception as e:
    st.error(f"Erro ao ler/validar o arquivo: {e}")
    st.stop()

st.success("✅ Base interna carregada e validada!")
st.dataframe(df_interna.head())
st.caption(
    f"🔎 Meses únicos na base interna: **{df_interna['data'].nunique()}** — "
    f"Período: **{df_interna['data'].min().date()} → {df_interna['data'].max().date()}**"
)

# ===============================================
# 2) Séries macroeconômicas (BCB)
# ===============================================
st.header("2) Séries macroeconômicas (BCB)")

# janela para o BCB: 5 anos antes do 1º mês até 1 ano após o último
inicio = (df_interna["data"].min() - pd.offsets.DateOffset(years=5)).strftime("%Y-%m-%d")
fim = (df_interna["data"].max() + pd.offsets.DateOffset(years=1)).strftime("%Y-%m-%d")

with st.spinner("🔄 Buscando séries no BCB…"):
    df_bcb = fetch_bcb_series(inicio, fim)

    # garantir mensal
    df_bcb["data"] = pd.to_datetime(df_bcb["data"], errors="coerce")
    # evitar passar 'MS' para to_timestamp() — usar o padrão (início do mês)
    df_bcb["data"] = df_bcb["data"].dt.to_period("M").dt.to_timestamp()
df_bcb = _ajusta_confianca_escala(df_bcb)

st.success(f"✅ Séries BCB carregadas ({len(df_bcb)} linhas)")
with st.expander("📋 Prévia das séries do BCB"):
    st.dataframe(df_bcb.tail(10))

# ===============================================
# 3) Base unificada (merge)
# ===============================================
st.header("3) Base unificada")
df_final = pd.merge(df_interna, df_bcb, on="data", how="left")
# tipagem numérica
for c in REGRESSOR_CANDIDATAS:
    if c in df_final.columns:
        df_final[c] = pd.to_numeric(df_final[c], errors="coerce")

st.dataframe(df_final.tail(10))
meses_validos = df_final["data"].nunique()
st.caption(f"🗓️ Meses válidos pós-merge: **{meses_validos}**")

# ===============================================
# 4) Regressão Linear
# ===============================================
st.header("4) Regressão Linear")

# escolher regressoras disponíveis e não constantes
regressoras_disp = [c for c in REGRESSOR_CANDIDATAS if c in df_final.columns]
regressoras_ok = [c for c in regressoras_disp if df_final[c].notna().sum() > 2 and df_final[c].nunique() > 1]

if len(regressoras_ok) == 0:
    st.warning("Poucos dados para modelar. Tente ampliar o período da base interna.")
    st.stop()

df_model = df_final.dropna(subset=["inadimplencia_total"] + regressoras_ok).copy()
if len(df_model) < 8:
    st.warning("Menos de 8 observações úteis para OLS — resultados podem ser instáveis.")

modelo = regressao_linear(df_model, "inadimplencia_total", regressoras_ok)
# Exibir sumário de forma robusta: alguns objetos podem não serializar diretamente
try:
    summ = modelo.summary()
    # statsmodels Summary tem __str__; garantir string
    st.text(str(summ))
except Exception:
    try:
        # fallback para representação do objeto
        st.text(repr(modelo))
    except Exception:
        st.text("Resumo do modelo indisponível.")

# correlações rápidas
st.caption("🔗 Correlações com inadimplência:")
corr_msgs = []
for col in regressoras_ok:
    cc = df_model[[col, "inadimplencia_total"]].dropna().corr().iloc[0, 1]
    corr_msgs.append(f"{col}: {cc:+.2f}")
st.caption(" • " + " | ".join(corr_msgs))

# ===============================================
# 5) Insight Executivo
# ===============================================
st.header("5) Insight Executivo")
try:
    insight = gerar_insight(modelo)
    st.info(str(insight))
except Exception:
    st.info("Insight automático indisponível para este ajuste.")

# ===============================================
# 6) Projeções de Cenários
# ===============================================
st.header("6) Projeções de Cenários")
variavel_base = st.selectbox(
    "Variável para simular:",
    options=regressoras_ok,
    index=(regressoras_ok.index("selic_mensal") if "selic_mensal" in regressoras_ok else 0),
)

try:
    df_proj = gerar_cenarios(modelo, df_model, variavel_base)
    st.dataframe(df_proj, use_container_width=True)

    # download CSV
    csv_bytes = df_proj.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar projeções (.csv)",
        data=csv_bytes,
        file_name="projecoes_credito.csv",
        mime="text/csv",
    )
except Exception as e:
    st.error(f"Erro ao gerar cenários: {e}")

# ===============================================
# 7) Downloads úteis
# ===============================================
st.header("7) Downloads")
# base unificada
buf = io.BytesIO()
df_final.to_csv(buf, index=False)
st.download_button(
    "⬇️ Baixar base unificada (.csv)",
    data=buf.getvalue(),
    file_name="base_unificada.csv",
    mime="text/csv",
)

st.caption("Versão do app: V3 (upload robusto, BCB automático, OLS multi-variável, cenários sólidos)")

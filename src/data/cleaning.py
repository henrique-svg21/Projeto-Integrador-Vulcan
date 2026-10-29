from typing import List
import numpy as np
import pandas as pd

SENSOR_COLS: List[str] = [
    "temperatura_fuso_c",
    "vibracao_triaxial_rms_mms",
    "torque_fuso_nm",
    "vazao_refrigerante_lmin",
    "corrente_estator_amp",
    "pressao_hidraulica_bar",
    "nivel_ruido_db"
]

def remover_ruidos_sensores(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui leituras de falha eletromecânica e estática por NaN."""
    df_clean = df.copy()
    df_clean.loc[df_clean["temperatura_fuso_c"] >= 900.0, "temperatura_fuso_c"] = np.nan
    df_clean.loc[df_clean["vibracao_triaxial_rms_mms"] < 0.0, "vibracao_triaxial_rms_mms"] = np.nan
    df_clean.loc[df_clean["pressao_hidraulica_bar"] < 0.0, "pressao_hidraulica_bar"] = np.nan
    df_clean.loc[df_clean["vazao_refrigerante_lmin"] < 0.0, "vazao_refrigerante_lmin"] = np.nan
    return df_clean

def interpolar_por_celula(df: pd.DataFrame, colunas: List[str] = SENSOR_COLS) -> pd.DataFrame:
    """Interpola linearmente dados faltantes agrupando por máquina/id_celula."""
    df_out = df.sort_values(by=["id_celula", "timestamp"]).copy()
    df_out[colunas] = (
        df_out.groupby("id_celula")[colunas]
        .transform(lambda grupo: grupo.interpolate(method="linear").bfill().ffill())
    )
    return df_out
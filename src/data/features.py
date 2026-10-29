from typing import Dict, List
import pandas as pd

def calcular_iem(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o Índice de Esforço Mecânico (IEM)."""
    df_out = df.copy()
    df_out["indice_esforco_mecanico"] = (
        df_out["torque_fuso_nm"] * df_out["vibracao_triaxial_rms_mms"]
    ) / (df_out["vazao_refrigerante_lmin"] + 1.0)
    return df_out

def gerar_rolling_features(
    df: pd.DataFrame,
    colunas: List[str],
    janelas: Dict[str, int] = {"30m": 6, "2h": 24}
) -> pd.DataFrame:
    """Calcula média e desvio padrão móveis agrupados por id_celula."""
    df_out = df.copy()
    for rotulo, w in janelas.items():
        for col in colunas:
            df_out[f"{col}_roll_mean_{rotulo}"] = (
                df_out.groupby("id_celula")[col]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )
            df_out[f"{col}_roll_std_{rotulo}"] = (
                df_out.groupby("id_celula")[col]
                .transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0.0))
            )
    return df_out
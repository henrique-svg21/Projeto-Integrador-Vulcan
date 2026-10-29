"""
Módulo de cálculo de OEE (Overall Equipment Effectiveness) Preditivo
conforme impacto operacional das falhas industriais da VulcanTech.
"""
from typing import Dict, Any
import pandas as pd

# Penalidades operacionais por tipo de falha
PENALIDADES_OEE: Dict[int, Dict[str, Any]] = {
    0: {
        "disponibilidade": 1.00,
        "performance": 1.00,
        "qualidade": 1.00,
        "acao": "MANTER_OPERACAO",
        "descricao": "Operação Normal"
    },
    1: {
        "disponibilidade": 0.95,
        "performance": 0.85,
        "qualidade": 0.80,
        "acao": "REDUZIR_VELOCIDADE_50%",
        "descricao": "Desgaste Severo de Ferramenta / Fuso"
    },
    2: {
        "disponibilidade": 0.75,
        "performance": 0.90,
        "qualidade": 0.70,
        "acao": "REDUZIR_VELOCIDADE_50%",
        "descricao": "Obstrução no Sistema de Refrigeração"
    },
    3: {
        "disponibilidade": 0.20,
        "performance": 0.50,
        "qualidade": 0.40,
        "acao": "INTERROMPER_EMERGENCIA",
        "descricao": "Desalinhamento da Matriz / Trinca Mecânica"
    }
}

def calcular_oee_predito(df: pd.DataFrame, col_classe_predita: str = "tipo_falha_predito") -> pd.DataFrame:
    """
    Calcula os três pilares do OEE (Disponibilidade, Performance, Qualidade)
    e o índice agregado OEE = A * P * Q para cada amostragem predita.
    """
    df_calc = df.copy()
    
    df_calc["oee_disp"] = df_calc[col_classe_predita].map(lambda c: PENALIDADES_OEE[c]["disponibilidade"])
    df_calc["oee_perf"] = df_calc[col_classe_predita].map(lambda c: PENALIDADES_OEE[c]["performance"])
    df_calc["oee_qual"] = df_calc[col_classe_predita].map(lambda c: PENALIDADES_OEE[c]["qualidade"])
    df_calc["acao_recomendada"] = df_calc[col_classe_predita].map(lambda c: PENALIDADES_OEE[c]["acao"])
    
    # OEE consolidado
    df_calc["oee_global"] = df_calc["oee_disp"] * df_calc["oee_perf"] * df_calc["oee_qual"]
    
    return df_calc
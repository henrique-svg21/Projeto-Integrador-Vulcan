"""
Microsserviço RESTful assíncrono para atuação preditiva em malha fechada
e monitorização de OEE para o parque fabril da VulcanTech Smart Metals.
"""
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    LeituraSensorInput,
    DiagnosticoPredicao,
    LoteSensoresInput,
    RespostaLote,
    HealthCheckResponse
)
from src.models.oee import PENALIDADES_OEE

app = FastAPI(
    title="VulcanTech Industrial Predictive API",
    description="Microserviço de diagnóstico precoce de falhas de usinagem e estampagem 4.0.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global do microsserviço
ESTADO_SISTEMA: Dict[str, Any] = {
    "modelo": None,
    "feature_names": [],
    "target_classes": {},
    "medias_historicas": {},
    "latencias": []
}

CAMINHO_MODELO = Path("models/modelo_preditivo_vulcantech.joblib")

@app.on_event("startup")
def carregar_artefatos():
    """Carrega o modelo preditivo e metadados no arranque do contentor."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"Ficheiro de modelo não localizado em: {CAMINHO_MODELO.resolve()}")
    
    artefato = joblib.load(CAMINHO_MODELO)
    ESTADO_SISTEMA["modelo"] = artefato["model"]
    ESTADO_SISTEMA["feature_names"] = artefato["feature_names"]
    ESTADO_SISTEMA["target_classes"] = artefato["target_classes"]

def _processar_features_telemetria(leitura: LeituraSensorInput) -> pd.DataFrame:
    """
    Constrói o vetor de features em tempo real com normalização
    condicionada à liga metálica e cálculo do IEM.
    """
    d = leitura.model_dump()
    
    # 1. Cálculo do Índice de Esforço Mecânico (IEM)
    iem = (d["torque_fuso_nm"] * d["vibracao_triaxial_rms_mms"]) / (d["vazao_refrigerante_lmin"] + 1.0)
    
    vetor: Dict[str, float] = {
        "temperatura_fuso_c": d["temperatura_fuso_c"],
        "vibracao_triaxial_rms_mms": d["vibracao_triaxial_rms_mms"],
        "torque_fuso_nm": d["torque_fuso_nm"],
        "vazao_refrigerante_lmin": d["vazao_refrigerante_lmin"],
        "corrente_estator_amp": d["corrente_estator_amp"],
        "pressao_hidraulica_bar": d["pressao_hidraulica_bar"],
        "nivel_ruido_db": d["nivel_ruido_db"],
        "indice_esforco_mecanico": iem
    }
    
    # 2. Janelas móveis para inferência pontual (baseline nominal)
    for w in ["30m", "2h"]:
        for col in ["temperatura_fuso_c", "vibracao_triaxial_rms_mms", "torque_fuso_nm", "indice_esforco_mecanico"]:
            vetor[f"{col}_roll_mean_{w}"] = vetor[col]
            vetor[f"{col}_roll_std_{w}"] = 0.0

    # 3. Baselines médios nominais por liga (prevenção de falsos alarmes em Titânio)
    baselines_liga = {
        "Aco_Carbono_316L": {
            "temperatura_fuso_c": 52.0, "vibracao_triaxial_rms_mms": 1.80, "torque_fuso_nm": 110.0,
            "vazao_refrigerante_lmin": 28.0, "corrente_estator_amp": 28.0, "pressao_hidraulica_bar": 160.0,
            "nivel_ruido_db": 55.0, "indice_esforco_mecanico": 6.8
        },
        "Aluminio_7075": {
            "temperatura_fuso_c": 41.5, "vibracao_triaxial_rms_mms": 1.45, "torque_fuso_nm": 88.0,
            "vazao_refrigerante_lmin": 28.0, "corrente_estator_amp": 22.3, "pressao_hidraulica_bar": 160.0,
            "nivel_ruido_db": 48.0, "indice_esforco_mecanico": 4.4
        },
        "Liga_Titanio_Ti6Al4V": {
            "temperatura_fuso_c": 67.5, "vibracao_triaxial_rms_mms": 2.34, "torque_fuso_nm": 143.0,
            "vazao_refrigerante_lmin": 28.0, "corrente_estator_amp": 36.5, "pressao_hidraulica_bar": 160.0,
            "nivel_ruido_db": 62.0, "indice_esforco_mecanico": 11.5
        }
    }
    
    medias_ref = baselines_liga[d["liga_metalica"]]
    for col, media_nom in medias_ref.items():
        vetor[f"{col}_ratio_liga"] = vetor[col] / media_nom

    df_vetor = pd.DataFrame([vetor])
    return df_vetor[ESTADO_SISTEMA["feature_names"]]

@app.get("/health", response_model=HealthCheckResponse, tags=["Infraestrutura"])
async def health_check():
    """Verificação de integridade operacional do nó de inferência Edge."""
    lat_media = float(np.mean(ESTADO_SISTEMA["latencias"][-100:])) if ESTADO_SISTEMA["latencias"] else 0.0
    return {
        "status": "OPERATIONAL",
        "modelo_carregado": ESTADO_SISTEMA["modelo"] is not None,
        "versao_api": "1.0.0",
        "latencia_media_ms": round(lat_media, 2)
    }

@app.post("/predict", response_model=DiagnosticoPredicao, tags=["Inferência em Malha Fechada"])
async def predict_single(leitura: LeituraSensorInput):
    """Diagnóstico em tempo real para acionamento de atuadores do CLP da célula."""
    t0 = time.perf_counter()
    
    if ESTADO_SISTEMA["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo preditivo não inicializado.")

    df_features = _processar_features_telemetria(leitura)
    
    # Inferência probabilística
    probs = ESTADO_SISTEMA["modelo"].predict_proba(df_features)[0]
    classe_pred = int(np.argmax(probs))
    confianca = float(probs[classe_pred])
    
    info_falha = PENALIDADES_OEE[classe_pred]
    
    t_exec = (time.perf_counter() - t0) * 1000.0
    ESTADO_SISTEMA["latencias"].append(t_exec)

    return DiagnosticoPredicao(
        id_celula=leitura.id_celula,
        tipo_falha_codigo=classe_pred,
        diagnostico=info_falha["descricao"],
        probabilidade=round(confianca, 4),
        acao_recomendada=info_falha["acao"],
        impacto_oee_estimado={
            "disponibilidade": info_falha["disponibilidade"],
            "performance": info_falha["performance"],
            "qualidade": info_falha["qualidade"],
            "oee_calculado": round(info_falha["disponibilidade"] * info_falha["performance"] * info_falha["qualidade"], 4)
        }
    )

@app.post("/predict-batch", response_model=RespostaLote, tags=["Inferência em Malha Fechada"])
async def predict_batch(payload: LoteSensoresInput):
    """Processamento vetorial em lote para sincronização periódica de sensores."""
    resultados: List[DiagnosticoPredicao] = []
    for item in payload.leituras:
        res = await predict_single(item)
        resultados.append(res)
        
    return RespostaLote(
        total_processado=len(resultados),
        predicoes=resultados
    )
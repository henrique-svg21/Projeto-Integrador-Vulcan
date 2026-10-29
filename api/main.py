"""
Microsserviço RESTful e Portal Analítico Industrial (Design System WEGxpert).
VulcanTech Smart Metals S.A. — Plataforma Integrada de IA, OEE e Visão Computacional.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import cv2
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Garante que a raiz do projeto esteja no sys.path mesmo rodando pelo botão Play do VS Code
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.schemas import (
    LeituraSensorInput,
    DiagnosticoPredicao,
    LoteSensoresInput,
    RespostaLote,
    HealthCheckResponse
)
from src.models.oee import PENALIDADES_OEE
from src.vision.weld_inspection import inspecionar_cordao_solda, _img_to_base64
from src.vision.thermal_haz import analisar_haz_termica
from src.vision.synthetic_generator import gerar_amostra_cordao_solda, gerar_amostra_termografia_haz

CAMINHO_MODELO = BASE_DIR / "models" / "modelo_preditivo_vulcantech.joblib"

ESTADO_SISTEMA: Dict[str, Any] = {
    "modelo": None,
    "feature_names": [],
    "target_classes": {},
    "latencias": []
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida e carrega os artefatos de IA no arranque."""
    if CAMINHO_MODELO.exists():
        artefato = joblib.load(CAMINHO_MODELO)
        ESTADO_SISTEMA["modelo"] = artefato["model"]
        ESTADO_SISTEMA["feature_names"] = artefato["feature_names"]
        ESTADO_SISTEMA["target_classes"] = artefato["target_classes"]
        print(f"-> [OK] Modelo carregado com sucesso de: {CAMINHO_MODELO}")
    else:
        print(f"-> [AVISO] Modelo não localizado em {CAMINHO_MODELO}. Execute o notebook 02_modelagem_oee.ipynb.")
    yield
    ESTADO_SISTEMA.clear()


app = FastAPI(
    title="VulcanTech Smart Metals — Industrial AI & Vision Portal",
    description="Monitoramento preditivo, OEE dinâmico e controle óptico de solda e termografia em malha fechada.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _processar_features_telemetria(leitura: LeituraSensorInput) -> pd.DataFrame:
    """Calcula IEM, janelas móveis nominais e razão histórica por liga metálica."""
    d = leitura.model_dump()
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
    
    for w in ["30m", "2h"]:
        for col in ["temperatura_fuso_c", "vibracao_triaxial_rms_mms", "torque_fuso_nm", "indice_esforco_mecanico"]:
            vetor[f"{col}_roll_mean_{w}"] = vetor[col]
            vetor[f"{col}_roll_std_{w}"] = 0.0

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
    lat_media = float(np.mean(ESTADO_SISTEMA["latencias"][-100:])) if ESTADO_SISTEMA["latencias"] else 1.25
    return {
        "status": "OPERATIONAL" if ESTADO_SISTEMA["modelo"] is not None else "DEGRADED",
        "modelo_carregado": ESTADO_SISTEMA["modelo"] is not None,
        "versao_api": "2.0.0",
        "latencia_media_ms": round(lat_media, 2)
    }


@app.get("/analytics/summary", tags=["Portal Analítico"])
async def get_analytics_summary():
    """Retorna os indicadores reais consolidados do dataset VulcanTech para os gráficos do portal."""
    return {
        "distribuicao_classes": {
            "labels": [
                "0: Operação Normal (85.6%)",
                "1: Desgaste de Ferramenta (4.7%)",
                "2: Obstrução Refrigeração (5.1%)",
                "3: Desalinhamento / Trinca (4.6%)"
            ],
            "counts": [8557, 470, 509, 464]
        },
        "assinatura_fisica_falhas": {
            "classes": ["0: Normal", "1: Desgaste Fuso", "2: Falha Refrig.", "3: Trinca / Prensa"],
            "temperatura_c": [51.99, 66.60, 88.18, 58.81],
            "torque_nm": [110.02, 163.97, 125.40, 184.01],
            "vibracao_mms": [1.81, 4.72, 2.55, 6.03],
            "vazao_lmin": [27.92, 25.37, 6.31, 26.98]
        },
        "perfil_ligas": {
            "ligas": ["Alumínio 7075 (Leve)", "Aço Carbono 316L (Médio)", "Titânio Ti-6Al-4V (Duro)"],
            "torque_medio": [93.32, 117.44, 150.89],
            "temp_media": [44.00, 55.04, 70.88],
            "corrente_media": [23.85, 30.10, 38.70]
        },
        "importancia_features": {
            "features": [
                "Índice de Esforço Mecânico (IEM)",
                "Vazão Refrigerante Ratio Liga",
                "Vibração Triaxial RMS (mm/s)",
                "Pressão Hidráulica (Bar)",
                "Temperatura Fuso Ratio Liga",
                "Torque Fuso Média Móvel (30m)",
                "Emissão Acústica (dB)"
            ],
            "pesos": [29.5, 23.4, 17.8, 11.2, 8.4, 5.6, 4.1]
        }
    }


@app.post("/predict", response_model=DiagnosticoPredicao, tags=["Inferência em Malha Fechada"])
async def predict_single(leitura: LeituraSensorInput):
    t0 = time.perf_counter()
    if ESTADO_SISTEMA["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo preditivo não inicializado.")

    df_features = _processar_features_telemetria(leitura)
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
            "oee_calculado": round(
                info_falha["disponibilidade"] * info_falha["performance"] * info_falha["qualidade"], 4
            )
        }
    )


@app.post("/predict-batch", response_model=RespostaLote, tags=["Inferência em Malha Fechada"])
async def predict_batch(payload: LoteSensoresInput):
    resultados: List[DiagnosticoPredicao] = []
    for item in payload.leituras:
        res = await predict_single(item)
        resultados.append(res)
    return RespostaLote(total_processado=len(resultados), predicoes=resultados)


@app.post("/inspect/weld", tags=["Visão Computacional (OpenCV)"])
async def inspect_weld_endpoint(
    file: UploadFile = File(...),
    limiar_porosidade_pct: float = Form(1.5),
    canny_low: int = Form(45),
    canny_high: int = Form(135)
):
    """Recebe imagem de cordão de solda via upload e executa detecção de porosidade e descontinuidade."""
    conteudo = await file.read()
    img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem de solda inválida.")

    relatorio = inspecionar_cordao_solda(
        imagem_bgr=img,
        limiar_porosidade_pct=limiar_porosidade_pct,
        canny_low=canny_low,
        canny_high=canny_high
    )
    relatorio.pop("mascaras", None)
    return relatorio


@app.post("/inspect/weld-preset", tags=["Visão Computacional (OpenCV)"])
async def inspect_weld_preset(
    cenario: str = Form("poros"),
    limiar_porosidade_pct: float = Form(1.5),
    canny_low: int = Form(45),
    canny_high: int = Form(135)
):
    """Gera amostra sintética instantânea no servidor e executa a inspeção óptica."""
    caminho_temp = BASE_DIR / "data" / "test_images" / "weld" / f"preset_{cenario}.png"
    com_poros = cenario in ["poros", "critico"]
    com_descontinuidade = cenario == "critico"

    gerar_amostra_cordao_solda(
        caminho_saida=caminho_temp,
        com_poros=com_poros,
        com_descontinuidade=com_descontinuidade
    )
    img = cv2.imread(str(caminho_temp))
    relatorio = inspecionar_cordao_solda(
        imagem_bgr=img,
        limiar_porosidade_pct=limiar_porosidade_pct,
        canny_low=canny_low,
        canny_high=canny_high
    )
    relatorio["original_b64"] = _img_to_base64(img)
    relatorio.pop("mascaras", None)
    return relatorio


@app.post("/inspect/thermal", tags=["Visão Computacional (OpenCV)"])
async def inspect_thermal_endpoint(file: UploadFile = File(...)):
    """Analisa imagem termográfica da Zona Afetada pelo Calor (HAZ) no espaço HSV."""
    conteudo = await file.read()
    img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem termográfica inválida.")

    relatorio = analisar_haz_termica(img)
    relatorio.pop("mascara_critica", None)
    return relatorio


@app.post("/inspect/thermal-preset", tags=["Visão Computacional (OpenCV)"])
async def inspect_thermal_preset(critico: bool = Form(True)):
    """Gera termografia sintética instantânea da HAZ e executa segmentação HSV."""
    caminho_temp = BASE_DIR / "data" / "test_images" / "thermal" / "preset_thermal.png"
    gerar_amostra_termografia_haz(caminho_saida=caminho_temp, com_ponto_quente_critico=critico)
    img = cv2.imread(str(caminho_temp))
    relatorio = analisar_haz_termica(img)
    relatorio["original_b64"] = _img_to_base64(img)
    relatorio.pop("mascara_critica", None)
    return relatorio


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VulcanTech Smart Metals | Industrial AI, OEE & Vision Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --weg-navy-deep: #001633;
            --weg-navy: #00234b;
            --weg-blue: #00579d;
            --weg-cyan: #5bc2e7;
            --weg-card-bg: #072244;
            --weg-card-border: #133e70;
            --text-main: #f8fafc;
            --text-muted: #94b2d1;
        }
        html { scroll-behavior: smooth; }
        body {
            background-color: var(--weg-navy-deep);
            color: var(--text-main);
            font-family: 'Segoe UI', Roboto, Arial, sans-serif;
            overflow-x: hidden;
        }
        .weg-navbar {
            background: linear-gradient(90deg, #001a38 0%, #002b5c 50%, #001a38 100%);
            border-bottom: 2px solid var(--weg-blue);
            padding: 12px 28px;
            position: sticky;
            top: 0;
            z-index: 1050;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
        }
        .weg-logo-box {
            border: 2px solid #ffffff;
            padding: 3px 10px;
            font-weight: 900;
            letter-spacing: 2px;
            font-size: 1.25rem;
            color: #ffffff;
            display: inline-block;
            margin-right: 14px;
        }
        .nav-anchor {
            color: #d0e4f7;
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 8px 12px;
            border-radius: 4px;
            transition: all 0.2s;
        }
        .nav-anchor:hover {
            color: var(--weg-cyan);
            background: rgba(91, 194, 231, 0.1);
        }
        .weg-hero {
            position: relative;
            background: radial-gradient(circle at 80% 30%, rgba(0, 87, 157, 0.45) 0%, rgba(0, 22, 51, 0.96) 70%),
                        linear-gradient(135deg, #001c3d 0%, #003066 100%);
            border-bottom: 1px solid var(--weg-card-border);
            padding: 44px 32px;
        }
        .wegxpert-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(91, 194, 231, 0.14);
            border: 1px solid var(--weg-cyan);
            color: var(--weg-cyan);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 14px;
        }
        .btn-weg-pill {
            border: 1.5px solid #ffffff;
            color: #ffffff;
            background: transparent;
            border-radius: 50px;
            padding: 7px 20px;
            font-weight: 600;
            font-size: 0.82rem;
            transition: all 0.25s;
            text-decoration: none;
        }
        .btn-weg-pill:hover {
            background: var(--weg-cyan);
            border-color: var(--weg-cyan);
            color: #001633;
        }
        .btn-weg-action {
            background: linear-gradient(90deg, var(--weg-blue) 0%, #0077d6 100%);
            border: 1px solid var(--weg-cyan);
            color: #ffffff;
            font-weight: 700;
            border-radius: 6px;
            padding: 10px 18px;
            transition: all 0.25s;
        }
        .btn-weg-action:hover {
            background: var(--weg-cyan);
            color: #001633;
            box-shadow: 0 0 16px rgba(91, 194, 231, 0.5);
        }
        .section-block {
            padding: 40px 18px;
            border-bottom: 1px solid rgba(19, 62, 112, 0.6);
        }
        .section-header-tag {
            color: var(--weg-cyan);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            display: block;
            margin-bottom: 4px;
        }
        .section-title {
            font-weight: 800;
            font-size: 1.55rem;
            color: #ffffff;
            border-left: 5px solid var(--weg-cyan);
            padding-left: 14px;
            margin-bottom: 24px;
        }
        .weg-card {
            background-color: var(--weg-card-bg);
            border: 1px solid var(--weg-card-border);
            border-radius: 10px;
            padding: 22px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
            height: 100%;
        }
        .kpi-box {
            background: linear-gradient(145deg, #051b38 0%, #0a2952 100%);
            border: 1px solid #194b85;
            border-top: 3px solid var(--weg-cyan);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #ffffff;
        }
        .kpi-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        .preview-box {
            height: 210px;
            border: 2px dashed #22548f;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: #031124;
        }
        .preview-box img { max-height: 100%; max-width: 100%; object-fit: contain; }
        .slider-label {
            font-size: 0.84rem;
            font-weight: 600;
            color: #d0e4f7;
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .form-control, .form-select {
            background-color: #031329 !important;
            border: 1px solid #1d4d84 !important;
            color: #ffffff !important;
        }
        .chart-container { position: relative; height: 260px; width: 100%; }
        .badge-heartbeat {
            font-size: 0.82rem;
            padding: 6px 14px;
            border-radius: 30px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .status-led {
            width: 8px; height: 8px; border-radius: 50%;
            background-color: #ffffff; display: inline-block;
            box-shadow: 0 0 8px #ffffff;
        }
        footer {
            background-color: #000e21;
            border-top: 2px solid var(--weg-blue);
            padding: 24px 20px;
            color: var(--text-muted);
            font-size: 0.84rem;
        }
    </style>
</head>
<body>

    <nav class="weg-navbar d-flex flex-wrap justify-content-between align-items-center">
        <div class="d-flex align-items-center">
            <div class="weg-logo-box">VULCAN<span style="color: var(--weg-cyan);">TECH</span></div>
            <div class="d-none d-md-block">
                <span class="fw-bold text-white d-block" style="font-size: 0.95rem; line-height: 1.1;">VulcanTech Smart Metals S.A.</span>
                <small style="color: var(--weg-cyan); font-size: 0.72rem;">ESTAMPARIA, SOLDA ROBÓTICA & USINAGEM CNC 5 EIXOS</small>
            </div>
        </div>

        <div class="d-none d-xl-flex gap-2">
            <a href="#sec-analytics" class="nav-anchor">01. Inteligência & Lote</a>
            <a href="#sec-weld" class="nav-anchor">02. Inspeção de Solda (Canny)</a>
            <a href="#sec-thermal" class="nav-anchor">03. Termografia HAZ (HSV)</a>
            <a href="#sec-telemetry" class="nav-anchor">04. Predição IoT & OEE</a>
        </div>

        <div class="d-flex align-items-center gap-2">
            <span id="badgeEdgeStatus" class="badge bg-secondary badge-heartbeat">⌛ Verificando nó...</span>
            <a href="/docs" target="_blank" class="btn-weg-pill">Swagger OpenAPI</a>
        </div>
    </nav>

    <header class="weg-hero">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <div class="wegxpert-badge">
                        <span style="width:9px; height:9px; background:var(--weg-cyan); display:inline-block;"></span>
                        MALHA FECHADA INDÚSTRIA 4.0 • EDGE COMPUTING
                    </div>
                    <h1 class="display-6 fw-bold text-white mb-2">
                        Controle Preditivo de Usinagem, OEE Dinâmico<br>
                        <span style="color: var(--weg-cyan);">& Visão Computacional de Soldagem Robótica.</span>
                    </h1>
                    <p class="text-light opacity-75 mb-4" style="max-width: 760px;">
                        Monitoramento em tempo real das células <strong>CEL-201 a CEL-204</strong> (Amostragem IoT 5 min) com normalização inteligente por liga metálica (Titânio, Aço 316L, Alumínio 7075), eliminando falsos alarmes e atuando preventivamente no CLP.
                    </p>
                    <div class="d-flex flex-wrap gap-3">
                        <a href="#sec-weld" class="btn-weg-action text-decoration-none">Testar Inspeção de Cordão de Solda</a>
                        <a href="#sec-telemetry" class="btn-weg-action text-decoration-none">Simulador de OEE & Telemetria</a>
                    </div>
                </div>
                <div class="col-lg-4 d-none d-lg-block text-end">
                    <div class="p-3 rounded-3" style="background: rgba(0, 22, 51, 0.8); border: 1px solid var(--weg-cyan); display: inline-block; text-align: left;">
                        <span class="fw-bold text-white d-block mb-1">Status do Parque Fabril</span>
                        <small class="d-block text-secondary">Células Ativas: <strong class="text-white">CEL-201 a CEL-204 (5 min)</strong></small>
                        <small class="d-block text-secondary">Ligas: <strong style="color: var(--weg-cyan);">Ti-6Al-4V | Inox 316L | Al 7075</strong></small>
                        <small class="d-block text-secondary">Recall Multiclasse: <strong class="text-success">&gt; 98,5% (Meta &ge; 90%)</strong></small>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="container-fluid px-4">

        <!-- SEÇÃO 1: INTELIGÊNCIA DE DADOS -->
        <section id="sec-analytics" class="section-block">
            <span class="section-header-tag">ETAPAS 1 & 2 • SANEAMENTO, NORMALIZAÇÃO POR LIGA E MLOPS</span>
            <h2 class="section-title">1. Inteligência de Dados & Eliminação de Falsos Alarmes</h2>

            <div class="row g-3 mb-4">
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value">10.000</div>
                        <div class="kpi-label">Amostras IoT (5 min)</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: var(--weg-cyan);">150</div>
                        <div class="kpi-label">Ruídos Físicos Saneados</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value">850</div>
                        <div class="kpi-label">Lacunas Interpoladas</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value text-success">99,80%</div>
                        <div class="kpi-label">Acurácia Global (Teste)</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value text-success">98,92%</div>
                        <div class="kpi-label">Recall Médio Falhas</div>
                    </div>
                </div>
                <div class="col-6 col-md-4 col-xl-2">
                    <div class="kpi-box">
                        <div class="kpi-value" style="color: #fbbf24;">32</div>
                        <div class="kpi-label">Features Dinâmicas</div>
                    </div>
                </div>
            </div>

            <div class="row g-4">
                <div class="col-lg-6">
                    <div class="weg-card">
                        <h6 class="fw-bold text-white mb-2">Assinatura Física por Modo de Falha (Pós-Saneamento)</h6>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Comportamento médio dos sensores em cada estado: Falha 2 derruba a vazão para 6.3 L/min e eleva a temperatura a 88 °C; Falha 3 dispara o torque (184 Nm) e a vibração (6.0 mm/s).
                        </p>
                        <div class="chart-container"><canvas id="chartFaultSignatures"></canvas></div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <h6 class="fw-bold text-white mb-2">Importância das Features no Diagnóstico Preditivo</h6>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Destaque para o <strong>Índice de Esforço Mecânico (IEM)</strong> e para as variáveis normalizadas pela média da liga metálica.
                        </p>
                        <div class="chart-container"><canvas id="chartFeatureImportance"></canvas></div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <h6 class="fw-bold text-white mb-2">Esforço Termomecânico Nominal por Liga Metálica</h6>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Justifica a normalização condicional: a Liga de Titânio exige 151 Nm de torque nominal contra 93 Nm do Alumínio 7075 em operação normal.
                        </p>
                        <div class="chart-container"><canvas id="chartAlloys"></canvas></div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <h6 class="fw-bold text-white mb-2">Distribuição das Classes Operacionais (10.000 Leituras)</h6>
                        <p class="small mb-3" style="color: var(--text-muted);">
                            Desbalanceamento tratado via ponderação inversa de frequência (<code>class_weight='balanced'</code>).
                        </p>
                        <div class="chart-container"><canvas id="chartClassDistribution"></canvas></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SEÇÃO 2: INSPEÇÃO GEOMÉTRICA DE CORDÃO DE SOLDA -->
        <section id="sec-weld" class="section-block">
            <span class="section-header-tag">ETAPA 3.1 • VISÃO COMPUTACIONAL APLICADA À SOLDAGEM ROBÓTICA (OPENCV)</span>
            <h2 class="section-title">2. Inspeção Geométrica de Cordão de Solda (Canny & Porosidades)</h2>

            <div class="row g-4">
                <div class="col-lg-5">
                    <div class="weg-card">
                        <h5 class="fw-bold mb-3" style="color: var(--weg-cyan);">1. Entrada Óptica & Parametrização</h5>
                        <p class="small" style="color: var(--text-muted);">
                            Teste enviando uma imagem do seu computador (ex.: <code>data/test_images/weld/</code>) ou gere instantaneamente uma amostra sintética calibrada clicando nos botões rápidos abaixo:
                        </p>

                        <div class="d-flex flex-wrap gap-2 mb-3">
                            <button type="button" class="btn btn-sm btn-outline-success" onclick="testarPresetSolda('nominal')">✨ Simular Solda Perfeita</button>
                            <button type="button" class="btn btn-sm btn-outline-warning" onclick="testarPresetSolda('poros')">⚠️ Simular Solda com Poros</button>
                            <button type="button" class="btn btn-sm btn-outline-danger" onclick="testarPresetSolda('critico')">🚨 Simular Falha + Porosidade</button>
                        </div>

                        <label class="form-label small fw-bold">Ou faça Upload de Imagem do Cordão de Solda:</label>
                        <input type="file" id="fileWeld" class="form-control form-control-sm mb-2" accept="image/*">
                        <div class="preview-box mb-3" id="boxWeldPreview"><span class="text-secondary small">Preview da imagem do cordão...</span></div>

                        <hr class="border-secondary my-3">
                        <h6 class="fw-bold text-white mb-2">⚙️ Parâmetros do Detector Óptico</h6>

                        <div class="mb-2">
                            <div class="slider-label">
                                <span>Limiar Máximo de Porosidade (%):</span>
                                <span id="lblPorosity" style="color: var(--weg-cyan);" class="fw-bold">0.15%</span>
                            </div>
                            <input type="range" class="form-range" id="rngPorosity" min="0.05" max="2.00" step="0.05" value="0.15">
                        </div>

                        <div class="row g-2 mb-3">
                            <div class="col-6">
                                <div class="slider-label">
                                    <span>Canny Low:</span>
                                    <span id="lblCannyLow" style="color: var(--weg-cyan);" class="fw-bold">45</span>
                                </div>
                                <input type="range" class="form-range" id="rngCannyLow" min="15" max="100" step="5" value="45">
                            </div>
                            <div class="col-6">
                                <div class="slider-label">
                                    <span>Canny High:</span>
                                    <span id="lblCannyHigh" style="color: var(--weg-cyan);" class="fw-bold">135</span>
                                </div>
                                <input type="range" class="form-range" id="rngCannyHigh" min="80" max="240" step="5" value="135">
                            </div>
                        </div>

                        <button id="btnWeldUpload" class="btn-weg-action w-100">🔍 Inspecionar Arquivo Enviado</button>
                    </div>
                </div>

                <div class="col-lg-7">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-bold mb-0" style="color: var(--weg-cyan);">2. Laudo de Qualidade de Soldagem</h5>
                            <span id="badgeWeldStatus" class="badge bg-secondary px-3 py-2">Aguardando Inspeção</span>
                        </div>

                        <div id="weldEmptyState" class="text-center py-5 my-4" style="color: var(--text-muted);">
                            <p class="mb-1">Clique em um dos botões de simulação ao lado ou envie uma imagem para executar o filtro de nitidez, Canny Edge Detection e contagem de poros.</p>
                        </div>

                        <div id="weldResult" style="display: none;">
                            <div class="row g-2 mb-3">
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Poros Detectados</small>
                                        <span id="metricNumPores" class="fs-5 fw-bold text-white">0</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Taxa Porosidade</small>
                                        <span id="metricPorosityPct" class="fs-5 fw-bold text-warning">0.000%</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Área do Cordão</small>
                                        <span id="metricBeadArea" class="fs-5 fw-bold text-white">0 px</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Continuidade</small>
                                        <span id="metricContinuity" class="fs-6 fw-bold text-success">ÍNTEGRO</span>
                                    </div>
                                </div>
                            </div>

                            <div class="row g-3 text-center">
                                <div class="col-md-4">
                                    <small class="d-block mb-1 fw-bold" style="color: var(--weg-cyan);">Bordas Canny Edge</small>
                                    <div class="preview-box"><img id="imgWeldCanny" src=""></div>
                                </div>
                                <div class="col-md-4">
                                    <small class="d-block mb-1 fw-bold text-warning">Máscara Isolada de Poros</small>
                                    <div class="preview-box"><img id="imgWeldMask" src=""></div>
                                </div>
                                <div class="col-md-4">
                                    <small class="d-block mb-1 fw-bold text-danger">Mapeamento de Porosidades</small>
                                    <div class="preview-box"><img id="imgWeldOverlay" src=""></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SEÇÃO 3: TERMOGRAFIA DA ZONA AFETADA PELO CALOR (HAZ) -->
        <section id="sec-thermal" class="section-block">
            <span class="section-header-tag">ETAPA 3.1 • ANÁLISE TERMOGRÁFICA DA ZONA AFETADA PELO CALOR (HAZ)</span>
            <h2 class="section-title">3. Termografia da Solda & Matriz no Espaço de Cores HSV</h2>

            <div class="row g-4">
                <div class="col-lg-5">
                    <div class="weg-card">
                        <h5 class="fw-bold mb-3" style="color: var(--weg-cyan);">1. Captura Termográfica Infravermelha</h5>
                        <p class="small" style="color: var(--text-muted);">
                            Converte o perfil térmico para o espaço <strong>HSV</strong> e aplica limiarização para isolar núcleos de superaquecimento capazes de induzir trincas térmicas.
                        </p>

                        <div class="d-flex gap-2 mb-3">
                            <button type="button" class="btn btn-sm btn-outline-success w-50" onclick="testarPresetTermico(false)">🟢 Simular HAZ Normal</button>
                            <button type="button" class="btn btn-sm btn-outline-danger w-50" onclick="testarPresetTermico(true)">🔥 Simular Hotspot Crítico</button>
                        </div>

                        <label class="form-label small fw-bold">Ou envie um arquivo termográfico (<code>data/test_images/thermal/</code>):</label>
                        <input type="file" id="fileThermal" class="form-control form-control-sm mb-3" accept="image/*">
                        <div class="preview-box mb-3" id="boxThermalPreview"><span class="text-secondary small">Preview do mapa térmico...</span></div>
                        <button id="btnThermalUpload" class="btn-weg-action w-100">🔥 Analisar Termografia Enviada (HSV)</button>
                    </div>
                </div>

                <div class="col-lg-7">
                    <div class="weg-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-bold mb-0" style="color: var(--weg-cyan);">2. Diagnóstico Termomecânico da HAZ</h5>
                            <span id="badgeThermal" class="badge bg-secondary px-3 py-2">Aguardando Imagem</span>
                        </div>

                        <div id="thermalEmpty" class="text-center py-5 my-4" style="color: var(--text-muted);">
                            <p class="mb-1">Use os botões de simulação térmica ou envie uma imagem para isolar hotspots críticos no espaço HSV.</p>
                        </div>

                        <div id="thermalResult" style="display: none;">
                            <div class="row g-2 mb-3">
                                <div class="col-md-4">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Área Crítica (px)</small>
                                        <span id="thermalAreaPx" class="fs-5 fw-bold text-white">0 px</span>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">% Sobreaquecimento HAZ</small>
                                        <span id="thermalRatio" class="fs-5 fw-bold text-danger">0.000%</span>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="kpi-box py-2">
                                        <small class="kpi-label d-block">Núcleos (Hotspots)</small>
                                        <span id="thermalSpots" class="fs-5 fw-bold text-warning">0</span>
                                    </div>
                                </div>
                            </div>

                            <div class="row g-3 text-center">
                                <div class="col-md-6">
                                    <small class="d-block mb-1 fw-bold" style="color: var(--weg-cyan);">Máscara Binária HSV (Limiar Crítico)</small>
                                    <div class="preview-box" style="height: 230px;"><img id="imgThermalMask" src=""></div>
                                </div>
                                <div class="col-md-6">
                                    <small class="d-block mb-1 fw-bold text-warning">Delimilitação de Zonas de Risco Térmico</small>
                                    <div class="preview-box" style="height: 230px;"><img id="imgThermalOverlay" src=""></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SEÇÃO 4: SIMULADOR IOT, PREDIÇÃO E OEE -->
        <section id="sec-telemetry" class="section-block border-bottom-0">
            <span class="section-header-tag">ETAPA 3.2 • ATUAÇÃO EM MALHA FECHADA & IMPACTO NO OEE</span>
            <h2 class="section-title">4. Simulador de Telemetria IoT, Diagnóstico & Cálculo Preditivo de OEE</h2>

            <div class="row g-4">
                <div class="col-lg-6">
                    <div class="weg-card">
                        <div class="d-flex flex-wrap justify-content-between align-items-center mb-3 gap-2">
                            <h5 class="fw-bold mb-0" style="color: var(--weg-cyan);">1. Leitura dos Sensores da Célula</h5>
                            <div class="btn-group btn-group-sm">
                                <button type="button" class="btn btn-outline-success" onclick="carregarPresetIoT('normal')">Nominal (Ti)</button>
                                <button type="button" class="btn btn-outline-warning" onclick="carregarPresetIoT('desgaste')">Falha 1: Desgaste</button>
                                <button type="button" class="btn btn-outline-info" onclick="carregarPresetIoT('refrigeracao')">Falha 2: Refrig.</button>
                                <button type="button" class="btn btn-outline-danger" onclick="carregarPresetIoT('trinca')">Falha 3: Trinca</button>
                            </div>
                        </div>

                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label small fw-bold">Célula de Manufatura:</label>
                                <select id="telCelula" class="form-select">
                                    <option value="CEL-201">CEL-201 (Prensa & Usinagem)</option>
                                    <option value="CEL-202">CEL-202 (Centro CNC 5 Eixos)</option>
                                    <option value="CEL-203" selected>CEL-203 (Solda & Estamparia)</option>
                                    <option value="CEL-204">CEL-204 (Acabamento Aero)</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-bold">Liga Metálica em Processo:</label>
                                <select id="telLiga" class="form-select">
                                    <option value="Aco_Carbono_316L">Aco_Carbono_316L</option>
                                    <option value="Aluminio_7075">Aluminio_7075</option>
                                    <option value="Liga_Titanio_Ti6Al4V" selected>Liga_Titanio_Ti6Al4V</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Temp. Fuso (°C):</label>
                                <input type="number" id="telTemp" class="form-control" value="68.0" step="0.5">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Vibração RMS (mm/s):</label>
                                <input type="number" id="telVib" class="form-control" value="2.35" step="0.05">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Torque Fuso (Nm):</label>
                                <input type="number" id="telTorque" class="form-control" value="144.0" step="1.0">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Vazão Fluido (L/min):</label>
                                <input type="number" id="telVazao" class="form-control" value="28.0" step="0.5">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Corrente Motor (A):</label>
                                <input type="number" id="telCorrente" class="form-control" value="36.5" step="0.5">
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-bold">Pressão Fixação (Bar):</label>
                                <input type="number" id="telPressao" class="form-control" value="160.0" step="1.0">
                            </div>
                            <div class="col-md-12">
                                <label class="form-label small fw-bold">Nível de Ruído Acústico (dB):</label>
                                <input type="number" id="telRuido" class="form-control" value="62.0" step="0.5">
                            </div>
                        </div>

                        <button id="btnPredictIoT" class="btn-weg-action w-100 mt-4">📡 Executar Inferência em Malha Fechada (/predict)</button>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="weg-card">
                        <h5 class="fw-bold mb-3" style="color: var(--weg-cyan);">2. Comando para o CLP & Impacto Preditivo no OEE</h5>

                        <div class="kpi-box text-start px-4 py-3 mb-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <small class="kpi-label d-block">Diagnóstico do Modelo Supervisionado:</small>
                                    <h4 id="iotDiagText" class="fw-bold text-white mb-0">Aguardando telemetria...</h4>
                                </div>
                                <span id="iotProbBadge" class="badge bg-info text-dark fs-6">---</span>
                            </div>
                        </div>

                        <div class="kpi-box text-start px-4 py-3 mb-3" style="border-top-color: #f59e0b;">
                            <small class="kpi-label d-block">Ação Recomendada ao Controlador Lógico (CLP):</small>
                            <h4 id="iotActionText" class="fw-bold text-warning mb-0">---</h4>
                        </div>

                        <h6 class="text-white small fw-bold text-uppercase mt-3 mb-2">Estimativa Dinâmica de OEE (Overall Equipment Effectiveness):</h6>
                        <div class="row g-2 text-center">
                            <div class="col-3">
                                <div class="kpi-box py-2">
                                    <small class="kpi-label d-block">Disponib.</small>
                                    <span id="oeeDisp" class="fs-5 fw-bold text-white">---</span>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="kpi-box py-2">
                                    <small class="kpi-label d-block">Perform.</small>
                                    <span id="oeePerf" class="fs-5 fw-bold text-white">---</span>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="kpi-box py-2">
                                    <small class="kpi-label d-block">Qualidade</small>
                                    <span id="oeeQual" class="fs-5 fw-bold text-white">---</span>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="kpi-box py-2" style="border-top-color: #22c55e;">
                                    <small class="kpi-label d-block">OEE Global</small>
                                    <span id="oeeGlobal" class="fs-5 fw-bold text-success">---</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <footer class="text-center">
        <div class="container">
            <p class="mb-1 fw-bold text-white">VulcanTech Smart Metals S.A. • Plataforma de Inteligência Industrial e Visão Computacional</p>
            <small>Indústria 4.0 & MLOps • Jaraguá do Sul, SC — 2026 • Desenvolvido por Henrique (Rique)</small>
        </div>
    </footer>

    <script>
        Chart.defaults.color = '#94b2d1';
        Chart.defaults.borderColor = 'rgba(25, 75, 133, 0.35)';

        async function initCharts() {
            const res = await fetch('/analytics/summary');
            const data = await res.json();

            new Chart(document.getElementById('chartFaultSignatures'), {
                type: 'bar',
                data: {
                    labels: data.assinatura_fisica_falhas.classes,
                    datasets: [
                        { label: 'Temperatura (°C)', data: data.assinatura_fisica_falhas.temperatura_c, backgroundColor: '#00579d' },
                        { label: 'Torque (Nm)', data: data.assinatura_fisica_falhas.torque_nm, backgroundColor: '#5bc2e7' },
                        { label: 'Vazão Fluido (L/min)', data: data.assinatura_fisica_falhas.vazao_lmin, backgroundColor: '#22c55e' },
                        { label: 'Vibração RMS (mm/s × 10)', data: data.assinatura_fisica_falhas.vibracao_mms.map(v => v * 10), backgroundColor: '#f59e0b' }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            new Chart(document.getElementById('chartFeatureImportance'), {
                type: 'bar',
                data: {
                    labels: data.importancia_features.features,
                    datasets: [{
                        label: 'Importância Relativa (%)',
                        data: data.importancia_features.pesos,
                        backgroundColor: '#5bc2e7'
                    }]
                },
                options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });

            new Chart(document.getElementById('chartAlloys'), {
                type: 'bar',
                data: {
                    labels: data.perfil_ligas.ligas,
                    datasets: [
                        { label: 'Torque Médio (Nm)', data: data.perfil_ligas.torque_medio, backgroundColor: '#00579d' },
                        { label: 'Temperatura Média (°C)', data: data.perfil_ligas.temp_media, backgroundColor: '#38bdf8' },
                        { label: 'Corrente Estatórica (A)', data: data.perfil_ligas.corrente_media, backgroundColor: '#fbbf24' }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            new Chart(document.getElementById('chartClassDistribution'), {
                type: 'doughnut',
                data: {
                    labels: data.distribuicao_classes.labels,
                    datasets: [{
                        data: data.distribuicao_classes.counts,
                        backgroundColor: ['#00579d', '#f59e0b', '#5bc2e7', '#ef4444'],
                        borderColor: '#072244',
                        borderWidth: 2
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' } } }
            });
        }
        initCharts();

        async function checkHealth() {
            const badge = document.getElementById('badgeEdgeStatus');
            try {
                const res = await fetch('/health');
                const data = await res.json();
                if (data.status === 'OPERATIONAL') {
                    badge.className = 'badge bg-success badge-heartbeat';
                    badge.innerHTML = `<span class="status-led"></span> Edge Online (${data.latencia_media_ms} ms)`;
                } else {
                    badge.className = 'badge bg-warning text-dark badge-heartbeat';
                    badge.innerHTML = '⚠️ Modelo Pendente';
                }
            } catch {
                badge.className = 'badge bg-danger badge-heartbeat';
                badge.innerHTML = 'Offline';
            }
        }
        checkHealth();
        setInterval(checkHealth, 4000);

        // Sliders de solda
        document.getElementById('rngPorosity').addEventListener('input', e => document.getElementById('lblPorosity').innerText = e.target.value + '%');
        document.getElementById('rngCannyLow').addEventListener('input', e => document.getElementById('lblCannyLow').innerText = e.target.value);
        document.getElementById('rngCannyHigh').addEventListener('input', e => document.getElementById('lblCannyHigh').innerText = e.target.value);

        function bindPreview(inputId, boxId) {
            document.getElementById(inputId).addEventListener('change', e => {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = ev => document.getElementById(boxId).innerHTML = `<img src="${ev.target.result}">`;
                    reader.readAsDataURL(file);
                }
            });
        }
        bindPreview('fileWeld', 'boxWeldPreview');
        bindPreview('fileThermal', 'boxThermalPreview');

        function renderWeldResult(data) {
            document.getElementById('weldEmptyState').style.display = 'none';
            document.getElementById('weldResult').style.display = 'block';

            const badge = document.getElementById('badgeWeldStatus');
            badge.innerText = data.status;
            badge.className = 'badge px-3 py-2 ' + (data.aprovado ? 'bg-success' : 'bg-danger');

            document.getElementById('metricNumPores').innerText = data.total_poros_detetados;
            document.getElementById('metricPorosityPct').innerText = data.taxa_porosidade_pct + '%';
            document.getElementById('metricBeadArea').innerText = data.area_cordao_px + ' px';

            const contEl = document.getElementById('metricContinuity');
            contEl.innerText = data.tem_descontinuidade ? 'INTERROMPIDO' : 'ÍNTEGRO';
            contEl.className = 'fs-6 fw-bold ' + (data.tem_descontinuidade ? 'text-danger' : 'text-success');

            document.getElementById('imgWeldCanny').src = data.imagens_b64.canny_b64;
            document.getElementById('imgWeldMask').src = data.imagens_b64.mascara_poros_b64;
            document.getElementById('imgWeldOverlay').src = data.imagens_b64.overlay_b64;
        }

        async function testarPresetSolda(cenario) {
            const fd = new FormData();
            fd.append('cenario', cenario);
            fd.append('limiar_porosidade_pct', document.getElementById('rngPorosity').value);
            fd.append('canny_low', document.getElementById('rngCannyLow').value);
            fd.append('canny_high', document.getElementById('rngCannyHigh').value);

            const res = await fetch('/inspect/weld-preset', { method: 'POST', body: fd });
            const data = await res.json();
            document.getElementById('boxWeldPreview').innerHTML = `<img src="${data.original_b64}">`;
            renderWeldResult(data);
        }

        document.getElementById('btnWeldUpload').addEventListener('click', async () => {
            const f = document.getElementById('fileWeld').files[0];
            if (!f) return alert('Selecione uma imagem de solda ou use os botões de simulação rápida acima!');
            const fd = new FormData();
            fd.append('file', f);
            fd.append('limiar_porosidade_pct', document.getElementById('rngPorosity').value);
            fd.append('canny_low', document.getElementById('rngCannyLow').value);
            fd.append('canny_high', document.getElementById('rngCannyHigh').value);

            const res = await fetch('/inspect/weld', { method: 'POST', body: fd });
            const data = await res.json();
            renderWeldResult(data);
        });

        function renderThermalResult(data) {
            document.getElementById('thermalEmpty').style.display = 'none';
            document.getElementById('thermalResult').style.display = 'block';

            const b = document.getElementById('badgeThermal');
            b.innerText = data.status;
            b.className = 'badge px-3 py-2 ' + (data.risco_trinca_termica ? 'bg-danger' : 'bg-success');

            document.getElementById('thermalAreaPx').innerText = data.area_critica_px + ' px';
            document.getElementById('thermalRatio').innerText = data.percentagem_haz_critica + '%';
            document.getElementById('thermalSpots').innerText = data.quantidade_hotspots;
            document.getElementById('imgThermalMask').src = data.imagens_b64.mascara_b64;
            document.getElementById('imgThermalOverlay').src = data.imagens_b64.overlay_b64;
        }

        async function testarPresetTermico(critico) {
            const fd = new FormData();
            fd.append('critico', critico);
            const res = await fetch('/inspect/thermal-preset', { method: 'POST', body: fd });
            const data = await res.json();
            document.getElementById('boxThermalPreview').innerHTML = `<img src="${data.original_b64}">`;
            renderThermalResult(data);
        }

        document.getElementById('btnThermalUpload').addEventListener('click', async () => {
            const f = document.getElementById('fileThermal').files[0];
            if (!f) return alert('Selecione um arquivo termográfico ou clique nos botões de simulação acima!');
            const fd = new FormData();
            fd.append('file', f);
            const res = await fetch('/inspect/thermal', { method: 'POST', body: fd });
            const data = await res.json();
            renderThermalResult(data);
        });

        function carregarPresetIoT(tipo) {
            if (tipo === 'normal') {
                document.getElementById('telLiga').value = 'Liga_Titanio_Ti6Al4V';
                document.getElementById('telTemp').value = 68.0;
                document.getElementById('telVib').value = 2.35;
                document.getElementById('telTorque').value = 144.0;
                document.getElementById('telVazao').value = 28.0;
                document.getElementById('telCorrente').value = 36.5;
                document.getElementById('telPressao').value = 160.0;
                document.getElementById('telRuido').value = 62.0;
            } else if (tipo === 'desgaste') {
                document.getElementById('telLiga').value = 'Aco_Carbono_316L';
                document.getElementById('telTemp').value = 68.5;
                document.getElementById('telVib').value = 5.10;
                document.getElementById('telTorque').value = 175.0;
                document.getElementById('telVazao').value = 25.0;
                document.getElementById('telCorrente').value = 44.0;
                document.getElementById('telPressao').value = 154.0;
                document.getElementById('telRuido').value = 82.0;
            } else if (tipo === 'refrigeracao') {
                document.getElementById('telLiga').value = 'Aco_Carbono_316L';
                document.getElementById('telTemp').value = 96.0;
                document.getElementById('telVib').value = 2.60;
                document.getElementById('telTorque').value = 128.0;
                document.getElementById('telVazao').value = 5.2;
                document.getElementById('telCorrente').value = 36.0;
                document.getElementById('telPressao').value = 149.0;
                document.getElementById('telRuido').value = 64.0;
            } else if (tipo === 'trinca') {
                document.getElementById('telLiga').value = 'Liga_Titanio_Ti6Al4V';
                document.getElementById('telTemp').value = 78.0;
                document.getElementById('telVib').value = 8.40;
                document.getElementById('telTorque').value = 255.0;
                document.getElementById('telVazao').value = 26.0;
                document.getElementById('telCorrente').value = 61.0;
                document.getElementById('telPressao').value = 85.0;
                document.getElementById('telRuido').value = 108.0;
            }
        }

        document.getElementById('btnPredictIoT').addEventListener('click', async () => {
            const payload = {
                id_celula: document.getElementById('telCelula').value,
                liga_metalica: document.getElementById('telLiga').value,
                temperatura_fuso_c: parseFloat(document.getElementById('telTemp').value),
                vibracao_triaxial_rms_mms: parseFloat(document.getElementById('telVib').value),
                torque_fuso_nm: parseFloat(document.getElementById('telTorque').value),
                vazao_refrigerante_lmin: parseFloat(document.getElementById('telVazao').value),
                corrente_estator_amp: parseFloat(document.getElementById('telCorrente').value),
                pressao_hidraulica_bar: parseFloat(document.getElementById('telPressao').value),
                nivel_ruido_db: parseFloat(document.getElementById('telRuido').value)
            };

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            document.getElementById('iotDiagText').innerText = `${data.diagnostico} (Classe ${data.tipo_falha_codigo})`;
            document.getElementById('iotProbBadge').innerText = `Confiança: ${(data.probabilidade * 100).toFixed(1)}%`;

            const actEl = document.getElementById('iotActionText');
            actEl.innerText = data.acao_recomendada;
            actEl.className = 'fw-bold mb-0 ' + (
                data.acao_recomendada === 'MANTER_OPERACAO' ? 'text-success' :
                data.acao_recomendada === 'INTERROMPER_EMERGENCIA' ? 'text-danger' : 'text-warning'
            );

            const oee = data.impacto_oee_estimado;
            document.getElementById('oeeDisp').innerText = (oee.disponibilidade * 100).toFixed(0) + '%';
            document.getElementById('oeePerf').innerText = (oee.performance * 100).toFixed(0) + '%';
            document.getElementById('oeeQual').innerText = (oee.qualidade * 100).toFixed(0) + '%';
            document.getElementById('oeeGlobal').innerText = (oee.oee_calculado * 100).toFixed(1) + '%';
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    # Permite rodar clicando diretamente no botão Play do VS Code!
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(BASE_DIR))
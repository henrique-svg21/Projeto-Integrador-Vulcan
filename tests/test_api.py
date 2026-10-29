"""
Suite de testes automatizados para validação dos endpoints da API RESTful.
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Garante acesso à raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from api.main import app, CAMINHO_MODELO

client = TestClient(app)

def test_healthcheck():
    """Valida se o endpoint de integridade responde 200 e reporta modelo carregado."""
    response = client.get("/health")
    assert response.status_code == 200
    dados = response.json()
    assert dados["status"] == "OPERATIONAL"
    assert dados["modelo_carregado"] is True

def test_predicao_operacao_normal():
    """Valida a inferência em condição nominal de máquina (Classe 0)."""
    payload = {
        "id_celula": "CEL-201",
        "liga_metalica": "Aco_Carbono_316L",
        "temperatura_fuso_c": 52.0,
        "vibracao_triaxial_rms_mms": 1.8,
        "torque_fuso_nm": 110.0,
        "vazao_refrigerante_lmin": 28.0,
        "corrente_estator_amp": 28.0,
        "pressao_hidraulica_bar": 160.0,
        "nivel_ruido_db": 55.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["tipo_falha_codigo"] == 0
    assert res["acao_recomendada"] == "MANTER_OPERACAO"
    assert res["impacto_oee_estimado"]["oee_calculado"] == 1.0

def test_predicao_falha_severa_desalinhamento():
    """Valida gatilho emergencial para vibração excessiva e perda de fixação (Classe 3)."""
    payload = {
        "id_celula": "CEL-203",
        "liga_metalica": "Liga_Titanio_Ti6Al4V",
        "temperatura_fuso_c": 80.0,
        "vibracao_triaxial_rms_mms": 8.5,     # Ruptura mecânica
        "torque_fuso_nm": 260.0,               # Sobrecarga
        "vazao_refrigerante_lmin": 25.0,
        "corrente_estator_amp": 60.0,
        "pressao_hidraulica_bar": 70.0,        # Queda drástica de pressão
        "nivel_ruido_db": 110.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["tipo_falha_codigo"] in [1, 3]  # Anomalia grave
    assert res["acao_recomendada"] in ["INTERROMPER_EMERGENCIA", "REDUZIR_VELOCIDADE_50%"]

def test_validacao_rejeicao_dados_invalidos():
    """Valida se o Pydantic rejeita leituras com valores fisicamente absurdos."""
    payload_invalido = {
        "id_celula": "CEL-999",  # Célula inexistente
        "liga_metalica": "Ouro",  # Liga inválida
        "temperatura_fuso_c": -50.0,
        "vibracao_triaxial_rms_mms": 1.0,
        "torque_fuso_nm": 100.0,
        "vazao_refrigerante_lmin": 25.0,
        "corrente_estator_amp": 20.0,
        "pressao_hidraulica_bar": 150.0,
        "nivel_ruido_db": 50.0
    }
    response = client.post("/predict", json=payload_invalido)
    assert response.status_code == 422  # Unprocessable Entity
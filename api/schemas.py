"""
Esquemas Pydantic para validação estrita de dados de telemetria IoT,
inferência preditiva e diagnóstico operacional da VulcanTech.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

LigaMetalicaTipo = Literal[
    "Aco_Carbono_316L",
    "Aluminio_7075",
    "Liga_Titanio_Ti6Al4V"
]

AcaoOperacionalTipo = Literal[
    "MANTER_OPERACAO",
    "REDUZIR_VELOCIDADE_50%",
    "INTERROMPER_EMERGENCIA"
]

class LeituraSensorInput(BaseModel):
    """Telemetria instantânea transmitida pelos sensores IoT do fuso e prensa."""
    id_celula: Literal["CEL-201", "CEL-202", "CEL-203", "CEL-204"] = Field(
        ..., description="Identificador da célula flexível de manufatura."
    )
    liga_metalica: LigaMetalicaTipo = Field(
        ..., description="Material em processamento no momento da recolha."
    )
    temperatura_fuso_c: float = Field(..., ge=0.0, le=300.0, description="Temperatura do cabeçote/matriz em °C.")
    vibracao_triaxial_rms_mms: float = Field(..., ge=0.0, le=50.0, description="Vibração triaxial RMS em mm/s.")
    torque_fuso_nm: float = Field(..., ge=0.0, le=500.0, description="Torque instantâneo no fuso em Nm.")
    vazao_refrigerante_lmin: float = Field(..., ge=0.0, le=100.0, description="Fluxo de fluido de corte em L/min.")
    corrente_estator_amp: float = Field(..., ge=0.0, le=150.0, description="Corrente de acionamento em Amperes.")
    pressao_hidraulica_bar: float = Field(..., ge=0.0, le=400.0, description="Pressão do sistema de fixação em Bar.")
    nivel_ruido_db: float = Field(..., ge=20.0, le=150.0, description="Emissão acústica de corte em dB.")

class DiagnosticoPredicao(BaseModel):
    """Resultado da inferência preditiva com atuação em malha fechada."""
    id_celula: str
    tipo_falha_codigo: int = Field(..., description="0=Normal, 1=Desgaste, 2=Refrigeração, 3=Trinca/Desalinhamento.")
    diagnostico: str
    probabilidade: float = Field(..., ge=0.0, le=1.0)
    acao_recomendada: AcaoOperacionalTipo
    impacto_oee_estimado: Dict[str, float]

class LoteSensoresInput(BaseModel):
    leituras: List[LeituraSensorInput]

class RespostaLote(BaseModel):
    total_processado: int
    predicoes: List[DiagnosticoPredicao]

class HealthCheckResponse(BaseModel):
    status: str
    modelo_carregado: bool
    versao_api: str
    latencia_media_ms: float
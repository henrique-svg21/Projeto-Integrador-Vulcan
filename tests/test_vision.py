"""
Suite de avaliação de acurácia, recall e latência do pipeline de visão computacional.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.vision.weld_inspection import inspecionar_cordao_solda
from src.vision.thermal_haz import analisar_haz_termica

def gerar_amostra_controlada(com_defeito: bool, seed: int):
    """Gera amostra determinística com Ground Truth conhecido."""
    np.random.seed(seed)
    h, w = 300, 600
    img = np.full((h, w, 3), 55, dtype=np.uint8)
    
    # Desenho do cordão
    centro_y = 150
    for x in range(40, 560):
        if com_defeito and 280 <= x <= 330:
            continue  # Descontinuidade geométrica intencional
        ondulacao = int(centro_y + 4 * np.sin(x / 14.0))
        largura = int(24 + 3 * np.cos(x / 18.0))
        cv2.line(img, (x, ondulacao - largura), (x, ondulacao + largura), (150, 150, 160), 1)

    if com_defeito:
        # Adicionar múltiplos poros severos
        for _ in range(16):
            px = np.random.randint(90, 510)
            py = np.random.randint(centro_y - 12, centro_y + 12)
            cv2.circle(img, (px, py), 4, (20, 20, 20), -1)

    return img

def test_latencia_visao_computacional():
    """Valida se a inspeção de imagem executa em menos de 50 ms (requisito Edge real-time)."""
    img = gerar_amostra_controlada(com_defeito=False, seed=1)
    
    # Aquecimento (Warm-up)
    _ = inspecionar_cordao_solda(img)
    
    tempos = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = inspecionar_cordao_solda(img)
        tempos.append((time.perf_counter() - t0) * 1000.0)
        
    latencia_media = np.mean(tempos)
    print(f"\n[BENCHMARK] Latência média por quadro: {latencia_media:.2f} ms")
    assert latencia_media < 50.0, f"Latência de {latencia_media:.2f} ms excede o teto de 50 ms para malha fechada."

def test_benchmark_acuracia_solda():
    """Avalia Precisão e Recall em um lote de 20 amostras com ground truth conhecido."""
    y_true = []
    y_pred = []
    
    for i in range(20):
        defeito_real = (i % 2 == 0)
        img = gerar_amostra_controlada(com_defeito=defeito_real, seed=i)
        
        resultado = inspecionar_cordao_solda(img)
        reprovado_pelo_algoritmo = not resultado["aprovado"]
        
        y_true.append(defeito_real)
        y_pred.append(reprovado_pelo_algoritmo)
        
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"\n[MÉTRICAS SOLDADURA] Precisão: {precisao * 100:.1f}% | Recall: {recall * 100:.1f}%")
    assert recall >= 0.80, "O Recall na identificação de defeitos críticos deve ser de no mínimo 80%."
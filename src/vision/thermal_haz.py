"""
Módulo de análise termográfica da Zona Afetada pelo Calor (HAZ)
utilizando conversão para espaço de cor HSV e limiarização multiescala.
"""
from typing import Dict, Any, List
import cv2
import numpy as np

def analisar_haz_termica(imagem_termica_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Converte para HSV e segmenta zonas com sobreaquecimento crítico
    na escala térmica (matizes vermelhos e magenta de alta intensidade).
    """
    # 1. Conversão para o Espaço de Cores HSV
    hsv = cv2.cvtColor(imagem_termica_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. Definição dos limiares de sobreaquecimento crítico no espaço HSV
    # Setor 1: Vermelho inferior (H: 0 a 10)
    limiar_inf_vermelho1 = np.array([0, 140, 160], dtype=np.uint8)
    limiar_sup_vermelho1 = np.array([10, 255, 255], dtype=np.uint8)
    
    # Setor 2: Vermelho superior / Magenta (H: 170 a 180)
    limiar_inf_vermelho2 = np.array([170, 140, 160], dtype=np.uint8)
    limiar_sup_vermelho2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mascara_v1 = cv2.inRange(hsv, limiar_inf_vermelho1, limiar_sup_vermelho1)
    mascara_v2 = cv2.inRange(hsv, limiar_inf_vermelho2, limiar_sup_vermelho2)
    mascara_sobreaquecimento = cv2.bitwise_or(mascara_v1, mascara_v2)
    
    # 3. Operação morfológica de abertura para eliminação de ruídos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mascara_limpa = cv2.morphologyEx(mascara_sobreaquecimento, cv2.MORPH_OPEN, kernel)
    
    # 4. Quantificação da Severidade Térmica
    area_critica_px = int(cv2.countNonZero(mascara_limpa))
    total_pixels_imagem = imagem_termica_bgr.shape[0] * imagem_termica_bgr.shape[1]
    percentagem_critica = (area_critica_px / total_pixels_imagem) * 100.0
    
    # Deteção de núcleos de sobreaquecimento (hotspots)
    contornos_hotspots, _ = cv2.findContours(
        mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    hotspots: List[Dict[str, Any]] = []
    for cnt in contornos_hotspots:
        area = cv2.contourArea(cnt)
        if area >= 25.0:
            x, y, w, h = cv2.boundingRect(cnt)
            hotspots.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "area_px": float(area)
            })
            
    # Alerta de paragem: deteção de hotspots térmicos severos
    risco_trinca_termica = bool(area_critica_px > 120)

    return {
        "risco_trinca_termica": risco_trinca_termica,
        "area_critica_px": area_critica_px,
        "percentagem_haz_critica": round(percentagem_critica, 3),
        "quantidade_hotspots": len(hotspots),
        "hotspots": hotspots,
        "mascara_critica": mascara_limpa
    }
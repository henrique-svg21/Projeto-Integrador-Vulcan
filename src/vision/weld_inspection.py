"""
Módulo de visão computacional para inspeção geométrica,
medição de descontinuidades e deteção de porosidades em soldadura robótica.
"""
from typing import Dict, Any, List
import cv2
import numpy as np

def inspecionar_cordao_solda(imagem_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Aplica pipeline óptico: filtro de nitidez, escala de cinzentos,
    detetor Canny e análise morfológica de contornos.
    """
    # 1. Filtro de Nitidez (Kernel de Sharpening laplaciano)
    kernel_nitidez = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    img_nitida = cv2.filter2D(imagem_bgr, -1, kernel_nitidez)
    
    # 2. Conversão para Escala de Cinzentos
    cinza = cv2.cvtColor(img_nitida, cv2.COLOR_BGR2GRAY)
    
    # 3. Suavização Gaussiana para supressão de ruído de sensor
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # 4. Detetor de Bordos de Canny
    bordos_canny = cv2.Canny(suavizada, threshold1=45, threshold2=135)
    
    # 5. Segmentação da máscara do cordão (limiarização Otsu)
    _, mascara_cordao = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Deteção de porosidades internas (inversão adaptativa restrita ao cordão)
    mascara_poros = cv2.adaptiveThreshold(
        suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )
    mascara_poros = cv2.bitwise_and(mascara_poros, mascara_cordao)
    
    # Limpeza morfológica de ruídos de pixel unitário
    elemento_estruturante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_poros_limpa = cv2.morphologyEx(mascara_poros, cv2.MORPH_OPEN, elemento_estruturante)
    
    # 7. Análise de Contornos dos Poros
    contornos_poros, _ = cv2.findContours(
        mascara_poros_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    area_poros_total = 0.0
    detalhes_poros: List[Dict[str, Any]] = []
    
    for cnt in contornos_poros:
        area = cv2.contourArea(cnt)
        if 2.0 <= area <= 250.0:  # Faixa de escala de porosidade
            area_poros_total += area
            (x, y), raio = cv2.minEnclosingCircle(cnt)
            detalhes_poros.append({
                "centro": (int(x), int(y)),
                "raio": int(raio),
                "area_px": float(area)
            })

    area_cordao_px = int(cv2.countNonZero(mascara_cordao))
    taxa_porosidade = (area_poros_total / area_cordao_px * 100.0) if area_cordao_px > 0 else 0.0
    
    # Critério de reprovação mecânica por norma técnica (porosidade > 1.5%)
    cordao_aprovado = bool(taxa_porosidade <= 1.5)

    return {
        "aprovado": cordao_aprovado,
        "taxa_porosidade_pct": round(taxa_porosidade, 3),
        "total_poros_detetados": len(detalhes_poros),
        "area_poros_px": round(area_poros_total, 2),
        "area_cordao_px": area_cordao_px,
        "detalhes_poros": detalhes_poros,
        "mascaras": {
            "bordos_canny": bordos_canny,
            "mascara_cordao": mascara_cordao,
            "mascara_poros": mascara_poros_limpa
        }
    }
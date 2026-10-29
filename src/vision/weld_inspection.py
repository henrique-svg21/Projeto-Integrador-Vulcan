"""
Módulo de visão computacional para inspeção geométrica,
medição de descontinuidades e detecção de porosidades em soldagem robótica.
"""
import base64
from typing import Dict, Any, List
import cv2
import numpy as np

def _img_to_base64(img: np.ndarray) -> str:
    """Converte matriz OpenCV para Data URI Base64 exibível em tags <img> HTML."""
    _, buffer = cv2.imencode(".png", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def inspecionar_cordao_solda(
    imagem_bgr: np.ndarray,
    limiar_porosidade_pct: float = 1.5,
    canny_low: int = 45,
    canny_high: int = 135
) -> Dict[str, Any]:
    """
    Aplica pipeline óptico: filtro de nitidez, escala de cinza,
    detector Canny, análise morfológica de contornos e codificação Base64.
    """
    # 1. Filtro de Nitidez (Kernel Laplaciano)
    kernel_nitidez = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    img_nitida = cv2.filter2D(imagem_bgr, -1, kernel_nitidez)
    
    # 2. Conversão para Escala de Cinza e Suavização Gaussiana
    cinza = cv2.cvtColor(img_nitida, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # 3. Detector de Bordas de Canny
    bordos_canny = cv2.Canny(suavizada, threshold1=canny_low, threshold2=canny_high)
    
    # 4. Segmentação da máscara do cordão (limiarização Otsu)
    _, mascara_cordao = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Detecção de porosidades internas restritas ao cordão
    mascara_poros = cv2.adaptiveThreshold(
        suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )
    mascara_poros = cv2.bitwise_and(mascara_poros, mascara_cordao)
    
    elemento_estruturante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_poros_limpa = cv2.morphologyEx(mascara_poros, cv2.MORPH_OPEN, elemento_estruturante)
    
    # 6. Análise de Contornos dos Poros e Desenho do Overlay de Inspeção
    contornos_poros, _ = cv2.findContours(
        mascara_poros_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    img_overlay = imagem_bgr.copy()
    # Destaca contorno externo do cordão em ciano
    contornos_cordao, _ = cv2.findContours(mascara_cordao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_overlay, contornos_cordao, -1, (231, 194, 91), 1)

    area_poros_total = 0.0
    detalhes_poros: List[Dict[str, Any]] = []
    
    for cnt in contornos_poros:
        area = cv2.contourArea(cnt)
        if 2.0 <= area <= 250.0:
            area_poros_total += area
            (x, y), raio = cv2.minEnclosingCircle(cnt)
            centro = (int(x), int(y))
            r_int = max(int(raio), 2)
            detalhes_poros.append({
                "centro": centro,
                "raio": r_int,
                "area_px": float(area)
            })
            # Destaca o poro em vermelho néon
            cv2.circle(img_overlay, centro, r_int + 3, (0, 0, 255), 2)
            cv2.circle(img_overlay, centro, 1, (0, 255, 255), -1)

    # 7. Verificação de Descontinuidade Geométrica (interrupção longitudinal do cordão)
    projecao_horizontal = np.sum(mascara_cordao > 0, axis=0)
    regiao_central = projecao_horizontal[60:-60] if imagem_bgr.shape[1] > 140 else projecao_horizontal
    tem_descontinuidade = bool(np.any(regiao_central < 5))

    area_cordao_px = int(cv2.countNonZero(mascara_cordao))
    taxa_porosidade = (area_poros_total / area_cordao_px * 100.0) if area_cordao_px > 0 else 0.0
    
    cordao_aprovado = bool((taxa_porosidade <= limiar_porosidade_pct) and (not tem_descontinuidade))

    return {
        "aprovado": cordao_aprovado,
        "status": "APROVADO (CONFORME)" if cordao_aprovado else "REPROVADO (POROSIDADE / FALHA)",
        "taxa_porosidade_pct": round(taxa_porosidade, 3),
        "total_poros_detetados": len(detalhes_poros),
        "tem_descontinuidade": tem_descontinuidade,
        "area_poros_px": round(area_poros_total, 2),
        "area_cordao_px": area_cordao_px,
        "detalhes_poros": detalhes_poros,
        "imagens_b64": {
            "canny_b64": _img_to_base64(bordos_canny),
            "mascara_poros_b64": _img_to_base64(mascara_poros_limpa),
            "overlay_b64": _img_to_base64(img_overlay)
        },
        "mascaras": {
            "bordos_canny": bordos_canny,
            "mascara_cordao": mascara_cordao,
            "mascara_poros": mascara_poros_limpa
        }
    }
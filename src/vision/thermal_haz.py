"""
Módulo de análise termográfica da Zona Afetada pelo Calor (HAZ)
utilizando conversão para espaço de cor HSV e limiarização multiescala.
"""
import base64
from typing import Dict, Any, List
import cv2
import numpy as np

def _img_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def analisar_haz_termica(imagem_termica_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Converte para HSV, segmenta zonas com superaquecimento crítico
    e gera imagens anotadas em Base64.
    """
    hsv = cv2.cvtColor(imagem_termica_bgr, cv2.COLOR_BGR2HSV)
    
    # Faixas HSV de superaquecimento crítico (Vermelho Intenso / Magenta)
    limiar_inf_v1 = np.array([0, 140, 160], dtype=np.uint8)
    limiar_sup_v1 = np.array([10, 255, 255], dtype=np.uint8)
    limiar_inf_v2 = np.array([170, 140, 160], dtype=np.uint8)
    limiar_sup_v2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mascara_v1 = cv2.inRange(hsv, limiar_inf_v1, limiar_sup_v1)
    mascara_v2 = cv2.inRange(hsv, limiar_inf_v2, limiar_sup_v2)
    mascara_sobreaquecimento = cv2.bitwise_or(mascara_v1, mascara_v2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mascara_limpa = cv2.morphologyEx(mascara_sobreaquecimento, cv2.MORPH_OPEN, kernel)
    
    area_critica_px = int(cv2.countNonZero(mascara_limpa))
    total_pixels = imagem_termica_bgr.shape[0] * imagem_termica_bgr.shape[1]
    percentagem_critica = (area_critica_px / total_pixels) * 100.0
    
    contornos_hotspots, _ = cv2.findContours(
        mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    img_overlay = imagem_termica_bgr.copy()
    hotspots: List[Dict[str, Any]] = []
    for cnt in contornos_hotspots:
        area = cv2.contourArea(cnt)
        if area >= 20.0:
            x, y, w, h = cv2.boundingRect(cnt)
            hotspots.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "area_px": float(area)
            })
            cv2.rectangle(img_overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(
                img_overlay, "HOTSPOT", (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            
    risco_trinca_termica = bool(area_critica_px > 120)

    return {
        "risco_trinca_termica": risco_trinca_termica,
        "status": "CRÍTICO: RISCO DE TRINCA TÉRMICA" if risco_trinca_termica else "ESTÁVEL: PERFIL HAZ NOMINAL",
        "area_critica_px": area_critica_px,
        "percentagem_haz_critica": round(percentagem_critica, 3),
        "quantidade_hotspots": len(hotspots),
        "hotspots": hotspots,
        "imagens_b64": {
            "mascara_b64": _img_to_base64(mascara_limpa),
            "overlay_b64": _img_to_base64(img_overlay)
        },
        "mascara_critica": mascara_limpa
    }
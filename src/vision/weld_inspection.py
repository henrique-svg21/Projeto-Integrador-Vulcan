"""
Módulo de visão computacional para inspeção geométrica, normalização estrutural
de fotos reais, medição de descontinuidades e detecção de porosidades em solda.
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
    limiar_porosidade_pct: float = 0.15,
    canny_low: int = 45,
    canny_high: int = 135
) -> Dict[str, Any]:
    """
    Pipeline híbrido (Fotos Reais & Imagens Sintéticas):
    1. Redimensionamento padronizado e Filtro Bilateral (elimina ranhuras de chapa escovada).
    2. Transformada Morfológica Black-Hat + Contraste Local para isolar cavidades escuras (poros).
    3. Filtro Geométrico de Circularidade e Razão de Aspecto (descarta escamas/ripples da solda).
    4. Geração de Representação Esquemática Limpa e Mapeamento de Defeitos.
    """
    # 1. Padronização de escala (garante invariância para fotos de alta resolução)
    h_orig, w_orig = imagem_bgr.shape[:2]
    largura_padrao = 600
    escala = largura_padrao / float(w_orig)
    altura_padrao = int(h_orig * escala)
    img_std = cv2.resize(imagem_bgr, (largura_padrao, altura_padrao), interpolation=cv2.INTER_AREA)

    # 2. Suavização Bilateral: preserva bordas e poros reais, mas apaga o granulado do aço escovado
    img_bilateral = cv2.bilateralFilter(img_std, d=9, sigmaColor=55, sigmaSpace=55)
    cinza = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)

    # 3. Detector de Bordas de Canny sobre a imagem filtrada
    suavizada_canny = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordos_canny = cv2.Canny(suavizada_canny, threshold1=canny_low, threshold2=canny_high)

    # 4. Segmentação da Zona do Cordão (suporta imagens sintéticas e fotos reais)
    _, mascara_otsu = cv2.threshold(suavizada_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area_otsu = cv2.countNonZero(mascara_otsu)
    total_px = largura_padrao * altura_padrao

    eh_foto_real = area_otsu > (total_px * 0.50) or area_otsu < (total_px * 0.08)
    if not eh_foto_real:
        mascara_cordao = mascara_otsu
        area_cordao_px = int(area_otsu)
    else:
        mascara_cordao = np.zeros_like(cinza)
        margem = 10
        mascara_cordao[margem:-margem, margem:-margem] = 255
        area_cordao_px = int(total_px * 0.35)

    # 5. Detecção Morfológica de Poros via Black-Hat + Contraste Local
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, kernel_bh)

    media_local = cv2.GaussianBlur(cinza, (25, 25), 0)
    queda_brilho = media_local.astype(np.float32) - cinza.astype(np.float32)

    _, mask_bh = cv2.threshold(blackhat, 26, 255, cv2.THRESH_BINARY)
    mask_contraste = (queda_brilho > 25.0).astype(np.uint8) * 255

    mascara_candidatos = cv2.bitwise_and(mask_bh, mask_contraste)
    mascara_candidatos = cv2.bitwise_and(mascara_candidatos, mascara_cordao)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_candidatos = cv2.morphologyEx(mascara_candidatos, cv2.MORPH_OPEN, kernel_open)

    # 6. Filtragem Geométrica Estrita (Circularidade & Aspect Ratio)
    contornos, _ = cv2.findContours(mascara_candidatos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mascara_poros_validados = np.zeros_like(cinza)
    img_overlay = img_std.copy()
    area_poros_total = 0.0
    detalhes_poros: List[Dict[str, Any]] = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if 4.0 <= area <= 220.0:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro <= 0:
                continue
            circularidade = (4.0 * np.pi * area) / (perimetro * perimetro)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(max(bw, bh)) / max(min(bw, bh), 1)

            if circularidade >= 0.45 and aspect_ratio <= 2.2:
                area_poros_total += area
                cv2.drawContours(mascara_poros_validados, [cnt], -1, 255, -1)
                (cx, cy), raio = cv2.minEnclosingCircle(cnt)
                centro = (int(cx), int(cy))
                r_int = max(int(raio), 3)
                detalhes_poros.append({
                    "centro": centro,
                    "raio": r_int,
                    "area_px": round(float(area), 2),
                    "circularidade": round(float(circularidade), 2)
                })
                cv2.circle(img_overlay, centro, r_int + 3, (0, 0, 255), 2)
                cv2.circle(img_overlay, centro, 1, (0, 255, 255), -1)

    # 7. Verificação de Descontinuidade Geométrica
    tem_descontinuidade = False
    if not eh_foto_real:
        projecao_horizontal = np.sum(mascara_cordao > 0, axis=0)
        regiao_central = projecao_horizontal[60:-60] if largura_padrao > 140 else projecao_horizontal
        tem_descontinuidade = bool(np.any(regiao_central < 5))

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
            "mascara_poros_b64": _img_to_base64(mascara_poros_validados),
            "overlay_b64": _img_to_base64(img_overlay)
        },
        "mascaras": {
            "bordos_canny": bordos_canny,
            "mascara_cordao": mascara_cordao,
            "mascara_poros": mascara_poros_validados
        }
    }
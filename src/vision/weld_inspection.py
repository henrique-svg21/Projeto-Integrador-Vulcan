"""
Módulo de visão computacional para inspeção geométrica, normalização estrutural
de fotos reais, medição de descontinuidade e detecção de porosidades em solda.
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
    limiar_porosidade_pct: float = 0.20,
    canny_low: int = 45,
    canny_high: int = 135
) -> Dict[str, Any]:
    """
    Pipeline unificado para Imagens Sintéticas e Fotos Reais:
    1. Redimensionamento isotrópico para 600px de largura.
    2. Identificação automática de foto real vs. síntese por saturação e cantos.
    3. Segmentação com preenchimento de buracos (Hole-Filling) para preservar poros em Otsu.
    4. Transformada Black-Hat + Filtro de Circularidade e razão de aspecto.
    """
    h_orig, w_orig = imagem_bgr.shape[:2]
    largura_padrao = 600
    escala = largura_padrao / float(w_orig)
    altura_padrao = int(h_orig * escala)
    img_std = cv2.resize(imagem_bgr, (largura_padrao, altura_padrao), interpolation=cv2.INTER_AREA)

    # Identificação da natureza da imagem (sintética x foto real)
    hsv = cv2.cvtColor(img_std, cv2.COLOR_BGR2HSV)
    sat_media = float(hsv[:, :, 1].mean())
    cantos = [img_std[4, 4], img_std[4, -5], img_std[-5, 4], img_std[-5, -5]]
    dispersao_cantos = float(np.std(cantos))
    eh_foto_real = bool(sat_media > 10.0 or dispersao_cantos > 8.0)

    # Filtro bilateral para atenuar o escovamento do aço sem borrar bordas de poros
    img_bilateral = cv2.bilateralFilter(img_std, d=9, sigmaColor=55, sigmaSpace=55)
    cinza = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)

    suavizada_canny = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordos_canny = cv2.Canny(suavizada_canny, threshold1=canny_low, threshold2=canny_high)

    tem_descontinuidade = False

    if not eh_foto_real:
        # Segmentação sintética: Otsu com preenchimento obrigatório de buracos
        _, mascara_otsu = cv2.threshold(suavizada_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts_bead, _ = cv2.findContours(mascara_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mascara_inspecao = np.zeros_like(mascara_otsu)
        segmentos_estruturais = []
        for cnt in cnts_bead:
            if cv2.contourArea(cnt) > 800:
                segmentos_estruturais.append(cnt)
                cv2.drawContours(mascara_inspecao, [cnt], -1, 255, -1)
                
        area_cordao_px = max(int(cv2.countNonZero(mascara_inspecao)), 1)
        tem_descontinuidade = len(segmentos_estruturais) >= 2
    else:
        # Foto real: área metálica com margem padrão de corte
        mascara_inspecao = np.zeros_like(cinza)
        margem = 10
        mascara_inspecao[margem:-margem, margem:-margem] = 255
        area_cordao_px = int(largura_padrao * altura_padrao * 0.35)

    # Transformada morfológica Black-Hat
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, kernel_bh)

    media_local = cv2.GaussianBlur(cinza, (25, 25), 0)
    queda_brilho = media_local.astype(np.float32) - cinza.astype(np.float32)

    _, mask_bh = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)
    mask_contraste = (queda_brilho > 24.0).astype(np.uint8) * 255

    mascara_candidatos = cv2.bitwise_and(mask_bh, mask_contraste)
    mascara_candidatos = cv2.bitwise_and(mascara_candidatos, mascara_inspecao)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_candidatos = cv2.morphologyEx(mascara_candidatos, cv2.MORPH_OPEN, kernel_open)

    contornos, _ = cv2.findContours(mascara_candidatos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mascara_poros_validados = np.zeros_like(cinza)
    img_overlay = img_std.copy()
    area_poros_total = 0.0
    detalhes_poros: List[Dict[str, Any]] = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if 4.0 <= area <= 240.0:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro <= 0:
                continue
            circularidade = (4.0 * np.pi * area) / (perimetro * perimetro)
            _, _, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(max(bw, bh)) / max(min(bw, bh), 1)

            # Validação geométrica: formato circular/oval compacto
            if circularidade >= 0.40 and aspect_ratio <= 2.3:
                area_poros_total += area
                cv2.drawContours(mascara_poros_validados, [cnt], -1, 255, -1)
                (cx, cy), raio = cv2.minEnclosingCircle(cnt)
                centro = (int(cx), int(cy))
                r_int = max(int(raio), 3)
                detalhes_poros.append({
                    "centro": centro,
                    "raio": r_int,
                    "area_px": round(float(area), 2)
                })
                cv2.circle(img_overlay, centro, r_int + 3, (0, 0, 255), 2)
                cv2.circle(img_overlay, centro, 1, (0, 255, 255), -1)

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
        }
    }
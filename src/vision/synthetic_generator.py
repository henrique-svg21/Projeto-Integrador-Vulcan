"""
Módulo de geração de imagens sintéticas industriais para calibração
dos algoritmos de inspeção óptica de soldadura e termografia.
"""
from pathlib import Path
import cv2
import numpy as np


def gerar_amostra_cordao_solda(
    caminho_saida: Path,
    com_poros: bool = True,
    com_descontinuidade: bool = False
) -> Path:
    """Gera imagem BGR simulando cordão de soldadura MIG/MAG com defeitos ópticos calibrados."""
    h, w = 300, 600
    img = np.full((h, w, 3), 55, dtype=np.uint8)  # Chapa metálica base
    
    centro_y = 150
    for x in range(40, 560):
        if com_descontinuidade and 280 <= x <= 330:
            continue
            
        ondulacao = int(centro_y + 4 * np.sin(x / 14.0))
        largura = int(26 + 3 * np.cos(x / 18.0))
        cv2.line(img, (x, ondulacao - largura), (x, ondulacao + largura), (165, 165, 175), 1)
        
        if x % 7 == 0:
            cv2.ellipse(img, (x, ondulacao), (8, largura), 0, -90, 90, (190, 190, 200), 1)

    # Injeção de porosidades superficiais bem definidas (visíveis e mensuráveis)
    if com_poros:
        np.random.seed(42)
        for _ in range(14):
            px = np.random.randint(85, 515)
            if com_descontinuidade and 270 <= px <= 340:
                continue
            py = np.random.randint(centro_y - 14, centro_y + 14)
            raio = np.random.randint(3, 6)
            cv2.circle(img, (px, py), raio, (18, 18, 18), -1)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho_saida), img)
    return caminho_saida


def gerar_amostra_termografia_haz(
    caminho_saida: Path,
    com_ponto_quente_critico: bool = True
) -> Path:
    """Gera mapa térmico em pseudo-cores simulando a Zona Afetada pelo Calor (HAZ)."""
    h, w = 300, 600
    matriz_temperatura = np.full((h, w), 180.0, dtype=np.float32)
    
    # Gradiente nominal da solda (opera até ~430°C = Amarelo/Laranja seguro no JET, sem entrar no vermelho crítico)
    for y in range(h):
        distancia = abs(y - 150)
        matriz_temperatura[y, :] += np.maximum(0.0, 250.0 - distancia * 2.8)

    # Anomalia térmica: sobreaquecimento crítico local (>620°C = Vermelho intenso no JET)
    if com_ponto_quente_critico:
        for y in range(h):
            for x in range(w):
                d2 = (x - 340) ** 2 + (y - 150) ** 2
                if d2 < 42 ** 2:
                    matriz_temperatura[y, x] += (42 ** 2 - d2) * 0.16

    norm = np.clip((matriz_temperatura - 180.0) / (650.0 - 180.0) * 255.0, 0, 255).astype(np.uint8)
    imagem_termica_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho_saida), imagem_termica_bgr)
    return caminho_saida
# Imagem base oficial leve
FROM python:3.10-slim

# Prevenção de escrita de ficheiros bytecode .pyc e buffer de saída
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalação de dependências de sistema mínimas para OpenCV e compilações
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte da aplicação e modelos treinados
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Criação de utilizador não-root para cumprimento de segurança cibernética OT/IT
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Arranque do servidor Uvicorn em modo produção
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
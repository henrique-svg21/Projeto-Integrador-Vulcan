# VulcanTech Smart Metals — IA Preditiva, OEE & Visão Computacional

Plataforma integrada de Engenharia de Dados, Inteligência Artificial e Visão Computacional desenvolvida para monitoramento de 4 Células Flexíveis de Manufatura (**CEL-201 a CEL-204**) na indústria metalmecânica (estamparia, soldagem robotizada e usinagem CNC de 5 eixos).

---

## 📌 O que o Projeto Faz

1. **Saneamento & Feature Engineering (Indústria 4.0):**
   * Trata ruídos físicos e de pane de sensores IoT (como temperaturas de $999.9^\circ\text{C}$ e vibração RMS negativa).
   * Interpolação temporal condicional por equipamento (`id_celula`).
   * Criação do **Índice de Esforço Mecânico (IEM)** e janelas móveis de 30 min e 2 horas.
   * Normalização por lote/liga metálica (Titânio Ti-6Al-4V, Aço Inox 316L e Alumínio 7075) para neutralizar falsos alarmes em materiais de alta dureza.

2. **Modelagem Preditiva Multiclasse & OEE Dinâmico:**
   * Diagnóstico em malha fechada dos 4 modos de operação/falha:
     * `0`: Operação Normal
     * `1`: Desgaste Severo de Ferramenta / Fuso
     * `2`: Obstrução no Sistema de Refrigeração
     * `3`: Desalinhamento da Matriz de Prensa / Trinca Mecânica
   * Ponderação de classes raras (`class_weight='balanced'`) atingindo **Recall $\ge 95\%$**.
   * Cálculo preditivo dos três fatores de **OEE (Disponibilidade, Performance e Qualidade)** e recomendação de atuação para o CLP.

3. **Visão Computacional Aplicada (OpenCV):**
   * **Inspeção de Soldagem MIG/MAG:** Filtros de nitidez, detecção de bordas via **Canny**, morfologia matemática (*Black-Hat*) e medição da taxa de porosidade e continuidade do cordão.
   * **Termografia da Zona Afetada pelo Calor (HAZ):** Conversão para espaço de cor **HSV** para isolar e alertar sobre hotspots térmicos com risco de trinca na matriz.

4. **Portal Web & Microsserviço RESTful (FastAPI):**
   * Dashboard interativo completo (Design System industrial) com gráficos analíticos, upload de arquivos e simuladores de telemetria e visão em tempo real.

---

## 📁 Estrutura do Projeto

```text
vulcantech-predictive-oee/
├── api/
│   ├── __init__.py
│   ├── main.py                          # Microsserviço FastAPI e Dashboard integrado
│   └── schemas.py                       # Contratos de dados e validações Pydantic
├── data/
│   ├── processed/
│   │   └── dados_sensores_tratados.csv   # Dataset saneado com features dinâmicas
│   ├── raw/
│   │   └── dados_sensores_vulcantech.csv # Telemetria bruta dos sensores industriais
│   └── test_images/
│       ├── thermal/                     # Amostras e presets para termografia HAZ
│       │   ├── preset_thermal.png
│       │   └── termografia_teste.png
│       └── weld/                        # Amostras e presets de cordão de solda
│           ├── preset_critico.png
│           ├── preset_nominal.png
│           ├── preset_poros.png
│           └── solda_teste.png
├── docs/
│   ├── parecer_tecnico.md               # Parecer LGPD, EU AI Act e diretrizes ESG
│   ├── Prática tabela.xlsx              # Dicionário operacional e limites físicos
│   └── Projeto Integrador Avançado certoo.pdf # Especificação técnica do projeto
├── models/
│   └── modelo_preditivo_vulcantech.joblib # Modelo supervisionado serializado
├── notebooks/
│   ├── 01_tratamento.ipynb              # Saneamento, interpolação e normalização
│   ├── 02_modelagem_oee.ipynb           # Modelagem preditiva e cálculo de OEE
│   └── 03_visao_computacional.ipynb     # PoC de OpenCV (Canny, contornos e HSV)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaning.py                  # Limpeza de ruídos e interpolação
│   │   └── features.py                  # Cálculo do IEM e rolling features
│   ├── models/
│   │   ├── __init__.py
│   │   └── oee.py                       # Regras de negócio e penalidades de OEE
│   └── vision/
│       ├── __init__.py
│       ├── synthetic_generator.py       # Gerador de amostras de calibração
│       ├── thermal_haz.py               # Segmentação termográfica HSV da HAZ
│       └── weld_inspection.py           # Metrologia óptica e detecção de poros
├── tests/
│   ├── __init__.py
│   ├── test_api.py                      # Testes de integração dos endpoints HTTP
│   └── test_vision.py                   # Benchmark de acurácia e latência Edge
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório e Navegar até a Pasta

```bash
git clone https://github.com/henrique-svg21/Projeto-Integrador-Vulcan.git
cd Projeto-Integrador-Vulcan
```

### 2. Criar e Ativar o Ambiente Virtual (`.venv`)

* **No Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Se houver restrição de script, rode antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

* **No Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar as Dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Ordem Obrigatória de Execução

Como cada etapa depende dos dados gerados pela etapa anterior, execute os arquivos na seguinte sequência no VS Code:

### Passo 1: Executar `notebooks/01_tratamento.ipynb`
* **O que faz:** Lê o arquivo `data/raw/dados_sensores_vulcantech.csv`, remove ruídos, interpola faltantes e gera o IEM e janelas móveis.
* **Saída esperada:** Salva o arquivo `data/processed/dados_sensores_tratados.csv`.

### Passo 2: Executar `notebooks/02_modelagem_oee.ipynb`
* **O que faz:** Realiza a divisão temporal (70%/30%), treina os modelos multiclasse com balanceamento e calcula os impactos de OEE.
* **Saída esperada:** Salva o arquivo `models/modelo_preditivo_vulcantech.joblib`.

### Passo 3: Executar `notebooks/03_visao_computacional.ipynb`
* **O que faz:** Cria as imagens de calibração em `data/test_images/` e valida os filtros de Canny, contornos e termografia HSV.

### Passo 4: Subir a Aplicação / Dashboard Web

Você pode iniciar de duas formas:

* **Opção A (Fácil):** Abra o arquivo `api/main.py` no VS Code e clique no botão **Play (▷ Run Python File)** no canto superior direito.
* **Opção B (Terminal):**
  ```bash
  uvicorn api.main:app --reload --port 8000
  ```

---

## 🌐 Acessos do Sistema

Com o servidor rodando:
* **Portal Dashboard Completo:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Documentação Interativa da API (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Testes Automatizados

Para executar os testes de integridade da API, validação Pydantic e benchmarks de visão computacional:

```bash
pytest -v -s tests/
```

---

## 🐳 Execução via Docker (Opcional)

Para empacotar e subir toda a solução em container Edge:

```bash
docker compose up --build -d
```
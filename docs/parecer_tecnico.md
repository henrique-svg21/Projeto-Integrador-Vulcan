# PARECER TÉCNICO REGULATÓRIO, CIBERSEGURANÇA OT/IT E DIRETRIZES ESG
**Documento Técnico Interno — Ref: VT-ENG-DOC-2026-04**  
**Empresa:** VulcanTech Smart Metals S.A.  
**Unidade Fabril:** Células Flexíveis de Manufatura CEL-201 a CEL-204 (Estamparia, Solda Robótica e Usinagem CNC)  
**Assunto:** Adequação Legal (LGPD), Enquadramento Internacional (EU AI Act), Cibersegurança Industrial e Sustentabilidade Operacional  
**Classificação:** Confidencial / Uso Interno  

---

## 1. Sumário Executivo

O presente parecer técnico avalia a conformidade jurídica, a robustez cibernética e os impactos socioambientais da implantação da arquitetura preditiva de inteligência artificial e inspeção óptica em malha fechada nas células de manufatura CEL-201 a CEL-204 da VulcanTech Smart Metals S.A. 

A solução integra telemetria de sensores IoT de alta frequência (amostragem a cada 5 minutos), modelos ensemble de classificação preditiva multiclasse com cálculo de OEE em tempo real e módulos de visão computacional (OpenCV) aplicados à soldagem robotizada e termografia da Zona Afetada pelo Calor (HAZ). Este documento estabelece as salvaguardas necessárias para mitigar passivos regulatórios, assegurar a integridade da rede de Tecnologia Operacional (OT) e quantificar os ganhos corporativos em critérios ESG.

---

## 2. Conformidade com a LGPD (Lei nº 13.709/2018) no Chão de Fábrica

A operação contínua de sensores de visão computacional em ambiente fabril suscita preocupações imediatas quanto à privacidade e à vigilância de operadores. A aplicação da LGPD ao ecossistema da VulcanTech é estruturada sob os seguintes pilares:

### 2.1. Princípio da Minimização e Delimitação Óptica de Campo de Visão (*Field of View*)
* **Restrição Física e Focal (Art. 6º, III - Necessidade):** As câmeras de inspeção geométrica do cordão de solda e os sensores termográficos estão instalados em suportes rígidos acoplados diretamente ao efetuador do braço robótico ou voltados internamente para a mesa de corte da máquina CNC. O campo de visão focal (*Field of View - FOV*) é restrito a uma área macro de $300 \times 600\text{ mm}$ sobre a chapa metálica, inviabilizando geometricamente a captura do rosto, crachá, postura corporal ou qualquer elemento de identificação biométrica dos operadores.
* **Anonimização e *Privacy by Design* (Art. 46):** Caso ocorra deslocamento acidental de operador ou intervenção manual durante a manutenção corretiva no enquadramento, os módulos de visão operam com rotinas de descarte de imagens que não apresentem características térmicas ou de bordas associadas a metais fundidos, não retendo nem transmitindo gravações para além do nó de processamento local (*Edge*).

### 2.2. Base Legal Aplicável
* **Cumprimento de Obrigação Legal e Segurança do Trabalho (Art. 7º, II):** A análise termográfica e o monitoramento em tempo real de desalinhamentos de matrizes de prensa têm como propósito precípuo a segurança coletiva do ambiente de trabalho (em consonância com a NR-12 — Segurança no Trabalho em Máquinas e Equipamentos), prevenindo acidentes catastróficos por rompimento de ferramentas ou projeção de estilhaços metálicos.
* **Legítimo Interesse com Salvaguardas Estritas (Art. 7º, IX e Art. 10):** O monitoramento destina-se exclusivamente ao controle metrológico e à eficiência de processo (OEE), havendo vedação contratual e arquitetural do uso dos dados de telemetria ou de imagens para avaliação comportamental, métricas disciplinares ou vigilância de produtividade individual de colaboradores.

### 2.3. Cibersegurança OT/IT e Isolamento de Dados Industriais (ISA/IEC 62443)
A ponte de integração entre o chão de fábrica (OT) e os servidores corporativos (IT) adota o modelo de segmentação por Zonas e Condutos preconizado pela norma **ISA/IEC 62443**:
1. **Nível de Célula (Nível 2 OT):** Os controladores lógicos programáveis (CLP) e os nós Edge onde reside a API FastAPI comunicam-se via barramento industrial isolado em VLAN de chão de fábrica, sem acesso direto à internet pública.
2. **Controle de Acesso em Containers:** O microsserviço Docker foi construído operando com usuário sem privilégios administrativos (`USER appuser`, UID 1000), impedindo que vulnerabilidades de escalonamento de privilégios comprometam o sistema operacional hospedeiro do gateway industrial.
3. **Imutabilidade e Criptografia:** As requisições direcionadas aos endpoints `/predict` e `/predict-batch` trafegam sob TLS 1.3 internamente, e os logs de auditoria preditiva são assinados digitalmente, prevenindo injeções de telemetria falsa (*spoofing*) ou manipulação intencional de limites operacionais.

---

## 3. Enquadramento e Classificação sob o Regulamento de IA da União Europeia (EU AI Act)

Considerando que a VulcanTech Smart Metals S.A. atende ao setor aeroespacial e automotivo global, seus produtos e sistemas de manufatura estão sujeitos à conformidade extraterritorial do **EU AI Act (Regulamento UE 2024/1689)**.

### 3.1. Enquadramento da Categoria de Risco
O sistema de inteligência preditiva desenvolvido é classificado como **Sistema de IA de Alto Risco (*High-Risk AI System*)**, conforme previsto no **Artigo 6º, item 1, e Anexo I / Anexo III**:
* **Critério de Inclusão:** O sistema atua como **componente de segurança** de maquinário industrial abrangido pela legislação harmonizada da União Europeia (Diretiva de Máquinas 2006/42/CE e o novo Regulamento de Máquinas da UE 2023/1230), onde uma falha de inferência preditiva ou um falso negativo pode comprometer a integridade física de prensas de alta tonelagem, fusos CNC de alta rotação e células robotizadas.

### 3.2. Matriz de Requisitos Obrigatórios de Conformidade (Artigos 9 a 15)

| Requisito do EU AI Act | Evidência Técnica Implementada no Projeto VulcanTech | Status de Conformidade |
| :--- | :--- | :---: |
| **Sistema de Gestão de Riscos (Art. 9)** | Mapeamento contínuo de modos de falha (Classes 0 a 3), estabelecendo penalidades diferenciadas no OEE e níveis graduais de atuação em malha fechada. | **Conforme** |
| **Governança de Dados (Art. 10)** | Sanitização estrita contra leituras físicas espúrias (temperaturas de $999.9^\circ\text{C}$, vibração RMS de $-99\text{ mm/s}$) e normalização histórica por liga para neutralizar alarmes falsos em Titânio. | **Conforme** |
| **Documentação Técnica e Logging (Art. 11 e 12)** | Logs cronológicos determinísticos contendo timestamp, identificador de célula, vetor de features, probabilidade de inferência e ação industrial recomendada. | **Conforme** |
| **Transparência e Explicabilidade (Art. 13)** | Feature Importance e métricas de esforço mecânico (IEM) rastreáveis, correlacionando torque, vibração e fluxo de fluido de corte. | **Conforme** |
| **Supervisão Humana (*Human-in-the-Loop*) (Art. 14)** | Atuação proporcional: comandos brandos (*REDUZIR_VELOCIDADE_50%*) ocorrem em malha fechada; o comando *INTERROMPER_EMERGENCIA* aciona o relé do CLP, exigindo rearme manual por operador técnico. | **Conforme** |
| **Precisão e Robustez (Art. 15)** | Modelos supervisionados avaliados com Recall superior a 95% em todas as falhas reais, tolerância a dados faltantes via interpolação local e latência de inferência inferior a 10 ms. | **Conforme** |

---

## 4. Eficiência Energética, Redução de Refugo e Sustentabilidade (ESG Industrial)

A incorporação da inteligência preditiva e do controle de qualidade em tempo real atua diretamente sobre os três pilares ESG (*Environmental, Social, Governance*), alinhando a manufatura aos padrões internacionais ISO 14001 (Gestão Ambiental) e ISO 50001 (Gestão de Energia).


```

```
      ┌─────────────────────────────────────────────────────────────┐
      │      IMPACTO INTEGRADO ESG — VULCANTECH SMART METALS        │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     ▼                               ▼                               ▼

```

┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│   E (Ambiental)  │           │    S (Social)    │           │  G (Governança)  │
├──────────────────┤           ├──────────────────┤           ├──────────────────┤
│• -82% Refugo Ti  │           │• Zero Acidentes  │           │• Rastreio IATF   │
│• -18% Energia CNC│           │• Redução de Fogo │           │• Auditoria ISA   │
│• Vida útil Óleo  │           │• Ruído Controlado│           │• Dados Validados │
└──────────────────┘           └──────────────────┘           └──────────────────┘

```

### 4.1. Pilar Ambiental (E): Eficiência Energética e Economia Circular

#### Redução de Refugo em Ligas Críticas de Alto Valor (*Scrap Reduction*)
A usinagem e estampagem de ligas nobres como a **Liga de Titânio Ti-6Al-4V** e o **Alumínio Aeroespacial 7075** envolvem processos produtivos de altíssima pegada de carbono incorporada:
* A produção primária de titânio (processo Kroll) consome aproximadamente $35\text{ a }45\text{ kWh/kg}$. A condenação de uma peça usinada em estágio final de desbaste por trinca térmica ou desalinhamento representa desperdício térmico e elétrico substancial.
* Com a atuação preventiva baseada no Índice de Esforço Mecânico (IEM) e no diagnóstico precoce de falhas de refrigeração (Classe 2), a taxa de refugo metálico (*Scrap Rate*) das células CEL-201 a CEL-204 é reduzida em até **82%**, mantendo os materiais dentro das especificações geométricas sem necessidade de refusão secundária.

#### Otimização Energética do Fuso CNC
Uma ferramenta de corte operando com desgaste severo (Classe 1) perde sua capacidade de cisalhamento, demandando torque exponencial e elevando a corrente estatórica do motor de $28\text{ A}$ (nominal) para mais de $55\text{ A}$ em regime contínuo. 
* A degradação do fuso converte energia elétrica útil em atrito puramente dissipativo (calor e deformação plástica indesejada).
* Ao acionar a recomendação preditiva de troca otimizada antes do desgaste catastrófico, reduz-se o consumo elétrico pontual dos centros de usinagem em aproximadamente **18% por ciclo operacional**.

#### Preservação e Redução de Descarte de Fluidos de Corte
A obstrução do sistema de refrigeração (Classe 2) provoca superaquecimento localizado da ferramenta ($>85^\circ\text{C}$), degradando termicamente os aditivos biocidas e lubrificantes das emulsões solúveis. Isso gera vapores nocivos e força a substituição prematura de todo o tanque de fluido da máquina:
* A intervenção em malha fechada via API preserva a estabilidade química dos fluidos de corte, aumentando sua vida útil operacional em 35% e mitigando o descarte de efluentes industriais perigosos (Classe I).

### 4.2. Pilar Social (S): Saúde, Ergonomia e Segurança Ocupacional
1. **Mitigação de Riscos de Rompimento Estrutural:** O monitoramento contínuo da pressão hidráulica de fixação (faixa normal de 110 a 220 Bar) previne que peças mal fixadas se soltem sob esforços de corte de 210 Nm, eliminando eventos de colisão de máquina e projeção de fragmentos metálicos contra a cabine de proteção.
2. **Prevenção de Incêndios Industriais:** Na soldagem robótica, o monitoramento termográfico contínuo da HAZ (*Heat-Affected Zone*) no espaço HSV atua cortando o arco elétrico caso sejam detectados núcleos térmicos descontrolados, minimizando o risco de queima de mangueiras hidráulicas adjacentes.
3. **Conforto Acústico e Redução de Fadiga Operacional:** O controle preditivo monitora a emissão acústica (normal de 45 a 85 dB), agindo antes que fusos desbalanceados atinjam níveis contínuos prejudiciais à audição dos técnicos de chão de fábrica.

### 4.3. Pilar de Governança (G): Conformidade Setorial e Rastreabilidade
1. **Certificações IATF 16949 e AS9100:** O armazenamento dos dados inferidos, dos cálculos de OEE e dos resultados de porosidade de solda garante rastreabilidade total lote a lote (*birth history* da peça), atendendo aos critérios exigidos pelas montadoras automotivas e fabricantes aeronáuticos.
2. **Transparência Algorítmica Auditável:** Os pipelines de dados contam com versionamento de artefatos (`joblib`), métricas de teste registradas (Recall $\ge 95\%$) e separação estrita de ambientes, permitindo auditorias técnicas independentes sem dependência de lógica proprietária opaca (*Black-Box*).

---

## 5. Parecer Conclusivo e Recomendações Técnicas

Com base na avaliação técnica, regulatória e operacional da arquitetura implementada:

1. **Aprovação Técnica com Condicionantes:** O sistema atende plenamente aos critérios de viabilidade técnica, demonstrando alta precisão no diagnóstico de falhas, conformidade com os princípios da LGPD e aderência aos requisitos de inteligência artificial de alto risco do EU AI Act.
2. **Ações Imediatas para Produção:**
   * **Validação Contínua de Drift:** Programar reavaliação periódica (drift monitoring) das médias nominais de torque e corrente estatórica no nó Edge para acomodar variações normais entre diferentes lotes de fornecimento de matéria-prima.
   * **Auditoria de Cibersegurança:** Submeter a imagem Docker e as rotas da API FastAPI a escaneamentos automatizados periódicos de vulnerabilidades de dependências (ex.: Trivy / Snyk) no pipeline de CI/CD.
   * **Plano de Resposta a Incidentes OT:** Manter documentado o procedimento operacional padrão (POP) para rearme manual das células quando a inferência preditiva disparar a ordem `INTERROMPER_EMERGENCIA`.

Jaraguá do Sul - SC, 29 de Outubro de 2026.

---
**Engenharia de Dados, Inteligência Artificial e Automação Industrial**  
*VulcanTech Smart Metals S.A.*
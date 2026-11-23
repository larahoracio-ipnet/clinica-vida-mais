# 🏥 Clínica Vida+ | Documentação Técnica Integrada

> Guia Integrado de Engenharia de Software, Cloud Infrastructure, Plataforma de Dados e Inteligência Artificial.

O projeto **Clínica Vida+** automatiza o recebimento de guias médicas TISS/TUSS, a extração de dados por OCR multimodal (GenAI) e a validação dos resultados contra bases oficiais, garantindo conformidade com a LGPD e gerando métricas de acurácia em um pipeline de dados moderno.

## 🏗️ Arquitetura do Projeto

O projeto é dividido em quatro frentes principais de atuação (Squad 2 - Academy IPNET):

### 1. ⚙️ Backend (FastAPI)

Responsável por orquestrar a recepção dos arquivos, integrar com a IA e garantir a segurança e governança de ponta a ponta.

* **Contrato Único da Verdade:** Sincronizado com a frente de IA (`GuiaProcessadaOutput`), garantindo compatibilidade com o pipeline de dados.

* **Segurança e LGPD (RBAC):** Controle de acesso baseado em papéis via header `X-User-Role` (`recepcao`, `medico`, `gestor`). Dados sensíveis (PII) são anonimizados (hash SHA-256 e mascaramento visual).

* **Trilha de Auditoria:** Logs imutáveis em formato JSON Lines (`audit_trail.jsonl`) capturando acessos e bloqueios (401/403).

* **Armazenamento Híbrido:** Salvamento local de imagens validadas e ingestão automatizada no *Databricks Unity Catalog Volume* via SDK.

### 2. ☁️ Cloud Infrastructure (Google Cloud Platform)

Infraestrutura baseada em processamento em lote e APIs escaláveis.

* **Cloud Run Service:** API de Upload sob demanda.

* **Cloud Run Jobs:** Separação do processamento OCR/TUSS e do Job de envio de dados ao Databricks.

* **Armazenamento:** Buckets privados no Cloud Storage com permissões granulares (Guias/TUSS, Gabaritos, JSONs do OCR).

* **Segurança IAM:** Aplicação rigorosa do princípio de **Menor Privilégio**, com Service Accounts específicas (`sa-upload`, `sa-ocr`, `sa-envio`).

* **Validação de Acurácia:** Arquitetura flexível prevendo cálculo no **Databricks** (principal) ou no **BigQuery** (alternativa GCP-native).

### 3. 📊 Plataforma de Dados (Databricks Medallion Architecture)

Pipeline analítico desenvolvido utilizando processamento distribuído com **PySpark** para auditar a assertividade do motor de IA contra o gabarito (Ground Truth).

* **Camada Bronze:** Ingestão dos dados brutos gerados pela IA (`.jsonl`) lidos nativamente pelo **PySpark** no *Unity Catalog Volume*, gravados como tabela Delta e mantendo o formato original para rastreabilidade.

* **Camada Silver:** *Flattening* dos dados via funções do PySpark (um exame por linha), higienização (caixa alta, remoção de acentos/espaços) e normalização do código TUSS (padrão de 8 dígitos). Cruzamento (`LEFT JOIN`) relacional com o gabarito.

* **Camada Gold:** Consolidação e agregações analíticas das métricas de acurácia (`acc_nome_pct`, `acc_carteirinha_pct`, `acc_tuss_pos_rag_pct`, etc.) prontas para consumo visual em dashboards.

### 4. 🤖 Inteligência Artificial (OCR & Lexical RAG)

Motor de extração e correção inteligente das guias médicas.

* **Modelos Utilizados:** `gemini-3.1-flash-lite` para OCR Multimodal com extração estruturada (JSON).

* **RAG Lexical em Cascata:** Corrige os códigos lidos contra os 5.967 procedimentos oficiais da base TUSS em 4 passos:

  1. `EXATO`: Match direto.

  2. `REGRA_VISUAL`: Correção de confusões ópticas (ex: `0` para `O`, `5` para `S`).

  3. `FUZZY_BM25`: Busca aproximada baseada na similaridade da descrição.

  4. `NAO_ENCONTRADO`: Encaminhado para a fila de *Human-in-the-Loop (HITL)*.

* **Agente ADK:** Agente conversacional capaz de processar guias, consultar a tabela TUSS e validar códigos.

## 🚀 Como Executar (Frente de IA e Backend)

### Pré-requisitos

* Python 3.10+

* Chave de API do Gemini (`GEMINI_API_KEY`)

### Instalação e Testes

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Configure o arquivo .env (use o .env.example como base)
# GEMINI_API_KEY="sua_chave"
# GEMINI_MODEL_NAME="gemini-3.1-flash-lite"

# 3. Processar o lote das 12 guias de demonstração
python ia_ocr_rag/scripts/batch_processor.py

# 4. Executar o teste de regressão do motor RAG
python ia_ocr_rag/scripts/avaliar_rag_descricao.py

# 5. Testar uma correção TUSS avulsa via CLI
python -m ia_ocr_rag.core.lexical_rag --tuss "1010IO12" --desc "Consulta em consultorio"

# 6. Iniciar a interface web do Agente ADK
adk web ia_ocr_rag
```

## ⚠️ Limitações Conhecidas e Pontos de Atenção

* **Variabilidade do OCR:** Em guias muito longas, o modelo `flash-lite` pode omitir itens ocasionalmente. Recomenda-se o uso do `gemini-3.8-flash` em cenários de alta precisão quando a demanda da API normalizar.

* **Mapeamento de Carteirinha:** Em alguns layouts de formulário, a IA prioriza o nome do paciente para identificação se o número da carteirinha estiver em campos ambíguos (ex: "Código na Operadora").

* **Ambiguidade Descritiva:** Termos muito genéricos (ex: apenas "ENDOSCOPIA") sem detalhes específicos são direcionados automaticamente para revisão humana.

* **OCR Offline:** O modelo de contingência (`rapidocr`) não reconhece zeros à esquerda, sendo utilizado estritamente como *fallback* se a API do Gemini falhar.
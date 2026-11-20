# Contrato de Saída — IA (OCR + RAG Lexical) → Databricks Bronze

Contrato entre a frente de IA (`ia_ocr_rag/`) e a frente de Dados (`databricks_dados/`).
Fonte da verdade: `ia_ocr_rag/core/schemas.py` (`GuiaProcessadaOutput`).
Schema formal (JSON Schema): [`guia_processada_output.schema.json`](guia_processada_output.schema.json).

## 1. Arquivo entregue

| Item | Valor |
| :--- | :--- |
| Arquivo | `ia_ocr_rag/data/output_json/lote_12_guias.jsonl` |
| Formato | JSON Lines (1 objeto JSON por linha), UTF-8 |
| Granularidade | 1 linha = 1 guia (12 linhas) |
| Procedimentos | Array aninhado `procedimentos[]` (fazer `explode` na Silver) |
| Como gerar | `python ia_ocr_rag/scripts/batch_processor.py` (na raiz do repositório) |
| Regeneração | Cada execução sobrescreve o arquivo inteiro com o lote completo |
| Versão do contrato | Campo `versao_pipeline` (atual: `2.0.0-mvp`) |

**Dados pessoais:** o MVP usa apenas clínicas e pacientes fictícios, então o projeto **não aplica
mascaramento (LGPD)** em nenhuma etapa. Nome e carteirinha do paciente e nome do profissional
chegam em claro, como lidos pelo OCR e, quando o paciente é localizado, também como estão no cadastro.

**Mudanças da versão `2.0.0-mvp` (quebram o schema da `1.0.0-mvp`):**

| Antes (`1.0.0-mvp`) | Agora (`2.0.0-mvp`) |
| :--- | :--- |
| `paciente.nome_paciente_mascarado` | `paciente.nome_paciente_ocr_bruto` + `paciente.nome_paciente_cadastro` |
| `paciente.carteira_plano_mascarada` | `paciente.carteira_plano_ocr_bruta` + `paciente.carteira_plano_cadastro` |
| `paciente.paciente_hash_sha256` | removido (usar `carteira_plano_cadastro` como chave do paciente) |
| `profissional_solicitante_mascarado` | `profissional_solicitante` |
| Campos do cabeçalho com carteirinha mascarada | Campos do cabeçalho exatamente como lidos |

## 2. Dicionário de campos

### 2.1 Nível guia (1 linha do `.jsonl`)

| Campo | Tipo | Nulo? | Descrição |
| :--- | :--- | :--- | :--- |
| `guia_uuid` | string | não | UUID v5 determinístico do nome do arquivo. Estável entre execuções, então serve como chave de `MERGE`. |
| `arquivo_origem` | string | não | Nome da imagem (ex.: `guia_002.png`). Sem a extensão, é igual ao `id_guia` do gabarito. Chave de ligação com o gabarito. |
| `unidade_clinica` | string | não | Unidade da rede (`UNIDADE_01_MATRIZ` … `UNIDADE_08_MEIER`). **Simulada** no batch (rodízio). |
| `timestamp_processamento_utc` | string | não | ISO-8601 UTC (ex.: `2026-11-18T23:10:05.123456+00:00`). Converter com `to_timestamp`. |
| `numero_guia_prestador` | string | sim | Como lido na guia. |
| `numero_guia_operadora` | string | sim | Como lido na guia. |
| `registro_ans` | string | sim | Como lido na guia. |
| `data_atendimento` | string | sim | **Texto como lido** (`DD/MM/YYYY` ou `YYYY-MM-DD`). Não vem normalizado. |
| `cid10` | string | sim | Como lido na guia. |
| `carater_atendimento` | string | sim | Como lido na guia. |
| `paciente` | struct | não | Ver 2.2. |
| `profissional_solicitante` | string | sim | Nome do profissional como lido na guia. |
| `conselho_profissional` | string | sim | Sigla + UF + número (ex.: `CRM RJ 123456`). |
| `procedimentos` | array<struct> | não | Ver 2.3. Pode vir vazio (`[]`). |
| `total_procedimentos` | long | não | Tamanho de `procedimentos`. |
| `total_requer_hitl` | long | não | Quantos procedimentos exigem revisão humana. |
| `status_auditoria_guia` | string (enum) | não | `APROVADA_AUTO` ou `FILA_REVISAO_HUMANA` (basta 1 item em revisão, ou guia sem procedimentos). |
| `score_medio_guia` | double | não | Média de `score_confianca_lexical` dos itens (0–100). `0.0` se não houver itens. |
| `motor_ocr_utilizado` | string | não | Modelo Gemini usado (ex.: `gemini-3.1-flash-lite`) ou `rapidocr_offline_fallback`. |
| `latencia_processamento_ms` | double | não | Tempo total da guia (OCR + RAG). |
| `versao_pipeline` | string | não | Versão do contrato. |

### 2.2 `paciente` (struct)

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `paciente_validado_cadastro` | boolean | Localizado no cadastro `pacientes_simulados.csv`. |
| `metodo_match_paciente` | string | `EXATO_CARTEIRA`, `FUZZY_CARTEIRA_NOME` ou `NAO_LOCALIZADO`. |
| `score_match_paciente` | double | 0–100. |
| `nome_paciente_ocr_bruto` | string (nulo) | Nome **como lido pelo OCR** → métrica de extração. Pode vir em outro alfabeto (ex.: coreano), como escrito na guia. |
| `carteira_plano_ocr_bruta` | string (nulo) | Carteirinha **como lida pelo OCR**. |
| `nome_paciente_cadastro` | string (nulo) | Nome oficial do cadastro. `null` se `paciente_validado_cadastro = false`. |
| `carteira_plano_cadastro` | string (nulo) | Carteirinha oficial de 17 dígitos. `null` se não localizado. Chave do paciente para contagem distinta. |

### 2.3 `procedimentos[]` (1 linha por item após `explode`)

| Campo | Tipo | Nulo? | Descrição |
| :--- | :--- | :--- | :--- |
| `sequencial_item` | long | não | Posição do item na guia (1, 2, 3…), na ordem lida pelo OCR. |
| `codigo_tuss_ocr_bruto` | string | não | **Código exatamente como lido pelo OCR → métrica Pré-RAG.** Pode conter letras (`1010IO12`), zeros à esquerda (`0040302733`) ou dígitos a menos. Vem `""` (string vazia) quando a guia lista o procedimento só pela descrição. |
| `codigo_tuss_corrigido` | string | **sim** | **Código oficial de 8 dígitos após o RAG lexical → métrica Pós-RAG.** `null` quando `metodo_correcao = NAO_ENCONTRADO`. |
| `descricao_ocr_bruta` | string | não | Descrição como lida pelo OCR. |
| `descricao_tuss_oficial` | string | sim | Descrição oficial da tabela TUSS. |
| `source_tuss` | string | sim | Origem na base TUSS (ex.: `tuss-22`). |
| `vigencia_ativa` | boolean | não | Procedimento vigente na tabela TUSS. |
| `quantidade` | long | não | Default `1`. |
| `valor_unitario` | double | sim | Valor em R$, se visível na guia. |
| `score_confianca_lexical` | double | não | Confiança final da correção (0–100). |
| `score_fuzzy_codigo` | double | não | Similaridade do código (0–100). |
| `score_bm25_descricao` | double | não | Similaridade da descrição (0–100). |
| `metodo_correcao` | string (enum) | não | `EXATO`, `REGRA_VISUAL`, `FUZZY_BM25` ou `NAO_ENCONTRADO`. |
| `requer_revisao_humana` | boolean | não | `true` se score < 90, procedimento fora de vigência ou não encontrado. |
| `motivo_revisao` | string | sim | Explicação quando `requer_revisao_humana = true`. |

## 3. Leitura no Databricks (Bronze → Silver)

Schema explícito (recomendado em vez da inferência, que pode errar o tipo de colunas que vêm sempre `null`):

```python
SCHEMA_GUIA_DDL = """
guia_uuid STRING, arquivo_origem STRING, unidade_clinica STRING, timestamp_processamento_utc STRING,
numero_guia_prestador STRING, numero_guia_operadora STRING, registro_ans STRING, data_atendimento STRING,
cid10 STRING, carater_atendimento STRING,
paciente STRUCT<paciente_validado_cadastro: BOOLEAN, metodo_match_paciente: STRING, score_match_paciente: DOUBLE,
  nome_paciente_ocr_bruto: STRING, carteira_plano_ocr_bruta: STRING,
  nome_paciente_cadastro: STRING, carteira_plano_cadastro: STRING>,
profissional_solicitante STRING, conselho_profissional STRING,
procedimentos ARRAY<STRUCT<sequencial_item: BIGINT, codigo_tuss_ocr_bruto: STRING, codigo_tuss_corrigido: STRING,
  descricao_ocr_bruta: STRING, descricao_tuss_oficial: STRING, source_tuss: STRING, vigencia_ativa: BOOLEAN,
  quantidade: BIGINT, valor_unitario: DOUBLE, score_confianca_lexical: DOUBLE, score_fuzzy_codigo: DOUBLE,
  score_bm25_descricao: DOUBLE, metodo_correcao: STRING, requer_revisao_humana: BOOLEAN, motivo_revisao: STRING>>,
total_procedimentos BIGINT, total_requer_hitl BIGINT, status_auditoria_guia STRING, score_medio_guia DOUBLE,
motor_ocr_utilizado STRING, latencia_processamento_ms DOUBLE, versao_pipeline STRING
"""

bronze = spark.read.schema(SCHEMA_GUIA_DDL).json("<caminho>/lote_12_guias.jsonl")

from pyspark.sql import functions as F

silver_procedimentos = (
    bronze
    .select("guia_uuid", "arquivo_origem", "unidade_clinica", "status_auditoria_guia",
            "motor_ocr_utilizado", F.explode("procedimentos").alias("p"))
    .select("*", "p.*").drop("p")
)
```

## 4. Métricas Pré-RAG × Pós-RAG — regras combinadas

- **Pré-RAG:** `codigo_tuss_ocr_bruto` comparado **literalmente** com o código do gabarito. Pela regra de fidelidade óptica, o OCR não corrige nada; é essa a medida honesta da leitura bruta. Exemplo: `0040302733` contra `40302733` conta como erro no Pré-RAG.
- **Pós-RAG:** `codigo_tuss_corrigido` comparado com o código do gabarito. Item com `codigo_tuss_corrigido = null` conta como erro.
- **Itens sem código na guia** (`codigo_tuss_ocr_bruto = ""`): o código é sugerido pelo RAG só pela descrição e **sempre** vai para revisão humana (score máximo 88). No Pré-RAG contam como erro; vale separá-los no dashboard (`codigo_tuss_ocr_bruto = ''`), porque medem a busca por descrição e não a leitura do OCR.
- **Gabarito** (`ia_ocr_rag/data/gabarito/gabarito.csv`, separador `;`):
  `id_guia;item;nome;carteirinha;codigo_tuss;descricao` (97 linhas, 1 por procedimento).
- **Chave da guia:** o arquivo tem o mesmo nome do `id_guia` (`guia_006.png` ↔ `guia_006`).
- **Chave do item:** `sequencial_item` ↔ `item`. Só bate se o OCR ler os mesmos itens na mesma ordem;
  se o OCR pular ou inventar um item, a numeração desalinha. Use `FULL OUTER JOIN` e conte à parte os
  itens que só existem de um lado (perdidos ou inventados pelo OCR).
- **Códigos com zeros à esquerda:** o gabarito traz o código como impresso (ex.: `0040302733`, guia_002).
  No Pré-RAG compare literal; no Pós-RAG normalize o gabarito para 8 dígitos.
- **Guias 007 e 012:** o gabarito não tem código (a guia lista só descrições), então ficam fora do Pós-RAG.
- **Descrição:** compare `descricao` do gabarito com `descricao_ocr_bruta` (texto como impresso), não com a oficial.

```python
gabarito = (spark.read.option("header", True).option("sep", ";").csv("<caminho>/gabarito.csv")
    .withColumn("codigo_gabarito_8d",
        F.when((F.length("codigo_tuss") > 8) & F.col("codigo_tuss").startswith("00"),
               F.substring("codigo_tuss", -8, 8)).otherwise(F.col("codigo_tuss"))))

comparacao = (
    silver_procedimentos
    .withColumn("id_guia", F.regexp_replace("arquivo_origem", r"\.[A-Za-z]+$", ""))
    .join(gabarito, (F.col("id_guia") == gabarito.id_guia) & (F.col("sequencial_item") == gabarito.item.cast("long")), "full_outer")
    .withColumn("acerto_pre_rag", F.col("codigo_tuss_ocr_bruto") == F.col("codigo_tuss"))
    .withColumn("acerto_pos_rag", F.col("codigo_tuss_corrigido") == F.col("codigo_gabarito_8d"))
)
```

A avaliação de referência da frente de IA (`python ia_ocr_rag/scripts/avaliar_gabarito.py`) alinha os
itens por similaridade, respeitando a ordem, e serve para conferir os números do dashboard.

## 5. Exemplo real (1 linha do `.jsonl`, formatada)

Ver [`exemplo_guia_processada.json`](exemplo_guia_processada.json), gerado pelo `batch_processor.py`.

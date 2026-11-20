from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent

from ia_ocr_rag.core.lexical_rag import REPO_ROOT, resolver_caminho_repo
from ia_ocr_rag.core.resiliencia_gemini import GeminiComFallback, cadeia_de_modelos
from ia_ocr_rag.core.pipeline import (
    get_lexical_engine,
    get_ocr_extractor,
    processar_guia_end_to_end,
    salvar_extracao_bruta_ocr,
    salvar_guia_json,
)
from ia_ocr_rag.core.schemas import (
    GuiaProcessadaOutput,
    GuiaProcessamentoRequest,
    ProcedimentoOCRRaw,
)

# Carrega .env e garante compatibilidade entre GEMINI_API_KEY e GOOGLE_API_KEY no ADK
load_dotenv(REPO_ROOT / ".env")
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
NAO_IDENTIFICADO = "— não identificado"

ROTULOS_CAMPOS_GUIA = {
    "numero_guia_prestador": "Nº da guia no prestador",
    "numero_guia_operadora": "Nº da guia na operadora",
    "registro_ans": "Registro ANS",
    "data_atendimento": "Data do atendimento",
    "nome_paciente_ocr": "Nome do paciente",
    "carteira_plano_ocr": "Carteira do plano",
    "nome_profissional_ocr": "Nome do profissional",
    "conselho_profissional_ocr": "Conselho profissional",
    "numero_conselho_ocr": "Nº no conselho",
    "uf_conselho_ocr": "UF do conselho",
    "cid10_ocr": "CID-10",
    "carater_atendimento": "Caráter do atendimento",
}


# ==============================================================================
# Utilitários internos
# ==============================================================================
def _normalizar_nome_guia(nome_arquivo: str) -> str:
    """'guia 6', 'guia_6', '6' ou o nome antigo 'Paciente_6' -> 'guia_006' (padrão do dataset)."""
    texto = nome_arquivo.strip()
    numero = re.fullmatch(r"(?i)(?:guia|paciente)?[\s_-]*0*(\d{1,3})(\.[a-z]+)?", texto)
    if numero:
        return f"guia_{int(numero.group(1)):03d}{numero.group(2) or ''}"
    return texto


def _resolver_arquivo_guia(nome_arquivo: str) -> Optional[Path]:
    """Aceita o nome do arquivo em input_guias/ (com ou sem extensão), o número da guia ou um caminho completo."""
    nome_arquivo = _normalizar_nome_guia(nome_arquivo)
    candidato = Path(nome_arquivo)
    if candidato.is_absolute() or candidato.exists():
        return candidato if candidato.exists() else None

    pasta_guias = resolver_caminho_repo(
        os.getenv("INPUT_GUIAS_DIR", "ia_ocr_rag/data/input_guias")
    )
    direto = pasta_guias / nome_arquivo
    if direto.exists():
        return direto
    for ext in sorted(EXTENSOES_IMAGEM):
        com_ext = pasta_guias / f"{nome_arquivo}{ext}"
        if com_ext.exists():
            return com_ext
    return None


def _gerar_guia_uuid(caminho: Path) -> str:
    """Mesmo UUID determinístico usado pelo batch_processor (estável entre execuções)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"clinica-vidamais-{caminho.name}"))


def _celula_md(valor: Any) -> str:
    if valor is None or (isinstance(valor, str) and not valor.strip()):
        return NAO_IDENTIFICADO
    return str(valor).replace("|", "\\|").replace("\n", " ").strip()


def _tabelas_extracao(campos: Dict[str, Any], procedimentos: List[Dict[str, Any]]) -> str:
    """Tabelas da extração bruta, montadas de forma determinística (o agente só reproduz)."""
    linhas = ["### Dados da guia", "", "| Campo | Valor extraído |", "| :--- | :--- |"]
    for chave, rotulo in ROTULOS_CAMPOS_GUIA.items():
        linhas.append(f"| {rotulo} | {_celula_md(campos.get(chave))} |")

    linhas += [
        "",
        "### Procedimentos",
        "",
        "| Seq | Código lido | Descrição lida | Qtde | Valor unit. (R$) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    if not procedimentos:
        linhas.append(f"| — | {NAO_IDENTIFICADO} | {NAO_IDENTIFICADO} | — | — |")
    for p in procedimentos:
        linhas.append(
            f"| {_celula_md(p.get('sequencial_item'))} | {_celula_md(p.get('codigo_tuss_ocr'))} | "
            f"{_celula_md(p.get('descricao_procedimento_ocr'))} | {_celula_md(p.get('quantidade'))} | "
            f"{_celula_md(p.get('valor_unitario_ocr'))} |"
        )
    return "\n".join(linhas)


def _tabela_correcao_tuss(resultado: GuiaProcessadaOutput) -> str:
    """Tabela 'lido pelo OCR x corrigido pelo RAG lexical', montada de forma determinística."""
    linhas = [
        "| Seq | Código lido (OCR) | Código corrigido (TUSS) | Descrição oficial TUSS | Método | Score | Revisão humana |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    if not resultado.procedimentos:
        linhas.append(f"| — | {NAO_IDENTIFICADO} | — | — | — | — | Sim |")
    for p in resultado.procedimentos:
        linhas.append(
            f"| {p.sequencial_item} | {_celula_md(p.codigo_tuss_ocr_bruto)} | "
            f"{_celula_md(p.codigo_tuss_corrigido)} | {_celula_md(p.descricao_tuss_oficial)} | "
            f"{p.metodo_correcao.value} | {p.score_confianca_lexical:.2f}% | "
            f"{'Sim' if p.requer_revisao_humana else 'Não'} |"
        )

    motivos = [
        f"- Item {p.sequencial_item}: {p.motivo_revisao}"
        for p in resultado.procedimentos
        if p.requer_revisao_humana and p.motivo_revisao
    ]
    if motivos:
        linhas += ["", "**Motivos de revisão humana:**", *motivos]
    return "\n".join(linhas)


# ==============================================================================
# Tools expostas ao agente
# ==============================================================================
def listar_guias_disponiveis() -> Dict[str, Any]:
    """
    Lista as imagens de guias médicas TISS/TUSS disponíveis na pasta de entrada
    (`ia_ocr_rag/data/input_guias/`).
    """
    pasta_guias = resolver_caminho_repo(
        os.getenv("INPUT_GUIAS_DIR", "ia_ocr_rag/data/input_guias")
    )
    if not pasta_guias.exists():
        return {"status": "erro", "mensagem": f"Pasta não encontrada: {pasta_guias}", "arquivos": []}

    arquivos = sorted(
        (f.name for f in pasta_guias.iterdir() if f.is_file() and f.suffix.lower() in EXTENSOES_IMAGEM),
        key=lambda nome: (len(nome), nome),
    )
    return {"status": "ok", "total_guias": len(arquivos), "arquivos": arquivos}


def extrair_dados_guia(
    nome_arquivo: str,
    usar_fallback_offline: bool = False,
) -> Dict[str, Any]:
    """
    ETAPA 1 (Extração): executa SOMENTE a extração estruturada via modelo multimodal sobre a
    imagem de uma guia de `ia_ocr_rag/data/input_guias/` e devolve os dados exatamente como foram
    lidos, sem correção TUSS. Use para medir a acurácia da extração contra o gabarito.

    Args:
        nome_arquivo: Nome do arquivo da guia (ex: 'guia_003.png', 'guia_003' ou 'guia 3').
        usar_fallback_offline: Se True, usa o motor RapidOCR local em vez do Gemini.
    """
    caminho = _resolver_arquivo_guia(nome_arquivo)
    if caminho is None:
        return {
            "status": "erro",
            "mensagem": f"Arquivo '{nome_arquivo}' não encontrado. Use `listar_guias_disponiveis`.",
        }

    inicio = time.perf_counter()
    guia_raw, motor = get_ocr_extractor().extrair_guia(caminho, force_offline=usar_fallback_offline)
    latencia_ms = round((time.perf_counter() - inicio) * 1000.0, 2)

    arquivo_bruto = salvar_extracao_bruta_ocr(_gerar_guia_uuid(caminho), caminho.name, motor, guia_raw)

    extracao = guia_raw.model_dump(mode="json")
    procedimentos = extracao.pop("procedimentos")
    return {
        "status": "sucesso",
        "arquivo_guia": caminho.name,
        "motor_ocr_utilizado": motor,
        "latencia_extracao_ms": latencia_ms,
        "arquivo_extracao_salvo": str(arquivo_bruto),
        "tabelas_markdown": _tabelas_extracao(extracao, procedimentos),
        "campos_extraidos": extracao,
        "procedimentos_extraidos": procedimentos,
    }


def processar_guia_medica(
    nome_arquivo: str,
    usar_fallback_offline: bool = False,
) -> Dict[str, Any]:
    """
    ETAPAS 1 + 2 (Extração + RAG Lexical): extrai a guia via modelo multimodal, corrige cada
    código TUSS contra a tabela oficial (Exato -> Regra Visual -> Fuzzy+BM25), classifica
    revisão humana (score < 90%) e salva o JSON canônico que alimenta o Databricks.

    Args:
        nome_arquivo: Nome do arquivo da guia (ex: 'guia_002.png', 'guia_002' ou 'guia 2').
        usar_fallback_offline: Se True, usa o motor RapidOCR local em vez do Gemini.
    """
    caminho = _resolver_arquivo_guia(nome_arquivo)
    if caminho is None:
        return {
            "status": "erro",
            "mensagem": f"Arquivo '{nome_arquivo}' não encontrado. Use `listar_guias_disponiveis`.",
        }

    req = GuiaProcessamentoRequest(
        guia_uuid=_gerar_guia_uuid(caminho),
        arquivo_nome=caminho.name,
        caminho_arquivo=str(caminho),
    )
    resultado = processar_guia_end_to_end(
        req, force_offline=usar_fallback_offline, salvar_ocr_bruto=True
    )
    caminho_salvo = salvar_guia_json(resultado)

    return {
        "status": "sucesso",
        "arquivo_guia": resultado.arquivo_origem,
        "status_auditoria_guia": resultado.status_auditoria_guia.value,
        "total_procedimentos": resultado.total_procedimentos,
        "total_requer_revisao_humana": resultado.total_requer_hitl,
        "score_medio_guia": resultado.score_medio_guia,
        "motor_ocr_utilizado": resultado.motor_ocr_utilizado,
        "latencia_processamento_ms": resultado.latencia_processamento_ms,
        "arquivo_json_salvo": str(caminho_salvo),
        "tabela_markdown": _tabela_correcao_tuss(resultado),
        "json_canonico": resultado.model_dump(mode="json"),
    }


def validar_codigo_tuss_manual(
    codigo_tuss_ocr: str,
    descricao_procedimento_ocr: str = "",
) -> Dict[str, Any]:
    """
    ETAPA 2 isolada (RAG Lexical): corrige um código TUSS avulso contra a tabela oficial,
    demonstrando ruído óptico (ex: '10101O12' -> '10101012'), Fuzzy Matching e BM25.

    Args:
        codigo_tuss_ocr: Código TUSS bruto (pode conter letras confundidas pelo OCR).
        descricao_procedimento_ocr: Descrição do procedimento associada (opcional).
    """
    validado = get_lexical_engine().validar_procedimento(
        ProcedimentoOCRRaw(
            sequencial_item=1,
            codigo_tuss_ocr=codigo_tuss_ocr,
            descricao_procedimento_ocr=descricao_procedimento_ocr,
        )
    )
    return validado.model_dump(mode="json")


def consultar_tabela_tuss(termo_ou_codigo: str, limite: int = 5) -> Dict[str, Any]:
    """
    Consulta os procedimentos mais próximos na tabela oficial TUSS (5.967 procedimentos)
    usando BM25 para descrições ou Fuzzy Ratio para códigos numéricos.

    Args:
        termo_ou_codigo: Termo médico (ex: 'hemograma completo') ou código parcial.
        limite: Quantidade máxima de candidatos retornados (padrão: 5).
    """
    engine = get_lexical_engine()
    candidatos = engine.consultar_top_candidatos(termo_ou_codigo, top_k=limite)
    return {
        "consulta": termo_ou_codigo,
        "total_base_tuss": len(engine.tuss_list),
        "candidatos": [
            {"codigo_tuss": cod, "descricao_oficial": desc, "score_similaridade": score}
            for cod, desc, score in candidatos
        ],
    }


INSTRUCOES_AGENTE_VIDAMAIS = """
Você é o Agente de Extração e Padronização TISS/TUSS da **Clínica Vida+**. Seu papel cobre duas etapas:
1. EXTRAÇÃO ESTRUTURADA: ler guias médicas via modelo multimodal e convertê-las em JSON estruturado.
2. RAG LEXICAL (NÃO-VETORIAL): corrigir os códigos TUSS lidos contra a tabela oficial (5.967 procedimentos), de forma auditável.

FERRAMENTAS E QUANDO USAR:
- `listar_guias_disponiveis`: quando o usuário quiser saber quais guias existem.
- `extrair_dados_guia`: quando o usuário pedir para EXTRAIR / LER uma guia (somente a etapa 1).
- `processar_guia_medica`: quando o usuário pedir para PROCESSAR / AUDITAR / CORRIGIR uma guia (etapas 1 + 2).
- `validar_codigo_tuss_manual`: quando o usuário quiser testar a correção de um código avulso (ex: '10101O12').
- `consultar_tabela_tuss`: quando o usuário quiser buscar procedimentos na tabela TUSS por termo ou código.

REGRAS DE COMPORTAMENTO:
1. NUNCA INVENTE DADOS: tudo o que você apresenta vem do retorno de uma ferramenta. Não leia imagens por conta própria e não sugira códigos TUSS de memória.
2. TABELAS PRONTAS: quando a ferramenta retornar `tabelas_markdown` ou `tabela_markdown`, reproduza-as EXATAMENTE como vieram, sem remover linhas, corrigir, completar ou reinterpretar (inclusive as linhas "— não identificado").
3. SEPARE LIDO DE CORRIGIDO: deixe sempre claro o que o modelo multimodal leu da imagem (código bruto) e o que o RAG lexical corrigiu (código oficial). Nunca apresente o código corrigido como se tivesse sido lido.
4. SEJA AUDITÁVEL: ao processar uma guia, informe abaixo da tabela o status (`APROVADA_AUTO` ou `FILA_REVISAO_HUMANA`), o score médio, o motor utilizado e a latência. Se algum item for para revisão humana, explique o motivo retornado pela ferramenta.
5. SEJA HONESTO SOBRE INCERTEZA: se a ferramenta indicar fallback offline, erro ou score baixo, diga isso claramente.
6. ESCOPO: você trata apenas de extração de guias e correção de códigos TUSS. Para outros assuntos, explique educadamente que estão fora do seu escopo.
7. JSON CANÔNICO: só exiba o `json_canonico` se o usuário pedir explicitamente o JSON.
""".strip()

# Modelo do chat com novas tentativas (tenacity) e troca automática de modelo em caso de falha
_MODELOS_AGENTE = cadeia_de_modelos()

root_agent = Agent(
    name="agente_extracao_tuss_vidamais",
    model=GeminiComFallback(model=_MODELOS_AGENTE[0], modelos_reserva=_MODELOS_AGENTE[1:]),
    description="Extração estruturada de guias TISS via modelo multimodal e correção de códigos TUSS via RAG lexical.",
    instruction=INSTRUCOES_AGENTE_VIDAMAIS,
    tools=[
        listar_guias_disponiveis,
        extrair_dados_guia,
        processar_guia_medica,
        validar_codigo_tuss_manual,
        consultar_tabela_tuss,
    ],
)

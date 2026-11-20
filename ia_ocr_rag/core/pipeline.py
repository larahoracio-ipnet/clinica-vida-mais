from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from ia_ocr_rag.core.lexical_rag import (
    REPO_ROOT,
    LexicalTUSSEngine,
    resolver_caminho_repo,
)
from ia_ocr_rag.core.ocr_extractor import GuiaOCRExtractor
from ia_ocr_rag.core.schemas import (
    GuiaOCRRaw,
    GuiaProcessadaOutput,
    GuiaProcessamentoRequest,
    ProcedimentoValidadoOutput,
    StatusAuditoriaGuia,
)

# Garante o carregamento do .env na raiz clinica-vida-mais/
load_dotenv(REPO_ROOT / ".env")


@lru_cache(maxsize=1)
def get_lexical_engine() -> LexicalTUSSEngine:
    """Instância única em memória dos 5.967 procedimentos TUSS e 200 beneficiários."""
    tuss_path = os.getenv(
        "TUSS_CSV_PATH",
        "ia_ocr_rag/data/reference/tuss-22_20260902.xlsx - tuss-22_20260902.csv",
    )
    pacientes_path = os.getenv(
        "PACIENTES_CSV_PATH",
        "ia_ocr_rag/data/reference/pacientes_simulados.csv",
    )
    threshold = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "90.0"))

    return LexicalTUSSEngine(
        tuss_csv_path=tuss_path,
        pacientes_csv_path=pacientes_path,
        hitl_threshold=threshold,
    )


@lru_cache(maxsize=1)
def get_ocr_extractor() -> GuiaOCRExtractor:
    """Instância única do extrator multimodal Gemini (com modelos reserva) + fallback RapidOCR."""
    return GuiaOCRExtractor(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_MODEL_NAME"),  # cadeia completa vem de GEMINI_FALLBACK_MODELS
    )


def _formatar_conselho(
    sigla: Optional[str],
    uf: Optional[str],
    numero: Optional[str],
) -> Optional[str]:
    partes = [p.strip() for p in (sigla, uf, numero) if p and p.strip()]
    return " ".join(partes) if partes else None


def salvar_extracao_bruta_ocr(
    guia_uuid: str,
    arquivo_origem: str,
    motor_ocr: str,
    guia_raw: GuiaOCRRaw,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Persiste exatamente o que o motor multimodal leu da imagem (antes do RAG),
    para medir a acurácia da extração contra o gabarito e reprocessar sem chamar a API.
    Artefato local de depuração (pasta ignorada pelo git).
    """
    base_out = resolver_caminho_repo(
        output_dir or os.getenv("OUTPUT_JSON_DIR", "ia_ocr_rag/data/output_json")
    )
    pasta_bruta = base_out / "ocr_bruto"
    pasta_bruta.mkdir(parents=True, exist_ok=True)

    registro = {
        "guia_uuid": guia_uuid,
        "arquivo_origem": arquivo_origem,
        "motor_ocr_utilizado": motor_ocr,
        "extracao": guia_raw.model_dump(mode="json"),
    }
    destino = pasta_bruta / f"{guia_uuid}.json"
    destino.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def montar_guia_processada(
    request: GuiaProcessamentoRequest,
    guia_raw: GuiaOCRRaw,
    motor_ocr: str,
    inicio: Optional[float] = None,
) -> GuiaProcessadaOutput:
    """
    Etapa pós-OCR (determinística, sem chamar a API):
      1. Validação cadastral do paciente.
      2. Padronização lexical TUSS (Exato + Ruído Óptico + Fuzzy + BM25).
      3. Roteamento automático vs. fila Human-in-the-Loop (score < 90%).
    """
    inicio = time.perf_counter() if inicio is None else inicio
    engine = get_lexical_engine()

    paciente = engine.validar_paciente(
        nome_ocr=guia_raw.nome_paciente_ocr,
        carteira_ocr=guia_raw.carteira_plano_ocr,
    )
    conselho_fmt = _formatar_conselho(
        guia_raw.conselho_profissional_ocr,
        guia_raw.uf_conselho_ocr,
        guia_raw.numero_conselho_ocr,
    )

    procedimentos_validados: List[ProcedimentoValidadoOutput] = [
        engine.validar_procedimento(proc) for proc in guia_raw.procedimentos
    ]

    total_procs = len(procedimentos_validados)
    total_hitl = sum(1 for p in procedimentos_validados if p.requer_revisao_humana)

    if total_procs == 0:
        score_medio = 0.0
        status_guia = StatusAuditoriaGuia.FILA_REVISAO_HUMANA
    else:
        score_medio = round(
            sum(p.score_confianca_lexical for p in procedimentos_validados) / total_procs,
            2,
        )
        status_guia = (
            StatusAuditoriaGuia.FILA_REVISAO_HUMANA
            if total_hitl > 0
            else StatusAuditoriaGuia.APROVADA_AUTO
        )

    latencia_ms = round((time.perf_counter() - inicio) * 1000.0, 2)

    return GuiaProcessadaOutput(
        guia_uuid=request.guia_uuid,
        arquivo_origem=request.arquivo_nome,
        unidade_clinica=request.unidade_clinica,
        numero_guia_prestador=guia_raw.numero_guia_prestador,
        numero_guia_operadora=guia_raw.numero_guia_operadora,
        registro_ans=guia_raw.registro_ans,
        data_atendimento=guia_raw.data_atendimento,
        cid10=guia_raw.cid10_ocr,
        carater_atendimento=guia_raw.carater_atendimento,
        paciente=paciente,
        profissional_solicitante=guia_raw.nome_profissional_ocr,
        conselho_profissional=conselho_fmt,
        procedimentos=procedimentos_validados,
        total_procedimentos=total_procs,
        total_requer_hitl=total_hitl,
        status_auditoria_guia=status_guia,
        score_medio_guia=score_medio,
        motor_ocr_utilizado=motor_ocr,
        latencia_processamento_ms=latencia_ms,
    )


def processar_guia_end_to_end(
    request: GuiaProcessamentoRequest,
    force_offline: bool = False,
    salvar_ocr_bruto: bool = False,
) -> GuiaProcessadaOutput:
    """
    Executa o fluxo completo da Clínica Vida+:
      1. Extração estruturada OCR (Gemini multimodal ou RapidOCR offline).
      2. Etapa pós-OCR: validação do paciente, correção TUSS e roteamento (montar_guia_processada).
    """
    inicio = time.perf_counter()

    caminho_img = resolver_caminho_repo(request.caminho_arquivo)
    guia_raw, motor_ocr = get_ocr_extractor().extrair_guia(caminho_img, force_offline=force_offline)

    if salvar_ocr_bruto:
        salvar_extracao_bruta_ocr(request.guia_uuid, request.arquivo_nome, motor_ocr, guia_raw)

    return montar_guia_processada(request, guia_raw, motor_ocr, inicio=inicio)


def salvar_guia_json(
    resultado: GuiaProcessadaOutput,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """Persiste o JSON individual validado na pasta de saída."""
    base_out = resolver_caminho_repo(
        output_dir or os.getenv("OUTPUT_JSON_DIR", "ia_ocr_rag/data/output_json")
    )
    pasta_individual = base_out / "individual"
    pasta_individual.mkdir(parents=True, exist_ok=True)

    arquivo_destino = pasta_individual / f"{resultado.guia_uuid}.json"
    arquivo_destino.write_text(
        resultado.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return arquivo_destino
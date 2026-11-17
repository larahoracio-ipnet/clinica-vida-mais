import logging
from datetime import datetime, timezone
import time
import hashlib

from backend.app.models.schemas import (
    GuiaProcessadaOutput,
    GuiaProcessamentoRequest,
    MetodoCorrecaoTUSS,
    PacienteLGPDOutput,
    ProcedimentoValidadoOutput,
    StatusAuditoriaGuia,
)

logger = logging.getLogger(__name__)

try:
    from ia_ocr_rag.pipeline import executar_pipeline_guia # type: ignore[import-not-found]

    IA_DISPONIVEL = True
except ImportError:    
    IA_DISPONIVEL = False
    logger.warning(
        "Pipeline 'ia_ocr_rag.pipeline' não encontrado. Operando com Mock Fallback alinhado."
    )

def _gerar_mock_auditoria(request: GuiaProcessamentoRequest) -> GuiaProcessadaOutput:
    """Gera um GuiaProcessadaOutput realista respeitando todas as regras de negócio da Clínica Vida+."""

    inicio_execucao = time.perf_counter()

    nome_lower = request.arquivo_nome.lower()
    simular_hitl = "erro" in nome_lower or "glosa" in nome_lower

    hash_paciente_mock = hashlib.sha256(b"00321234567890001:salt_mvp").hexdigest()

    paciente_mock = PacienteLGPDOutput(
        paciente_validado_cadastro=True,
        metodo_match_paciente="EXATO_CARTEIRA",
        score_match_paciente=100.0,
        nome_paciente_mascarado="M**** S**** O*******",
        carteira_plano_mascarada="0032***********01",
        paciente_hash_sha256=hash_paciente_mock,
    )

    if simular_hitl:
        procedimentos_mock = [
            ProcedimentoValidadoOutput(
                sequencial_item=1,
                codigo_tuss_ocr_bruto="1010101B",  # Simula erro comum de OCR trocando '2' por 'B'
                codigo_tuss_corrigido="10101012",
                descricao_ocr_bruta="Consulta em consultorio",
                descricao_tuss_oficial="Consulta em consultório (no horário normal ou preestabelecido)",
                source_tuss="TUSS_22_ANS",
                vigencia_ativa=True,
                quantidade=1,
                valor_unitario=150.0,
                score_confianca_lexical=78.5,  # Abaixo de 90.0 -> dispara revisão humana (HITL)
                score_fuzzy_codigo=87.5,
                score_bm25_descricao=72.0,
                metodo_correcao=MetodoCorrecaoTUSS.FUZZY_BM25,
                requer_revisao_humana=True,
                motivo_revisao="Score lexical (78.5) abaixo do limiar de 90.0 para aprovação automática.",
            )
        ]
        total_hitl = 1
        status_guia = StatusAuditoriaGuia.FILA_REVISAO_HUMANA
        score_medio = 78.5
    else:
        # Caminho feliz: OCR e RAG lexical encontraram correspondência exata (score >= 90.0)
        procedimentos_mock = [
            ProcedimentoValidadoOutput(
                sequencial_item=1,
                codigo_tuss_ocr_bruto="10101012",
                codigo_tuss_corrigido="10101012",
                descricao_ocr_bruta="Consulta em consultorio",
                descricao_tuss_oficial="Consulta em consultório (no horário normal ou preestabelecido)",
                source_tuss="TUSS_22_ANS",
                vigencia_ativa=True,
                quantidade=1,
                valor_unitario=150.0,
                score_confianca_lexical=98.5,
                score_fuzzy_codigo=100.0,
                score_bm25_descricao=97.0,
                metodo_correcao=MetodoCorrecaoTUSS.EXATO,
                requer_revisao_humana=False,
                motivo_revisao=None,
            ),
            ProcedimentoValidadoOutput(
                sequencial_item=2,
                codigo_tuss_ocr_bruto="40304361",
                codigo_tuss_corrigido="40304361",
                descricao_ocr_bruta="Hemograma completo",
                descricao_tuss_oficial="Hemograma completo",
                source_tuss="TUSS_22_ANS",
                vigencia_ativa=True,
                quantidade=1,
                valor_unitario=35.50,
                score_confianca_lexical=100.0,
                score_fuzzy_codigo=100.0,
                score_bm25_descricao=100.0,
                metodo_correcao=MetodoCorrecaoTUSS.EXATO,
                requer_revisao_humana=False,
                motivo_revisao=None,
            ),
        ]
        total_hitl = 0
        status_guia = StatusAuditoriaGuia.APROVADA_AUTO
        score_medio = 99.25

    # Calcula quanto tempo o processamento levou e converte de segundos para milissegundos (ms)
    latencia_ms = round((time.perf_counter() - inicio_execucao) * 1000, 2)

    # Retorna o contrato final exatamente no formato esperado pelo Backend e pelo Databricks
    return GuiaProcessadaOutput(
        guia_uuid=request.guia_uuid,
        arquivo_origem=request.arquivo_nome,
        unidade_clinica=request.unidade_clinica,
        timestamp_processamento_utc=datetime.now(timezone.utc).isoformat(),
        numero_guia_prestador="2026000123",
        numero_guia_operadora="99887766",
        registro_ans="326305",
        data_atendimento=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        cid10="J06.9",  # Dado clínico sensível: será ocultado para 'recepcao' e exibido para 'medico'
        carater_atendimento="Eletivo",
        paciente=paciente_mock,
        profissional_solicitante_mascarado="Dr. C***** A******",
        conselho_profissional="CRM-RJ 123456",
        procedimentos=procedimentos_mock,
        total_procedimentos=len(procedimentos_mock),
        total_requer_hitl=total_hitl,
        status_auditoria_guia=status_guia,
        score_medio_guia=score_medio,
        motor_ocr_utilizado="mock-fallback-mvp",
        latencia_processamento_ms=latencia_ms,
        versao_pipeline="1.0.0-mvp",
    )

def auditar_guia(request: GuiaProcessamentoRequest) -> GuiaProcessadaOutput:

    if IA_DISPONIVEL:
        logger.info("Executando pipeline real da IA para guia %s", request.guia_uuid)

        resultado_bruto = executar_pipeline_guia(request)
        return GuiaProcessadaOutput.model_validate(resultado_bruto)

    return _gerar_mock_auditoria(request)
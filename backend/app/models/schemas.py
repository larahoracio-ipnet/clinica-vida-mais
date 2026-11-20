from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# 1. PERFIS DE ACESSO DO BACKEND (RBAC - Role-Based Access Control)
# ==============================================================================
class PerfilUsuario(str, Enum):

    RECEPCAO = "recepcao"  # Faz upload de guias e consulta apenas status administrativo
    MEDICO = "medico"      # Consulta dados clínicos completos (CID-10, procedimentos e justificativas)
    GESTOR = "gestor"      # Acessa apenas KPIs agregados (bloqueado de ver guias individuais)


# ==============================================================================
# 2. CONTRATOS COMPARTILHADOS COM A IA (ia_ocr_rag/core/schemas.py)
# ==============================================================================

try:
    from ia_ocr_rag.core.schemas import (   # type: ignore[import-not-found]
        GuiaProcessadaOutput,
        GuiaProcessamentoRequest,
        MetodoCorrecaoTUSS,
        PacienteOutput,
        ProcedimentoValidadoOutput,
        StatusAuditoriaGuia,
    )
except ImportError:

    class MetodoCorrecaoTUSS(str, Enum):
        """Estratégia lexical utilizada pelo RAG para validar/corrigir o código TUSS."""

        EXATO = "EXATO"
        REGRA_VISUAL = "REGRA_VISUAL"
        FUZZY_BM25 = "FUZZY_BM25"
        NAO_ENCONTRADO = "NAO_ENCONTRADO"

    class StatusAuditoriaGuia(str, Enum):
        """Define se a guia foi aprovada direto ou precisa de revisão humana (HITL)."""

        APROVADA_AUTO = "APROVADA_AUTO"
        FILA_REVISAO_HUMANA = "FILA_REVISAO_HUMANA"

    class GuiaProcessamentoRequest(BaseModel):
        """Payload de entrada que o backend envia para o pipeline de IA processar a imagem."""

        guia_uuid: str = Field(..., description="Identificador único da guia gerado no upload.")
        arquivo_nome: str = Field(..., description="Nome original do arquivo enviado.")
        caminho_arquivo: str = Field(..., description="Caminho em texto (str) onde a imagem foi salva no disco.")
        unidade_clinica: str = Field(
            default="UNIDADE_01_MATRIZ",
            description="Unidade da Clínica Vida+ onde o atendimento ocorreu.",
        )
        operadora_origem: Optional[str] = Field(
            default=None,
            description="Nome da operadora informado opcionalmente na recepção.",
        )

    class PacienteOutput(BaseModel):
        """Dados do paciente: como lidos pelo OCR e, se localizado, como estão no cadastro."""

        paciente_validado_cadastro: bool
        metodo_match_paciente: str
        score_match_paciente: float = Field(..., ge=0.0, le=100.0)
        nome_paciente_ocr_bruto: Optional[str] = None
        carteira_plano_ocr_bruta: Optional[str] = None
        nome_paciente_cadastro: Optional[str] = None
        carteira_plano_cadastro: Optional[str] = None

    class ProcedimentoValidadoOutput(BaseModel):
        """Resultado da validação TUSS (comparação Pré-RAG vs Pós-RAG)."""

        sequencial_item: int
        codigo_tuss_ocr_bruto: str
        codigo_tuss_corrigido: Optional[str] = None
        descricao_ocr_bruta: str
        descricao_tuss_oficial: Optional[str] = None
        source_tuss: Optional[str] = None
        vigencia_ativa: bool = True
        quantidade: int = 1
        valor_unitario: Optional[float] = None
        score_confianca_lexical: float = Field(..., ge=0.0, le=100.0)
        score_fuzzy_codigo: float = Field(..., ge=0.0, le=100.0)
        score_bm25_descricao: float = Field(..., ge=0.0, le=100.0)
        metodo_correcao: MetodoCorrecaoTUSS
        requer_revisao_humana: bool
        motivo_revisao: Optional[str] = None

    class GuiaProcessadaOutput(BaseModel):
        """Contrato completo devolvido pela IA e salvo intacto para ingestão no Databricks."""

        guia_uuid: str
        arquivo_origem: str
        unidade_clinica: str
        timestamp_processamento_utc: str = Field(
            default_factory=lambda: datetime.now(timezone.utc).isoformat()
        )
        numero_guia_prestador: Optional[str] = None
        numero_guia_operadora: Optional[str] = None
        registro_ans: Optional[str] = None
        data_atendimento: Optional[str] = None
        cid10: Optional[str] = None
        carater_atendimento: Optional[str] = None
        paciente: PacienteOutput
        profissional_solicitante: Optional[str] = None
        conselho_profissional: Optional[str] = None
        procedimentos: List[ProcedimentoValidadoOutput]
        total_procedimentos: int
        total_requer_hitl: int
        status_auditoria_guia: StatusAuditoriaGuia
        score_medio_guia: float
        motor_ocr_utilizado: str
        latencia_processamento_ms: float
        versao_pipeline: str = "2.0.0-mvp"


# ==============================================================================
# 3. CONTRATOS DE RESPOSTA DA API REST (PROJEÇÕES POR PERFIL / ENDPOINT)
# ==============================================================================
class UploadGuiaResponse(BaseModel):

    mensagem: str = Field(..., description="Confirmação do processamento.")
    guia_uuid: str = Field(..., description="UUID único gerado para rastreamento da guia.")
    arquivo_salvo: str = Field(..., description="Nome do arquivo armazenado em storage/raw.")
    sha256_arquivo: str = Field(..., description="Hash SHA-256 da imagem original para garantir integridade.")
    unidade_clinica: str
    status_auditoria_guia: StatusAuditoriaGuia
    total_procedimentos: int
    total_requer_hitl: int
    score_medio_guia: float
    latencia_processamento_ms: float


class GuiaRecepcaoResponse(BaseModel):

    guia_uuid: str
    arquivo_origem: str
    unidade_clinica: str
    numero_guia_prestador: Optional[str] = None
    numero_guia_operadora: Optional[str] = None
    registro_ans: Optional[str] = None
    data_atendimento: Optional[str] = None
    paciente_validado_cadastro: bool
    nome_paciente: Optional[str] = None
    carteira_plano: Optional[str] = None
    status_auditoria_guia: StatusAuditoriaGuia
    total_procedimentos: int
    total_requer_hitl: int
    timestamp_processamento_utc: str


GuiaMedicoResponse = GuiaProcessadaOutput
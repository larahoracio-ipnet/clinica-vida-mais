from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MetodoCorrecaoTUSS(str, Enum):
    """Classifica a estratégia lexical que resolveu o código TUSS."""
    EXATO = "EXATO"
    REGRA_VISUAL = "REGRA_VISUAL"
    FUZZY_BM25 = "FUZZY_BM25"
    NAO_ENCONTRADO = "NAO_ENCONTRADO"


class StatusAuditoriaGuia(str, Enum):
    """Define o roteamento operacional da guia na Clínica Vida+."""
    APROVADA_AUTO = "APROVADA_AUTO"
    FILA_REVISAO_HUMANA = "FILA_REVISAO_HUMANA"


# ==============================================================================
# 1. CONTRATO DE ENTRADA (backend/ FastAPI | adk web | batch_processor -> IA)
# ==============================================================================
class GuiaProcessamentoRequest(BaseModel):
    guia_uuid: str = Field(..., description="Identificador único da guia gerado na recepção/API.")
    arquivo_nome: str = Field(..., description="Nome original do arquivo da guia médica.")
    caminho_arquivo: str = Field(..., description="Caminho do arquivo de imagem no disco.")
    unidade_clinica: str = Field(
        default="UNIDADE_01_MATRIZ",
        description="Unidade clínica de origem entre as 8 unidades da rede Vida+."
    )
    operadora_origem: Optional[str] = Field(
        default=None,
        description="Operadora de saúde informada no atendimento (opcional)."
    )


# ==============================================================================
# 2. CONTRATO DE EXTRAÇÃO BRUTA DO OCR (Gemini 3.8 Flash Structured Output)
# ==============================================================================
class ProcedimentoOCRRaw(BaseModel):
    sequencial_item: int = Field(..., description="Número sequencial do procedimento na guia (1, 2, 3...).")
    codigo_tuss_ocr: str = Field(..., description="Código TUSS exatamente como lido na imagem.")
    descricao_procedimento_ocr: str = Field(..., description="Descrição do procedimento lida na imagem.")
    quantidade: int = Field(default=1, description="Quantidade solicitada/executada.")
    valor_unitario_ocr: Optional[float] = Field(
        default=None,
        description="Valor unitário em reais, caso visível na guia."
    )


class GuiaOCRRaw(BaseModel):
    numero_guia_prestador: Optional[str] = Field(default=None, description="Número da guia no prestador.")
    numero_guia_operadora: Optional[str] = Field(default=None, description="Número da guia na operadora.")
    registro_ans: Optional[str] = Field(default=None, description="Registro ANS da operadora (6 dígitos).")
    data_atendimento: Optional[str] = Field(default=None, description="Data do atendimento (YYYY-MM-DD ou DD/MM/YYYY).")
    nome_paciente_ocr: Optional[str] = Field(default=None, description="Nome do beneficiário extraído da guia.")
    carteira_plano_ocr: Optional[str] = Field(default=None, description="Carteirinha do plano (17 dígitos) extraída da guia.")
    nome_profissional_ocr: Optional[str] = Field(default=None, description="Nome do profissional solicitante/executante.")
    conselho_profissional_ocr: Optional[str] = Field(default=None, description="Sigla do conselho (ex: CRM, CRO, CREFITO).")
    numero_conselho_ocr: Optional[str] = Field(default=None, description="Número de inscrição no conselho.")
    uf_conselho_ocr: Optional[str] = Field(default=None, description="UF do conselho profissional (ex: RJ, SP).")
    cid10_ocr: Optional[str] = Field(default=None, description="Código CID-10 se preenchido na guia.")
    carater_atendimento: Optional[str] = Field(default=None, description="Eletivo ou Urgência/Emergência.")
    procedimentos: List[ProcedimentoOCRRaw] = Field(
        default_factory=list,
        description="Lista de procedimentos médicos extraídos da grade TISS/TUSS."
    )


# ==============================================================================
# 3. CONTRATO DE SAÍDA FINAL (IA -> backend/ FastAPI & databricks_dados/ Bronze)
# ==============================================================================
class PacienteOutput(BaseModel):
    """
    Identificação do paciente, sem mascaramento (MVP com dados fictícios).
    Segue o mesmo padrão "lido × corrigido" dos códigos TUSS: o valor como o OCR leu
    e, quando o paciente é localizado, o valor oficial do cadastro.
    """
    paciente_validado_cadastro: bool = Field(
        ..., description="True se localizado no cadastro mestre pacientes_simulados.csv."
    )
    metodo_match_paciente: str = Field(
        ..., description="EXATO_CARTEIRA, FUZZY_CARTEIRA_NOME ou NAO_LOCALIZADO."
    )
    score_match_paciente: float = Field(..., ge=0.0, le=100.0)
    nome_paciente_ocr_bruto: Optional[str] = Field(
        default=None, description="Nome do paciente exatamente como lido pelo OCR."
    )
    carteira_plano_ocr_bruta: Optional[str] = Field(
        default=None, description="Carteirinha exatamente como lida pelo OCR."
    )
    nome_paciente_cadastro: Optional[str] = Field(
        default=None, description="Nome oficial no cadastro (null se o paciente não foi localizado)."
    )
    carteira_plano_cadastro: Optional[str] = Field(
        default=None, description="Carteirinha oficial de 17 dígitos no cadastro (null se não localizado)."
    )


class ProcedimentoValidadoOutput(BaseModel):
    sequencial_item: int
    codigo_tuss_ocr_bruto: str = Field(
        ..., description="Código bruto lido pelo OCR (métrica de acurácia Pré-RAG no Databricks)."
    )
    codigo_tuss_corrigido: Optional[str] = Field(
        default=None, description="Código oficial TUSS de 8 dígitos após RAG Lexical (acurácia Pós-RAG)."
    )
    descricao_ocr_bruta: str
    descricao_tuss_oficial: Optional[str] = Field(
        default=None, description="Descrição oficial limpa (display_name.strip()) da tabela TUSS."
    )
    source_tuss: Optional[str] = Field(default=None, description="Coluna source da base TUSS.")
    vigencia_ativa: bool = Field(
        default=True, description="True se extras_fim_vigencia estiver vazio na base oficial."
    )
    quantidade: int = 1
    valor_unitario: Optional[float] = None
    score_confianca_lexical: float = Field(..., ge=0.0, le=100.0)
    score_fuzzy_codigo: float = Field(..., ge=0.0, le=100.0)
    score_bm25_descricao: float = Field(..., ge=0.0, le=100.0)
    metodo_correcao: MetodoCorrecaoTUSS
    requer_revisao_humana: bool = Field(
        ..., description="True quando score_confianca_lexical < 90.0 ou código não localizado."
    )
    motivo_revisao: Optional[str] = None


class GuiaProcessadaOutput(BaseModel):
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
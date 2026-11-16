from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

class StatusGuia(str, Enum):
    PROCESSADA = "processada"
    PENDENTE_REVISAO = "pendente_revisao"
    ERRO = "erro"

class ProcedimentoTUSS(BaseModel):
    codigo_tuss: str = Field(..., description="Código do procedimento na tabela TUSS")
    descricao: str = Field(..., description="Nome do procedimento")
    quantidade: int = Field(..., gt=0, description="Quantidade realizada")
    valor: float = Field(..., ge=0, description="Valor unitário em reais")

class GuiaTISS(BaseModel):
    numero_guia: str
    operadora: str
    data_atendimento: date
    paciente_nome: str
    paciente_cpf: str
    procedimentos: list[ProcedimentoTUSS]

class ResultadoAuditoria(BaseModel):
    guia_uuid: str
    status: StatusGuia
    guia: GuiaTISS
    inconsistencias: list[str] = Field(default_factory=list)
    score_confianca: float = Field(..., ge=0, le=1)
    criado_em: datetime
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, status

from backend.app.core.config import STORAGE_DIR
from backend.app.models.schemas import (
    GuiaMedicoResponse,
    GuiaProcessadaOutput,
    GuiaRecepcaoResponse,
    PerfilUsuario,
)

logger = logging.getLogger(__name__)

LOGS_DIR = STORAGE_DIR / "logs"
AUDIT_TRAIL_PATH = LOGS_DIR / "audit_trail.jsonl"


# ==============================================================================
# 1. TRILHA DE AUDITORIA IMUTÁVEL (AUDIT TRAIL LGPD)
# ==============================================================================
def registrar_auditoria(
    evento: str,
    perfil_usuario: str,
    guia_uuid: Optional[str] = None,
    detalhes: Optional[str] = None,
    sucesso: bool = True,
) -> None:
    """
    Grava uma linha JSON imutável (append-only) em backend/storage/logs/audit_trail.jsonl.
    Todo acesso, upload ou tentativa bloqueada (401/403) fica registrada para rastreabilidade.
    """
    # Garante que a pasta backend/storage/logs/ exista antes de tentar abrir o arquivo
    # parents=True cria pastas intermediárias se faltarem; exist_ok=True não dá erro se já existir
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    registro = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evento": evento,
        "perfil_usuario": perfil_usuario,
        "guia_uuid": guia_uuid,
        "sucesso": sucesso,
        "detalhes": detalhes,
    }

    with AUDIT_TRAIL_PATH.open("a", encoding="utf-8") as arquivo_log:
        arquivo_log.write(json.dumps(registro, ensure_ascii=False) + "\n")


# ==============================================================================
# 2. AUTENTICAÇÃO E AUTORIZAÇÃO VIA HEADER X-User-Role (RBAC)
# ==============================================================================
def obter_perfil_atual(
    x_user_role: Optional[str] = Header(
        default=None,
        alias="X-User-Role",
        description="Perfil do requisitante: recepcao, medico ou gestor",
    )
) -> PerfilUsuario:
    """
    Dependência base do FastAPI: lê e valida o header X-User-Role.
    Se estiver ausente ou inválido, registra na auditoria e barra com HTTP 401.
    """
    # Se o cliente não enviou o header X-User-Role, não sabemos quem ele é -> HTTP 401 Unauthorized
    if not x_user_role:
        registrar_auditoria(
            evento="ACESSO_NEGADO_SEM_PERFIL",
            perfil_usuario="anonimo",
            detalhes="Header X-User-Role ausente na requisição.",
            sucesso=False,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header 'X-User-Role' obrigatório (valores aceitos: recepcao, medico, gestor).",
        )

   
    try:
        return PerfilUsuario(x_user_role.strip().lower())
    except ValueError:
        
        registrar_auditoria(
            evento="ACESSO_NEGADO_PERFIL_INVALIDO",
            perfil_usuario=x_user_role.strip()[:30],  
            detalhes="Valor inválido enviado no header X-User-Role.",
            sucesso=False,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Perfil '{x_user_role}' inválido. Use: recepcao, medico ou gestor.",
        )


def exigir_perfis(*perfis_permitidos: PerfilUsuario) -> Callable[[PerfilUsuario], PerfilUsuario]:
    """
    Fábrica de dependências RBAC (Menor Privilégio).
    Uso nas rotas: Depends(exigir_perfis(PerfilUsuario.RECEPCAO, PerfilUsuario.MEDICO))
    """

    def _verificar_permissao(
        perfil: PerfilUsuario = Depends(obter_perfil_atual),
    ) -> PerfilUsuario:
       
        if perfil not in perfis_permitidos:
            permitidos_str = ", ".join(p.value for p in perfis_permitidos)
            registrar_auditoria(
                evento="ACESSO_BLOQUEADO_RBAC",
                perfil_usuario=perfil.value,
                detalhes=f"Perfil sem permissão para esta rota. Permitidos: [{permitidos_str}].",
                sucesso=False,
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Acesso negado para o perfil '{perfil.value}'. Permitido apenas para: {permitidos_str}.",
            )
        return perfil

    return _verificar_permissao


# ==============================================================================
# 3. PROJEÇÃO DE DADOS POR PERFIL (DEFESA EM PROFUNDIDADE / LGPD)
# ==============================================================================
def aplicar_visao_por_perfil(
    guia: GuiaProcessadaOutput,
    perfil: PerfilUsuario,
) -> GuiaRecepcaoResponse | GuiaMedicoResponse:
    """
    Filtra os campos de uma guia processada de acordo com o cargo do requisitante:
    - gestor: bloqueado (403) de ver guias individuais (só acessa KPIs agregados).
    - recepcao: recebe apenas dados administrativos (sem CID-10 e sem procedimentos).
    - medico: recebe o contrato clínico completo.
    """

    if perfil == PerfilUsuario.GESTOR:
        registrar_auditoria(
            evento="CONSULTA_GUIA_BLOQUEADA_GESTOR",
            perfil_usuario=perfil.value,
            guia_uuid=guia.guia_uuid,
            detalhes="Gestor tentou acessar guia individual (violação do Menor Privilégio).",
            sucesso=False,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Perfil 'gestor' não possui permissão para visualizar guias individuais.",
        )

    # Se for recepção, constrói a visão administrativa enxuta omitindo dados clínicos sensíveis
    if perfil == PerfilUsuario.RECEPCAO:
        registrar_auditoria(
            evento="CONSULTA_GUIA_ADMINISTRATIVA",
            perfil_usuario=perfil.value,
            guia_uuid=guia.guia_uuid,
            detalhes="Visualização administrativa retornada (CID-10 e procedimentos ocultados).",
            sucesso=True,
        )
        return GuiaRecepcaoResponse(
            guia_uuid=guia.guia_uuid,
            arquivo_origem=guia.arquivo_origem,
            unidade_clinica=guia.unidade_clinica,
            numero_guia_prestador=guia.numero_guia_prestador,
            numero_guia_operadora=guia.numero_guia_operadora,
            registro_ans=guia.registro_ans,
            data_atendimento=guia.data_atendimento,
            paciente_validado_cadastro=guia.paciente.paciente_validado_cadastro,
            nome_paciente_mascarado=guia.paciente.nome_paciente_mascarado,
            carteira_plano_mascarada=guia.paciente.carteira_plano_mascarada,
            status_auditoria_guia=guia.status_auditoria_guia,
            total_procedimentos=guia.total_procedimentos,
            total_requer_hitl=guia.total_requer_hitl,
            timestamp_processamento_utc=guia.timestamp_processamento_utc,
        )

    # Perfil MEDICO: acesso completo ao prontuário/guia auditada
    registrar_auditoria(
        evento="CONSULTA_GUIA_CLINICA",
        perfil_usuario=perfil.value,
        guia_uuid=guia.guia_uuid,
        detalhes="Visualização clínica completa retornada ao médico.",
        sucesso=True,
    )
    return guia
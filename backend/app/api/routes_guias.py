from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.core.config import Settings, get_settings

from backend.app.core.security import (
    aplicar_visao_por_perfil,
    exigir_perfis,
    obter_perfil_atual,
    registrar_auditoria,
)

from backend.app.models.schemas import (
    GuiaMedicoResponse,
    GuiaProcessamentoRequest,
    GuiaRecepcaoResponse,
    PerfilUsuario,
    UploadGuiaResponse,
)

from backend.app.services.ai_bridge import auditar_guia

from backend.app.services.storage_service import (
    calcular_sha256,
    carregar_guia_processada,
    salvar_imagem_bruta,
    salvar_json_processado,
)

router = APIRouter(prefix="/api/v1/guias", tags=["Guias TISS/TUSS"])

EXTENSOES_PERMITIDAS = {".png", ".jpg", ".jpeg", ".pdf"}

# ==============================================================================
# ROTA 1: UPLOAD E AUDITORIA UNITÁRIA DE GUIA MÉDICA
# ==============================================================================

@router.post(
    "/upload",
    response_model=UploadGuiaResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Envia uma guia médica para auditoria TISS/TUSS",
)

async def upload_guia(
    arquivo: UploadFile = File(..., description="Imagem (PNG/JPG) ou PDF da guia médica"),
    unidade_clinica: str = Form(
        default="UNIDADE_01_MATRIZ",
        description="Unidade de origem entre as 8 unidades da Clínica Vida+",
    ),
    operadora_origem: Optional[str] = Form(
        default=None,
        description="Nome da operadora informado opcionalmente na recepção",
    ),
    # RBAC: apenas 'recepcao' e 'medico' podem enviar guias;
    perfil: PerfilUsuario = Depends(
        exigir_perfis(PerfilUsuario.RECEPCAO, PerfilUsuario.MEDICO)
    ),

    settings: Settings = Depends(get_settings),
) -> UploadGuiaResponse:
    """Orquestra validação, integridade SHA-256, salvamento bruto, auditoria IA e persistência."""

    # 1. Validação de extensão do arquivo (allowlist)
    nome_arquivo = arquivo.filename or ""
    extensao = Path(nome_arquivo).suffix.lower()

    if extensao not in EXTENSOES_PERMITIDAS:
        registrar_auditoria(
            evento="UPLOAD_REJEITADO_EXTENSAO",
            perfil_usuario=perfil.value,
            detalhes=f"Extensão não permitida: '{extensao}' no arquivo '{nome_arquivo}'.",
            sucesso=False,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato '{extensao}' inválido. Envie apenas: {', '.join(sorted(EXTENSOES_PERMITIDAS))}.",
        )

    # 2. Leitura assíncrona dos bytes e validação de arquivo vazio ou acima do limite (MAX_UPLOAD_MB)

    conteudo_bytes = await arquivo.read()

    if len(conteudo_bytes) == 0:
        registrar_auditoria(
            evento="UPLOAD_REJEITADO_VAZIO",
            perfil_usuario=perfil.value,
            detalhes=f"Arquivo vazio enviado: '{nome_arquivo}'.",
            sucesso=False,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo enviado está vazio (0 Bytes)"
        )

    limite_bytes = settings.max_upload_mb * 1024 * 1024
    if len(conteudo_bytes) > limite_bytes:
        registrar_auditoria(
            evento="UPLOAD_REJEITADO_TAMANHO",
            perfil_usuario=perfil.value,
            detalhes=f"Arquivo '{nome_arquivo}' excedeu o limite de {settings.max_upload_mb} MB.",
            sucesso=False,
        )

        raise HTTPException(
            status_code=413,
            detail=f"Arquivo excede o limite máximo permitido de {settings.max_upload_mb} MB.",
        )

    # 3. Geração do UUID único da guia e cálculo do hash SHA-256 para integridade

    guia_uuid = str(uuid4())
    sha256_arquivo = calcular_sha256(conteudo_bytes)

    # 4. Persistência da imagem original em backend/storage/raw/

    caminho_imagem_raw = salvar_imagem_bruta(
        guia_uuid=guia_uuid,
        nome_original=nome_arquivo,
        conteudo_bytes=conteudo_bytes
    )

    # 5. Montagem do contrato de entrada da IA e execução da auditoria (OCR + RAG)

    request_ia = GuiaProcessamentoRequest(
        guia_uuid=guia_uuid,
        arquivo_nome=Path(nome_arquivo).name,
        caminho_arquivo=str(caminho_imagem_raw),
        unidade_clinica=unidade_clinica,
        operadora_origem=operadora_origem,
    )
    resultado_auditoria = auditar_guia(request_ia)

    # 6. Salvamento do JSON final em backend/storage/processed/ + envio para o Databricks Volume

    _, enviado_databricks = salvar_json_processado(resultado_auditoria)

    # 7. Registro da operação bem-sucedida na trilha de auditoria LGPD

    registrar_auditoria(
        evento="UPLOAD_E_AUDITORIA_CONCLUIDOS",
        perfil_usuario=perfil.value,
        guia_uuid=guia_uuid,
        detalhes=(
            f"Arquivo '{Path(nome_arquivo).name}' processado. "
            f"SHA256={sha256_arquivo[:12]}... | Databricks={enviado_databricks}"
        ),
        sucesso=True
    )

    return UploadGuiaResponse(
        mensagem="Guia recebida, auditada e persistida com sucesso.",
        guia_uuid=guia_uuid,
        arquivo_salvo=caminho_imagem_raw.name,
        sha256_arquivo=sha256_arquivo,
        unidade_clinica=resultado_auditoria.unidade_clinica,
        status_auditoria_guia=resultado_auditoria.status_auditoria_guia,
        total_procedimentos=resultado_auditoria.total_procedimentos,
        total_requer_hitl=resultado_auditoria.total_requer_hitl,
        score_medio_guia=resultado_auditoria.score_medio_guia,
        latencia_processamento_ms=resultado_auditoria.latencia_processamento_ms,
    )

# ==============================================================================
# ROTA 2: CONSULTA INDIVIDUAL DE GUIA COM CONTROLE DE VISÃO POR PERFIL (RBAC)
# ==============================================================================

@router.get(
    "/{guia_uuid}",
    response_model=GuiaRecepcaoResponse | GuiaMedicoResponse,
    summary="Consulta o resultado de uma guia auditada conforme o perfil X-User-Role",
)

def consultar_guia(
    guia_uuid: str,
    perfil: PerfilUsuario = Depends(obter_perfil_atual),
) -> GuiaRecepcaoResponse | GuiaMedicoResponse:
    """
    Aplica o Princípio do Menor Privilégio:
    - gestor -> 403 Forbidden
    - recepcao -> dados administrativos (sem CID-10 e sem procedimentos)
    - medico -> dados clínicos completos
    """

    guia = carregar_guia_processada(guia_uuid)

    if guia is None:
        registrar_auditoria(
            evento="CONSULTA_GUIA_INEXISTENTE",
            perfil_usuario=perfil.value,
            guia_uuid=guia_uuid,
            detalhes="Tentativa de consulta a guia_uuid não encontrado no storage.",
            sucesso=False,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guia com UUID '{guia_uuid}' não encontrada.",
        )

    return aplicar_visao_por_perfil(guia, perfil)
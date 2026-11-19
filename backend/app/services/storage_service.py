import hashlib
import logging
from pathlib import Path
from typing import Optional

from backend.app.core.config import STORAGE_DIR, get_settings
from backend.app.models.schemas import GuiaProcessadaOutput

try:
    from databricks.sdk import WorkspaceClient  # type: ignore[import-not-found]
except ImportError:
    WorkspaceClient = None

logger = logging.getLogger(__name__)

RAW_DIR = STORAGE_DIR / "raw"
PROCESSED_DIR = STORAGE_DIR / "processed"


def calcular_sha256(conteudo_bytes: bytes) -> str:
    """Calcula o hash SHA-256 hexadecimal dos bytes da imagem enviada."""
    return hashlib.sha256(conteudo_bytes).hexdigest()


def salvar_imagem_bruta(
    guia_uuid: str,
    nome_original: str,
    conteudo_bytes: bytes,
) -> Path:
    """
    Salva a imagem original na pasta backend/storage/raw/ prefixada com o UUID da guia.
    Retorna o Path completo onde o arquivo foi gravado no disco.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    nome_seguro = Path(nome_original).name or "guia_sem_nome.bin"

    caminho_destino = RAW_DIR / f"{guia_uuid}_{nome_seguro}"

    caminho_destino.write_bytes(conteudo_bytes)
    logger.info("Imagem bruta salva localmente em: %s", caminho_destino)

    return caminho_destino


def _tentar_upload_databricks(caminho_json_local: Path) -> bool:
    """
    Função PRIVADA: tenta enviar o JSON processado para o Unity Catalog Volume do Databricks.
    Se o Databricks não estiver configurado (caso padrão do Community Edition) ou falhar,
    apenas registra um aviso e retorna False sem derrubar a requisição.
    """
    settings = get_settings()

    if not settings.databricks_configured or WorkspaceClient is None:
        logger.info(
            "Databricks Volume não configurado no .env. Mantendo persistência apenas local."
        )
        return False

    try:
        client = WorkspaceClient(
            host=settings.databricks_host,
            token=settings.databricks_token,
        )
        base_volume = settings.databricks_volume_path.rstrip("/")
        caminho_remoto = f"{base_volume}/{caminho_json_local.name}"

        with caminho_json_local.open("rb") as arquivo_binario:
            client.files.upload(caminho_remoto, arquivo_binario, overwrite=True)

        logger.info("JSON enviado com sucesso ao Databricks Volume: %s", caminho_remoto)
        return True
    except Exception as erro:
        logger.warning(
            "Falha ao enviar '%s' para o Databricks (fallback local ativo): %s",
            caminho_json_local.name,
            erro,
        )
        return False


def salvar_json_processado(resultado: GuiaProcessadaOutput) -> tuple[Path, bool]:
    """
    Salva o contrato GuiaProcessadaOutput intacto em backend/storage/processed/{guia_uuid}.json
    e aciona o envio híbrido opcional para o Databricks.
    Retorna uma tupla: (caminho_local_do_json, enviado_para_databricks).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    caminho_json = PROCESSED_DIR / f"{resultado.guia_uuid}.json"

    caminho_json.write_text(resultado.model_dump_json(indent=2), encoding="utf-8")
    logger.info("JSON processado salvo localmente em: %s", caminho_json)

    enviado_nuvem = _tentar_upload_databricks(caminho_json)

    return caminho_json, enviado_nuvem


def carregar_guia_processada(guia_uuid: str) -> Optional[GuiaProcessadaOutput]:
    """
    Busca um JSON processado pelo guia_uuid no disco e o reconstrói como GuiaProcessadaOutput.
    Retorna None se a guia não existir.
    """
    if Path(guia_uuid).name != guia_uuid:
        logger.warning("Tentativa de path traversal bloqueada no guia_uuid: %s", guia_uuid)
        return None

    caminho_json = PROCESSED_DIR / f"{guia_uuid}.json"

    if not caminho_json.exists():
        return None

    conteudo_json = caminho_json.read_text(encoding="utf-8")
    return GuiaProcessadaOutput.model_validate_json(conteudo_json)
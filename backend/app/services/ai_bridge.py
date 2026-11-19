from dotenv import load_dotenv

from backend.app.core.config import BACKEND_DIR
from backend.app.models.schemas import (
    GuiaProcessadaOutput,
    GuiaProcessamentoRequest,
)
from ia_ocr_rag.core.pipeline import processar_guia_end_to_end

# Carrega as variáveis de ambiente (GEMINI_API_KEY, etc.) antes de acionar a IA
load_dotenv(BACKEND_DIR.parent / ".env")
load_dotenv(BACKEND_DIR / ".env")


def auditar_guia(request: GuiaProcessamentoRequest) -> GuiaProcessadaOutput:
    """Aciona o pipeline real de OCR multimodal + RAG Lexical TUSS."""
    return processar_guia_end_to_end(request)
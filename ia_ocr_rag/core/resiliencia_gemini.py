"""
Resiliência das chamadas ao Gemini: novas tentativas (tenacity) + troca de modelo (fallback).

Estratégia, aplicada tanto à extração OCR quanto ao agente do ADK:
  1. Tenta o modelo principal (GEMINI_MODEL_NAME).
  2. Em erro TRANSITÓRIO (429 cota/limite, 500/502/503/504 sobrecarga, timeout de rede),
     repete no mesmo modelo com espera exponencial (2 s, 4 s, ...), até GEMINI_TENTATIVAS_POR_MODELO.
  3. Se o modelo continuar falhando, ou falhar com erro NÃO transitório (ex.: 404 modelo
     descontinuado, resposta inválida), passa para o próximo de GEMINI_FALLBACK_MODELS.
  4. Se todos falharem, a exceção sobe (no OCR, isso aciona o fallback offline RapidOCR).

As tentativas são limitadas de propósito: repetir "até dar certo" travaria a demonstração
e, em erro 429, só consumiria mais cota.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, Callable, List, Optional, Tuple, TypeVar

import httpx
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types
from pydantic import Field
from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

CODIGOS_HTTP_TRANSITORIOS = {429, 500, 502, 503, 504}


def erro_transitorio(exc: BaseException) -> bool:
    """Erros que costumam passar sozinhos e justificam repetir no mesmo modelo."""
    if isinstance(exc, errors.APIError):
        return exc.code in CODIGOS_HTTP_TRANSITORIOS
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def cadeia_de_modelos(principal: Optional[str] = None) -> List[str]:
    """Modelo principal seguido dos modelos reserva do .env, sem repetições."""
    principal = (principal or os.getenv("GEMINI_MODEL_NAME") or "gemini-3.1-flash-lite").strip()
    reservas = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
    return list(dict.fromkeys([principal, *reservas]))


def timeout_ms() -> int:
    return int(float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")) * 1000)


def _politica_tentativas(tentativas: Optional[int] = None) -> dict:
    return dict(
        stop=stop_after_attempt(tentativas or int(os.getenv("GEMINI_TENTATIVAS_POR_MODELO", "2"))),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(erro_transitorio),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class TodosModelosFalharam(RuntimeError):
    """Nenhum modelo da cadeia respondeu."""


def executar_com_fallback(
    chamada: Callable[[str], T],
    modelos: List[str],
    tentativas_por_modelo: Optional[int] = None,
) -> Tuple[T, str]:
    """
    Executa `chamada(modelo)` percorrendo a cadeia de modelos.
    Retorna (resultado, modelo_que_respondeu) para manter a rastreabilidade do motor usado.
    """
    falhas: List[str] = []
    for modelo in modelos:
        try:
            for tentativa in Retrying(**_politica_tentativas(tentativas_por_modelo)):
                with tentativa:
                    resultado = chamada(modelo)
            if modelo != modelos[0]:
                logger.warning("Resposta obtida com o modelo reserva '%s'.", modelo)
            return resultado, modelo
        except Exception as exc:  # noqa: BLE001 - qualquer falha leva ao próximo modelo
            falhas.append(f"{modelo}: {exc}")
            logger.warning("Modelo '%s' falhou (%s). Tentando o próximo da cadeia.", modelo, exc)
    raise TodosModelosFalharam(" | ".join(falhas))


class GeminiComFallback(Gemini):
    """
    Modelo do ADK com novas tentativas e troca automática de modelo.
    Uso: Agent(model=GeminiComFallback(model="principal", modelos_reserva=[...]), ...)
    """

    modelos_reserva: List[str] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        cadeia = list(dict.fromkeys([llm_request.model or self.model, *self.modelos_reserva]))
        ultima_falha: Optional[BaseException] = None

        for modelo in cadeia:
            try:
                async for tentativa in AsyncRetrying(**_politica_tentativas()):
                    with tentativa:
                        # Cópia por tentativa: o ADK altera a requisição ao enviá-la
                        requisicao = llm_request.model_copy(deep=True)
                        requisicao.model = modelo
                        opcoes = requisicao.config.http_options or types.HttpOptions()
                        opcoes.timeout = timeout_ms()
                        requisicao.config.http_options = opcoes
                        # A resposta é acumulada antes de ser repassada: assim uma falha no meio
                        # nunca entrega ao chat uma resposta parcial seguida de outra completa.
                        respostas = [
                            r async for r in super(GeminiComFallback, self).generate_content_async(requisicao, stream)
                        ]
                if modelo != cadeia[0]:
                    logger.warning("Agente respondeu com o modelo reserva '%s'.", modelo)
                for resposta in respostas:
                    yield resposta
                return
            except Exception as exc:  # noqa: BLE001
                ultima_falha = exc
                logger.warning("Agente: modelo '%s' falhou (%s). Tentando o próximo da cadeia.", modelo, exc)

        assert ultima_falha is not None
        raise ultima_falha

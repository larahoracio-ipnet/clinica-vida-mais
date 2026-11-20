"""
Verificação óptica determinística de formulários de marcação (checklist).

Problema: em formulários com uma lista pré-impressa de exames e uma bolinha ao lado de cada
opção, o modelo multimodal (Gemini lite) erra quais bolinhas estão pintadas, principalmente em
linhas vizinhas de itens marcados (ex.: guia_006: incluía "T4 livre" e "Coagulograma", vazios).

Solução (sem IA, reprodutível):
  1. Triagem rápida (~20 ms): procura manchas de tinta de caneta (azul/roxo) com formato de
     bolinha. Sem manchas, a guia não é um checklist e nada muda.
  2. RapidOCR local localiza as linhas de texto e seus códigos TUSS.
  3. Cada bolinha pintada é ligada à linha de texto imediatamente à direita, pelo centro vertical
     mais próximo. Só é checklist se pelo menos 2 bolinhas caírem em linhas com código TUSS.
  4. Os procedimentos finais são exatamente os itens marcados: usa a leitura do Gemini quando o
     código coincide; se o Gemini omitiu o item, usa o texto lido pelo RapidOCR.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from PIL import Image

from ia_ocr_rag.core.schemas import GuiaOCRRaw, ProcedimentoOCRRaw

logger = logging.getLogger(__name__)

MIN_BOLINHAS_CHECKLIST = 2
REGEX_CODIGO = re.compile(r"\d{7,10}")


@dataclass(frozen=True)
class ItemMarcado:
    codigo: str
    texto_linha: str
    x_bolinha: float
    y_bolinha: float


def _mascara_tinta_caneta(rgb: np.ndarray) -> np.ndarray:
    """Pixels de tinta azul/roxa (caneta); o formulário impresso é preto/cinza e fica de fora."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    matiz, saturacao, brilho = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((matiz >= 95) & (matiz <= 150) & (saturacao >= 70) & (brilho <= 200)).astype(np.uint8)


def _manchas_formato_bolinha(mascara: np.ndarray, tamanho_min: float, tamanho_max: float) -> List[tuple]:
    """Componentes conexos quase quadrados e bem preenchidos (bolinha pintada, não letra)."""
    n, _, stats, centros = cv2.connectedComponentsWithStats(mascara, 8)
    manchas = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (
            tamanho_min <= w <= tamanho_max
            and tamanho_min <= h <= tamanho_max
            and 0.6 <= w / h <= 1.6
            and area >= 0.4 * w * h
        ):
            manchas.append((x, y, w, h, float(centros[i][0]), float(centros[i][1])))
    return manchas


def _descricao_da_linha(texto: str, codigo: str) -> str:
    """'OHemograma (40304361)' -> 'Hemograma'."""
    sem_codigo = re.sub(r"\(?\s*" + codigo + r"\s*\)?", " ", texto)
    sem_codigo = re.sub(r"^\s*[Oo○●]\s*(?=[A-ZÁ-Ú])", "", sem_codigo)
    return re.sub(r"\s+", " ", sem_codigo).strip(" -()")


def detectar_itens_marcados(
    imagem: Image.Image,
    obter_rapid_ocr: Callable[[], object],
) -> Optional[List[ItemMarcado]]:
    """
    Retorna os itens com bolinha pintada, na ordem do formulário (coluna da esquerda e depois a
    da direita, de cima para baixo), ou None se a imagem não for um formulário de marcação.
    """
    rgb = np.array(imagem.convert("RGB"))
    lado = max(rgb.shape[:2])
    mascara = _mascara_tinta_caneta(rgb)

    # 1. Triagem rápida: sem manchas com formato de bolinha, não chama o RapidOCR
    if len(_manchas_formato_bolinha(mascara, 0.006 * lado, 0.035 * lado)) < MIN_BOLINHAS_CHECKLIST:
        return None

    # 2. Linhas de texto (RapidOCR local)
    resultado = obter_rapid_ocr()(rgb)
    linhas = []
    for caixa, texto in zip(resultado.boxes if resultado.boxes is not None else [], resultado.txts or ()):
        xs, ys = caixa[:, 0], caixa[:, 1]
        linhas.append((float(xs.min()), float(ys.min()), float(ys.max()), str(texto)))
    if not linhas:
        return None
    altura_linha = float(np.median([y1 - y0 for _, y0, y1, _ in linhas]))

    # 3. Bolinha pintada -> linha de texto à direita com o centro vertical mais próximo
    marcados: List[ItemMarcado] = []
    for x, y, w, h, cx, cy in _manchas_formato_bolinha(mascara, 0.3 * altura_linha, 1.2 * altura_linha):
        candidatas = [
            (abs((y0 + y1) / 2 - cy), texto)
            for lx, y0, y1, texto in linhas
            if lx >= x + w and lx - (x + w) <= 3 * altura_linha and abs((y0 + y1) / 2 - cy) <= 0.6 * altura_linha
        ]
        if not candidatas:
            continue
        _, texto = min(candidatas)
        codigo = REGEX_CODIGO.search(texto.replace(" ", ""))
        if codigo:
            marcados.append(ItemMarcado(codigo.group(0), texto, cx, cy))

    if len(marcados) < MIN_BOLINHAS_CHECKLIST:
        return None

    # Ordem do formulário: agrupa as bolinhas em colunas pela posição horizontal
    marcados.sort(key=lambda m: m.x_bolinha)
    colunas: List[List[ItemMarcado]] = [[marcados[0]]]
    for item in marcados[1:]:
        if item.x_bolinha - colunas[-1][-1].x_bolinha > 0.1 * lado:
            colunas.append([])
        colunas[-1].append(item)
    return [item for coluna in colunas for item in sorted(coluna, key=lambda m: m.y_bolinha)]


def aplicar_itens_marcados(guia_raw: GuiaOCRRaw, marcados: List[ItemMarcado]) -> GuiaOCRRaw:
    """Mantém só os procedimentos marcados, completando com o RapidOCR o que o Gemini omitiu."""
    lidos_por_codigo = {}
    for proc in guia_raw.procedimentos:
        lidos_por_codigo.setdefault(re.sub(r"\D", "", proc.codigo_tuss_ocr), proc)

    procedimentos: List[ProcedimentoOCRRaw] = []
    for seq, item in enumerate(marcados, start=1):
        lido = lidos_por_codigo.get(item.codigo)
        if lido is not None:
            atualizacao = {"sequencial_item": seq}
            # Descrição com OUTRO código entre parênteses = leitura desalinhada (comum no parser
            # offline): usa a descrição da própria linha marcada
            outros_codigos = set(REGEX_CODIGO.findall(lido.descricao_procedimento_ocr.replace(" ", ""))) - {item.codigo}
            if outros_codigos:
                atualizacao["descricao_procedimento_ocr"] = _descricao_da_linha(item.texto_linha, item.codigo)
            procedimentos.append(lido.model_copy(update=atualizacao))
        else:
            procedimentos.append(
                ProcedimentoOCRRaw(
                    sequencial_item=seq,
                    codigo_tuss_ocr=item.codigo,
                    descricao_procedimento_ocr=_descricao_da_linha(item.texto_linha, item.codigo),
                )
            )

    removidos = sorted(set(lidos_por_codigo) - {m.codigo for m in marcados})
    adicionados = sorted({m.codigo for m in marcados} - set(lidos_por_codigo))
    logger.info(
        "Formulário de marcação: %d itens marcados | removidos (bolinha vazia): %s | adicionados: %s",
        len(procedimentos), removidos or "-", adicionados or "-",
    )
    return guia_raw.model_copy(update={"procedimentos": procedimentos})


def verificar_formulario_marcacao(
    caminho_ou_imagem: Path | Image.Image,
    guia_raw: GuiaOCRRaw,
    obter_rapid_ocr: Callable[[], object],
) -> GuiaOCRRaw:
    """Ponto de entrada usado pelo extrator OCR. Guias que não são checklist voltam intactas."""
    try:
        if isinstance(caminho_ou_imagem, Image.Image):
            imagem = caminho_ou_imagem
        else:
            with Image.open(caminho_ou_imagem) as img:
                imagem = img.convert("RGB")
        marcados = detectar_itens_marcados(imagem, obter_rapid_ocr)
    except Exception as exc:  # noqa: BLE001 - a verificação nunca derruba a extração
        logger.warning("Verificação de marcação ignorada (%s).", exc)
        return guia_raw
    return guia_raw if marcados is None else aplicar_itens_marcados(guia_raw, marcados)

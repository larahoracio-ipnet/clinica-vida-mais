"""
Teste de regressão do RAG Lexical: busca de código TUSS somente pela descrição.

Como funciona:
  - Usa como gabarito os procedimentos do lote que casaram por código EXATO
    (o código impresso na guia existe na TUSS, então o par descrição -> código é confiável).
  - Esconde o código e pede ao motor que encontre o procedimento apenas pela descrição.
  - Mede o percentual de acerto (top-1).

O que este script NÃO é:
  - Não é a acurácia oficial da solução. Essa é medida pela frente de Dados (Databricks),
    comparando o lote com o gabarito manual (métricas Pré-RAG e Pós-RAG).
  - O conjunto de validação foi usado para calibrar o motor, então o número tende a ser otimista.
    Use-o para comparar versões do `lexical_rag.py` (antes x depois de uma alteração).

Uso (na raiz clinica-vida-mais/, depois de gerar o lote com batch_processor.py):
  python ia_ocr_rag/scripts/avaliar_rag_descricao.py
  python ia_ocr_rag/scripts/avaliar_rag_descricao.py --detalhes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Garante importação correta mesmo se executado diretamente via python ia_ocr_rag/scripts/...
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from ia_ocr_rag.core.lexical_rag import resolver_caminho_repo
from ia_ocr_rag.core.pipeline import get_lexical_engine
from ia_ocr_rag.core.schemas import GuiaProcessadaOutput, MetodoCorrecaoTUSS, ProcedimentoOCRRaw

load_dotenv(REPO_ROOT / ".env")


def carregar_casos(caminho_lote: Path) -> List[Tuple[str, str, str]]:
    """Retorna (arquivo, descrição lida, código correto) sem repetições de descrição+código."""
    casos: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    for linha in caminho_lote.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        guia = GuiaProcessadaOutput.model_validate_json(linha)
        for p in guia.procedimentos:
            if p.metodo_correcao == MetodoCorrecaoTUSS.EXATO and p.descricao_ocr_bruta.strip():
                chave = (p.descricao_ocr_bruta.strip().lower(), p.codigo_tuss_corrigido or "")
                casos.setdefault(chave, (guia.arquivo_origem, p.descricao_ocr_bruta, p.codigo_tuss_corrigido or ""))
    return list(casos.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Teste de regressão da busca TUSS por descrição")
    parser.add_argument("--detalhes", action="store_true", help="Lista cada erro encontrado")
    parser.add_argument(
        "--lote",
        default=os.path.join(os.getenv("OUTPUT_JSON_DIR", "ia_ocr_rag/data/output_json"), "lote_12_guias.jsonl"),
        help="Arquivo .jsonl gerado pelo batch_processor.py",
    )
    args = parser.parse_args()

    caminho_lote = resolver_caminho_repo(args.lote)
    if not caminho_lote.exists():
        print(f"[ERRO] Lote não encontrado: {caminho_lote}. Rode antes: python ia_ocr_rag/scripts/batch_processor.py")
        return 1

    casos = carregar_casos(caminho_lote)
    if not casos:
        print("[AVISO] Nenhum procedimento com match EXATO no lote para usar como gabarito.")
        return 1

    engine = get_lexical_engine()
    erros = []
    for arquivo, descricao, codigo_correto in casos:
        resultado = engine.validar_procedimento(
            ProcedimentoOCRRaw(sequencial_item=1, codigo_tuss_ocr="", descricao_procedimento_ocr=descricao)
        )
        if resultado.codigo_tuss_corrigido != codigo_correto:
            erros.append((arquivo, descricao, codigo_correto, resultado.codigo_tuss_corrigido, resultado.descricao_tuss_oficial))

    acertos = len(casos) - len(erros)
    print("=" * 80)
    print("RAG LEXICAL | TESTE DE REGRESSÃO DA BUSCA TUSS SÓ POR DESCRIÇÃO")
    print("=" * 80)
    print(f"Lote de referência : {caminho_lote.name}")
    print(f"Casos avaliados    : {len(casos)} (descrição + código únicos com match EXATO)")
    print(f"Acertos (top-1)    : {acertos}/{len(casos)} = {100.0 * acertos / len(casos):.1f}%")

    if args.detalhes and erros:
        print("\nErros:")
        for arquivo, descricao, esperado, obtido, descricao_obtida in erros:
            print(f"  [{arquivo}] {descricao[:45]!r}")
            print(f"      esperado {esperado} | obtido {obtido} {(descricao_obtida or '')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

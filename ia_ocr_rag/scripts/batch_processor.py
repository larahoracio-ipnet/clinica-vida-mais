from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Garante importação correta mesmo se executado diretamente via python ia_ocr_rag/scripts/batch_processor.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ia_ocr_rag.core.lexical_rag import resolver_caminho_repo
from ia_ocr_rag.core.pipeline import processar_guia_end_to_end, salvar_guia_json
from ia_ocr_rag.core.schemas import GuiaProcessadaOutput, GuiaProcessamentoRequest

load_dotenv(REPO_ROOT / ".env")

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

UNIDADES_CLINICA_VIDAMAIS = [
    "UNIDADE_01_MATRIZ",
    "UNIDADE_02_CENTRO",
    "UNIDADE_03_BARRA",
    "UNIDADE_04_TIJUCA",
    "UNIDADE_05_BOTAFOGO",
    "UNIDADE_06_NITEROI",
    "UNIDADE_07_IPANEMA",
    "UNIDADE_08_MEIER",
]


def executar_lote(force_offline: bool = False) -> None:
    input_dir = resolver_caminho_repo(
        os.getenv("INPUT_GUIAS_DIR", "ia_ocr_rag/data/input_guias")
    )
    output_dir = resolver_caminho_repo(
        os.getenv("OUTPUT_JSON_DIR", "ia_ocr_rag/data/output_json")
    )
    sleep_seconds = float(os.getenv("BATCH_SLEEP_SECONDS", "4.5"))

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de guias não encontrado: {input_dir}")

    arquivos = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSOES_IMAGEM
    )

    if not arquivos:
        print(f"[AVISO] Nenhuma imagem encontrada em: {input_dir}")
        return

    print("=" * 80)
    print(f"CLÍNICA VIDA+ | PROCESSAMENTO BATCH DE GUIAS TISS/TUSS ({len(arquivos)} guias)")
    print(f"Modo Offline Forçado: {force_offline} | Intervalo Rate-Limit: {sleep_seconds}s")
    print("=" * 80)

    resultados: List[GuiaProcessadaOutput] = []

    for idx, arquivo in enumerate(arquivos):
        guia_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"clinica-vidamais-{arquivo.name}"))
        unidade = UNIDADES_CLINICA_VIDAMAIS[idx % len(UNIDADES_CLINICA_VIDAMAIS)]

        req = GuiaProcessamentoRequest(
            guia_uuid=guia_uuid,
            arquivo_nome=arquivo.name,
            caminho_arquivo=str(arquivo),
            unidade_clinica=unidade,
        )

        print(f"\n[{idx + 1:02d}/{len(arquivos):02d}] Processando '{arquivo.name}' ({unidade})...")
        res = processar_guia_end_to_end(req, force_offline=force_offline, salvar_ocr_bruto=True)
        json_path = salvar_guia_json(res, output_dir=output_dir)
        resultados.append(res)

        print(
            f"   -> Motor: {res.motor_ocr_utilizado} | Procs: {res.total_procedimentos} | "
            f"Score Médio: {res.score_medio_guia:.1f}% | Status: {res.status_auditoria_guia.value} | "
            f"Latência: {res.latencia_processamento_ms:.0f}ms"
        )
        print(f"   -> JSON individual: {json_path.name}")

        # Respeita rate-limit do Gemini Free Tier entre chamadas remotas
        if idx < len(arquivos) - 1 and not force_offline and res.motor_ocr_utilizado != "rapidocr_offline_fallback":
            time.sleep(sleep_seconds)

    # Gera arquivo consolidado .jsonl (1 objeto JSON por linha -> ideal para Databricks Bronze)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "lote_12_guias.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in resultados:
            f.write(r.model_dump_json() + "\n")

    total_procs_lote = sum(r.total_procedimentos for r in resultados)
    total_hitl_lote = sum(r.total_requer_hitl for r in resultados)
    media_geral = (
        sum(r.score_medio_guia for r in resultados) / len(resultados)
        if resultados
        else 0.0
    )

    print("\n" + "=" * 80)
    print("RESUMO EXECUTIVO DO LOTE (HANDOFF PARA DATABRICKS BRONZE - DIMITRI)")
    print("=" * 80)
    print(f"Total de guias processadas : {len(resultados)}")
    print(f"Total de procedimentos     : {total_procs_lote}")
    print(f"Itens marcados p/ HITL     : {total_hitl_lote} (score < 90% ou não encontrado)")
    print(f"Score médio de confiança   : {media_geral:.2f}%")
    print(f"Arquivo consolidado JSONL  : {jsonl_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processamento em lote das 12 guias TISS/TUSS")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Força o uso do motor RapidOCR local sem consumir cota da API Gemini.",
    )
    args = parser.parse_args()
    executar_lote(force_offline=args.offline)
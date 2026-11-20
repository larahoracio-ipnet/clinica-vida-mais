"""
Avaliação do lote processado contra o gabarito manual (data/gabarito/gabarito.csv).

Mede, guia a guia e no total:
  1. EXTRAÇÃO (motor multimodal): nome e carteirinha do paciente (como lidos e após a
     validação no cadastro), itens encontrados (recall/precisão), código lido (Pré-RAG)
     e fidelidade da descrição lida.
  2. CORREÇÃO TUSS (RAG lexical): código corrigido (Pós-RAG) comparado ao gabarito.
  3. REVISÃO HUMANA: se o roteamento automático protegeu contra códigos errados.

Alinhamento: os itens do gabarito e do OCR são pareados por similaridade (código e descrição)
respeitando a ordem da guia, para que um item pulado ou inventado não desalinhe os demais.

Entradas: lote_12_guias.jsonl + output_json/ocr_bruto/ (mesma execução do batch_processor.py).
Saída: relatório no terminal + CSV por item (sem dados pessoais) em output_json/.

Uso (na raiz clinica-vida-mais/):
  python ia_ocr_rag/scripts/avaliar_gabarito.py
  python ia_ocr_rag/scripts/avaliar_gabarito.py --detalhes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from rapidfuzz import fuzz

from ia_ocr_rag.core.lexical_rag import (
    extrair_apenas_digitos,
    remover_acentos,
    remover_zeros_esquerda_excedentes,
    resolver_caminho_repo,
)
from ia_ocr_rag.core.schemas import GuiaProcessadaOutput, ProcedimentoValidadoOutput

load_dotenv(REPO_ROOT / ".env")

LIMIAR_PAREAMENTO = 60.0  # similaridade mínima para considerar que é o mesmo item


# ==============================================================================
# Normalizações
# ==============================================================================
def norm_texto(valor: Optional[str]) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", remover_acentos((valor or "").lower())))


def codigo_oficial(valor: Optional[str]) -> str:
    """Código TUSS de 8 dígitos (remove zeros à esquerda de preenchimento, ex.: 0040302733)."""
    return remover_zeros_esquerda_excedentes(extrair_apenas_digitos(valor))


def arquivo_da_guia(id_guia: str) -> str:
    """guia_007 -> guia_007.png (o nome do arquivo é o próprio id_guia do gabarito)."""
    return f"{id_guia}.png"


# ==============================================================================
# Estruturas
# ==============================================================================
@dataclass
class ItemGabarito:
    item: int
    codigo: str
    descricao: str


@dataclass
class GuiaGabarito:
    id_guia: str
    nome: str
    carteirinha: str
    itens: List[ItemGabarito] = field(default_factory=list)


@dataclass
class Pareamento:
    gab: Optional[ItemGabarito]
    ocr: Optional[ProcedimentoValidadoOutput]


def carregar_gabarito(caminho: Path) -> Dict[str, GuiaGabarito]:
    texto = caminho.read_text(encoding="utf-8-sig")
    delimitador = ";" if texto.count(";") > texto.count(",") else ","
    guias: Dict[str, GuiaGabarito] = {}
    for row in csv.DictReader(texto.splitlines(), delimiter=delimitador):
        gid = row["id_guia"].strip()
        g = guias.setdefault(gid, GuiaGabarito(gid, (row.get("nome") or "").strip(), (row.get("carteirinha") or "").strip()))
        g.itens.append(ItemGabarito(int(row["item"]), (row.get("codigo_tuss") or "").strip(), (row.get("descricao") or "").strip()))
    for g in guias.values():
        g.itens.sort(key=lambda i: i.item)
    return guias


def similaridade(gab: ItemGabarito, ocr: ProcedimentoValidadoOutput) -> float:
    """
    Similaridade para parear itens. O código só conta quando é idêntico: códigos TUSS da mesma
    família (ex.: 4030xxxx) são parecidos entre si e gerariam pareamentos falsos.
    """
    s_desc = float(fuzz.token_set_ratio(norm_texto(gab.descricao), norm_texto(ocr.descricao_ocr_bruta)))
    cg, co = codigo_oficial(gab.codigo), codigo_oficial(ocr.codigo_tuss_ocr_bruto)
    s_cod = 100.0 if (cg and co and cg == co) else 0.0
    return max(s_desc, s_cod)


def alinhar(gabs: List[ItemGabarito], ocrs: List[ProcedimentoValidadoOutput]) -> List[Pareamento]:
    """Alinhamento ótimo que preserva a ordem (programação dinâmica, como um diff ponderado)."""
    n, m = len(gabs), len(ocrs)
    sim = [[similaridade(g, o) for o in ocrs] for g in gabs]
    melhor = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            casar = sim[i][j] + melhor[i + 1][j + 1] if sim[i][j] >= LIMIAR_PAREAMENTO else -1.0
            melhor[i][j] = max(casar, melhor[i + 1][j], melhor[i][j + 1])
    for i in range(n - 1, -1, -1):
        melhor[i][m] = 0.0
    pares: List[Pareamento] = []
    i = j = 0
    while i < n and j < m:
        if sim[i][j] >= LIMIAR_PAREAMENTO and melhor[i][j] == sim[i][j] + melhor[i + 1][j + 1]:
            pares.append(Pareamento(gabs[i], ocrs[j])); i += 1; j += 1
        elif melhor[i][j] == melhor[i + 1][j]:
            pares.append(Pareamento(gabs[i], None)); i += 1
        else:
            pares.append(Pareamento(None, ocrs[j])); j += 1
    pares += [Pareamento(g, None) for g in gabs[i:]] + [Pareamento(None, o) for o in ocrs[j:]]
    return pares


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else "n/a"


# ==============================================================================
# Avaliação
# ==============================================================================
def main() -> int:
    out_dir = os.getenv("OUTPUT_JSON_DIR", "ia_ocr_rag/data/output_json")
    parser = argparse.ArgumentParser(description="Avaliação do lote contra o gabarito manual")
    parser.add_argument("--gabarito", default=os.getenv("GABARITO_CSV_PATH", "ia_ocr_rag/data/gabarito/gabarito.csv"))
    parser.add_argument("--lote", default=os.path.join(out_dir, "lote_12_guias.jsonl"))
    parser.add_argument("--detalhes", action="store_true", help="Mostra o pareamento item a item de cada guia")
    args = parser.parse_args()

    caminho_gab = resolver_caminho_repo(args.gabarito)
    caminho_lote = resolver_caminho_repo(args.lote)
    pasta_bruta = resolver_caminho_repo(out_dir) / "ocr_bruto"
    for c in (caminho_gab, caminho_lote):
        if not c.exists():
            print(f"[ERRO] Arquivo não encontrado: {c}")
            return 1

    gabarito = carregar_gabarito(caminho_gab)
    lote = {
        g.arquivo_origem: g
        for g in (GuiaProcessadaOutput.model_validate_json(l) for l in caminho_lote.read_text(encoding="utf-8").splitlines() if l.strip())
    }

    c = defaultdict(int)  # contadores globais
    linhas_csv: List[Dict[str, object]] = []
    resumo_guias: List[Tuple[str, str, str, str, str]] = []

    for gid in sorted(gabarito):
        gab = gabarito[gid]
        arquivo = arquivo_da_guia(gid)
        guia = lote.get(arquivo)
        if guia is None:
            print(f"[AVISO] {gid} ({arquivo}) não está no lote.")
            continue

        # ---------------- Cabeçalho: nome e carteirinha (extração bruta) ----------------
        guia_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"clinica-vidamais-{arquivo}"))
        bruto_path = pasta_bruta / f"{guia_uuid}.json"
        extracao = json.loads(bruto_path.read_text(encoding="utf-8"))["extracao"] if bruto_path.exists() else None
        if extracao is not None:
            if gab.nome:
                c["nome_total"] += 1
                sim_nome = fuzz.ratio(norm_texto(gab.nome), norm_texto(extracao.get("nome_paciente_ocr")))
                c["nome_exato"] += sim_nome == 100
                c["nome_aprox"] += sim_nome >= 85
            if gab.carteirinha:
                c["cart_total"] += 1
                c["cart_ok"] += extrair_apenas_digitos(extracao.get("carteira_plano_ocr")) == extrair_apenas_digitos(gab.carteirinha)
        else:
            c["sem_bruto"] += 1

        # Paciente após a validação no cadastro (ex.: nome em Hangul -> grafia latina do cadastro)
        if gab.nome:
            c["nome_cad_total"] += 1
            c["nome_cad_ok"] += fuzz.ratio(norm_texto(gab.nome), norm_texto(guia.paciente.nome_paciente_cadastro)) >= 85
        if gab.carteirinha:
            c["cart_cad_total"] += 1
            c["cart_cad_ok"] += extrair_apenas_digitos(guia.paciente.carteira_plano_cadastro) == extrair_apenas_digitos(gab.carteirinha)

        # ---------------- Procedimentos ----------------
        pares = alinhar(gab.itens, guia.procedimentos)
        g_enc = g_pre = g_pos = 0
        g_com_codigo = sum(1 for i in gab.itens if i.codigo)
        for par in pares:
            g_item, o_item = par.gab, par.ocr
            linha = {
                "id_guia": gid, "arquivo": arquivo,
                "item_gabarito": g_item.item if g_item else "", "seq_ocr": o_item.sequencial_item if o_item else "",
                "descricao_gabarito": g_item.descricao if g_item else "", "descricao_lida": o_item.descricao_ocr_bruta if o_item else "",
                "codigo_gabarito": g_item.codigo if g_item else "", "codigo_lido": o_item.codigo_tuss_ocr_bruto if o_item else "",
                "codigo_corrigido": (o_item.codigo_tuss_corrigido or "") if o_item else "",
                "metodo": o_item.metodo_correcao.value if o_item else "", "score": o_item.score_confianca_lexical if o_item else "",
                "revisao_humana": o_item.requer_revisao_humana if o_item else "",
            }
            if g_item and not o_item:
                linha["situacao"] = "NAO_EXTRAIDO"
                c["itens_perdidos"] += 1
            elif o_item and not g_item:
                linha["situacao"] = "EXTRA_NAO_ESTA_NO_GABARITO"
                c["itens_extras"] += 1
            else:
                linha["situacao"] = "PAREADO"
                g_enc += 1
                c["desc_sim_soma"] += fuzz.ratio(norm_texto(g_item.descricao), norm_texto(o_item.descricao_ocr_bruta))
                c["desc_n"] += 1
                if g_item.codigo:
                    pre_literal = o_item.codigo_tuss_ocr_bruto.strip() == g_item.codigo
                    pre_digitos = codigo_oficial(o_item.codigo_tuss_ocr_bruto) == codigo_oficial(g_item.codigo)
                    pos = (o_item.codigo_tuss_corrigido or "") == codigo_oficial(g_item.codigo)
                    g_pre += pre_literal; g_pos += pos
                    c["pre_literal"] += pre_literal; c["pre_digitos"] += pre_digitos; c["pos"] += pos
                    linha.update(pre_rag_ok=pre_literal, pos_rag_ok=pos)
                    # Matriz de roteamento
                    chave = ("certo" if pos else "errado") + ("_revisao" if o_item.requer_revisao_humana else "_auto")
                    c[chave] += 1
                    c["rag_corrigiu"] += (not pre_digitos) and pos
                    c["rag_estragou"] += pre_digitos and not pos
                else:
                    c["pareado_sem_codigo"] += 1
                    c["sem_codigo_revisao"] += o_item.requer_revisao_humana
            linhas_csv.append(linha)

        c["gab_itens"] += len(gab.itens)
        c["gab_com_codigo"] += g_com_codigo
        c["ocr_itens"] += len(guia.procedimentos)
        c["pareados"] += g_enc
        resumo_guias.append((
            gid, f"{g_enc}/{len(gab.itens)}", f"{len(guia.procedimentos)}",
            f"{g_pre}/{g_com_codigo}" if g_com_codigo else "sem código",
            f"{g_pos}/{g_com_codigo}" if g_com_codigo else "sem código",
        ))

        if args.detalhes:
            print(f"\n== {gid} ({arquivo})")
            for par in pares:
                gi, oi = par.gab, par.ocr
                esq = f"[{gi.item:2}] {gi.codigo or '—':>10} {gi.descricao[:30]:30}" if gi else " " * 47
                dir_ = f"{oi.codigo_tuss_ocr_bruto or '—':>10} → {oi.codigo_tuss_corrigido or 'None':8} {oi.descricao_ocr_bruta[:26]}" if oi else "(não extraído)"
                print(f"   {esq} | {dir_}")

    # ---------------- Relatório ----------------
    print("\n" + "=" * 86)
    print("AVALIAÇÃO CONTRA O GABARITO MANUAL | EXTRAÇÃO (OCR) + CORREÇÃO TUSS (RAG LEXICAL)")
    print("=" * 86)
    print(f"Gabarito: {caminho_gab.name} ({c['gab_itens']} itens em {len(gabarito)} guias) | Lote: {caminho_lote.name}")

    print("\n1) EXTRAÇÃO PELO MOTOR MULTIMODAL")
    print(f"   Nome do paciente (exato)        : {pct(c['nome_exato'], c['nome_total'])}")
    print(f"   Nome do paciente (≥85% similar) : {pct(c['nome_aprox'], c['nome_total'])}")
    print(f"   Carteirinha (17 dígitos exatos) : {pct(c['cart_ok'], c['cart_total'])}")
    print(f"   Após validação no cadastro      : nome {pct(c['nome_cad_ok'], c['nome_cad_total'])} | "
          f"carteirinha {pct(c['cart_cad_ok'], c['cart_cad_total'])}")
    print(f"   Itens encontrados (recall)      : {pct(c['pareados'], c['gab_itens'])}")
    print(f"   Itens corretos (precisão)       : {pct(c['pareados'], c['ocr_itens'])}")
    print(f"   Itens não extraídos             : {c['itens_perdidos']}")
    print(f"   Itens extras (fora do gabarito) : {c['itens_extras']}")
    if c["desc_n"]:
        print(f"   Fidelidade da descrição lida    : {c['desc_sim_soma'] / c['desc_n']:.1f}% de similaridade média")
    if c["sem_bruto"]:
        print(f"   [AVISO] {c['sem_bruto']} guia(s) sem extração bruta em ocr_bruto/: nome/carteirinha não avaliados.")

    base = c["gab_com_codigo"]
    print(f"\n2) CÓDIGO TUSS ({base} itens do gabarito com código; item não extraído conta como erro)")
    print(f"   Pré-RAG, código lido literal    : {pct(c['pre_literal'], base)}")
    print(f"   Pré-RAG, só dígitos (sem zeros) : {pct(c['pre_digitos'], base)}")
    print(f"   Pós-RAG, código corrigido       : {pct(c['pos'], base)}")
    print(f"   RAG corrigiu leitura errada     : {c['rag_corrigiu']}")
    print(f"   RAG estragou leitura correta    : {c['rag_estragou']}")

    print("\n3) ROTEAMENTO / REVISÃO HUMANA (itens pareados com código no gabarito)")
    print(f"   Certo e aprovado automaticamente : {c['certo_auto']}")
    print(f"   Certo, mas enviado para revisão  : {c['certo_revisao']}")
    print(f"   Errado e enviado para revisão    : {c['errado_revisao']}  (proteção funcionou)")
    print(f"   ERRADO E APROVADO AUTOMATICAMENTE: {c['errado_auto']}  (risco de glosa)")
    if c["pareado_sem_codigo"]:
        print(f"   Itens sem código no gabarito     : {c['pareado_sem_codigo']} (só descrição; "
              f"{c['sem_codigo_revisao']} em revisão humana; Pós-RAG não avaliável)")

    print("\n4) POR GUIA")
    print(f"   {'guia':10} {'encontrados':>12} {'lidos OCR':>10} {'Pré-RAG':>12} {'Pós-RAG':>12}")
    for gid, enc, lidos, pre, pos in resumo_guias:
        print(f"   {gid:10} {enc:>12} {lidos:>10} {pre:>12} {pos:>12}")

    saida_csv = resolver_caminho_repo(out_dir) / "avaliacao_gabarito_itens.csv"
    campos = ["id_guia", "arquivo", "situacao", "item_gabarito", "seq_ocr", "codigo_gabarito", "codigo_lido",
              "codigo_corrigido", "metodo", "score", "revisao_humana", "pre_rag_ok", "pos_rag_ok",
              "descricao_gabarito", "descricao_lida"]
    with saida_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"\nDetalhe por item (sem dados pessoais): {saida_csv}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

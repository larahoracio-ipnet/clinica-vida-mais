from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

from ia_ocr_rag.core.schemas import (
    MetodoCorrecaoTUSS,
    PacienteOutput,
    ProcedimentoOCRRaw,
    ProcedimentoValidadoOutput,
)

# Raiz do monorepo clinica-vida-mais/ (2 níveis acima de ia_ocr_rag/core/lexical_rag.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Mapa de substituição de confusões ópticas típicas em guias escaneadas
MAPA_RUIDO_OPTICO: Dict[str, str] = {
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "L": "1",
    "|": "1",
    "!": "1",
    "Z": "2",
    "z": "2",
    "S": "5",
    "s": "5",
    "G": "6",
    "b": "6",
    "T": "7",
    "B": "8",
    "g": "9",
    "q": "9",
}

STOPWORDS_PT = {
    "de", "da", "do", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "e", "ou", "a", "o", "as", "os", "um", "uma",
}


# Termos genéricos da nomenclatura TUSS que não distinguem um procedimento de outro
TERMOS_GENERICOS_TUSS = {"pesquisa", "dosagem", "inclui", "cada", "qualquer", "ou", "e", "serico", "serica"}

# Sinônimos e abreviações clínicas -> termos da nomenclatura oficial TUSS.
# Chaves e valores já normalizados (minúsculas, sem acento, sem stopwords).
# É uma expansão de consulta determinística e auditável (não há inferência de modelo).
SINONIMOS_CLINICOS: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("glicemia", "jejum"): ("glicose",),
    ("glicemia",): ("glicose",),
    ("hemograma", "completo"): ("hemograma",),
    ("hemoglobina", "glicada"): ("hemoglobina", "glicada", "fracao", "a1c"),
    ("hba1c",): ("hemoglobina", "glicada", "fracao", "a1c"),
    ("tsh",): ("tireoestimulante", "hormonio", "tsh"),
    ("t4l",): ("t4", "livre"),
    ("t3l",): ("t3", "livre"),
    ("ggt",): ("gama", "glutamil", "transferase"),
    ("gama", "gt"): ("gama", "glutamil", "transferase"),
    ("tgo",): ("transaminase", "oxalacetica"),
    ("tgp",): ("transaminase", "piruvica"),
    ("pcr",): ("proteina", "c", "reativa"),
    ("cpk",): ("creatino", "fosfoquinase", "total"),
    ("ck",): ("creatino", "fosfoquinase", "total"),
    ("psa",): ("antigeno", "especifico", "prostatico", "psa"),
    ("vhs",): ("hemossedimentacao",),
    ("parcial", "urina"): ("rotina", "urina"),
    ("urina", "tipo", "1"): ("rotina", "urina"),
    ("sumario", "urina"): ("rotina", "urina"),
    ("eas",): ("rotina", "urina"),
    ("perfil", "lipidico", "completo"): ("perfil", "lipidico", "lipidograma"),
    ("lipidograma",): ("perfil", "lipidico", "lipidograma"),
    ("colesterol", "total", "fracoes"): ("perfil", "lipidico", "lipidograma"),
    ("anti", "tireoglobulina"): ("anticorpos", "antitireoide", "tireoglobulina"),
    ("vitamina", "d"): ("vitamina", "d", "25", "hidroxi"),
    ("ecografia",): ("us",),
    ("ultrassom",): ("us",),
    ("ultrassonografia",): ("us",),
    ("usg",): ("us",),
    ("abdominal",): ("abdome",),
    ("campo", "visual"): ("campimetria",),
    ("endoscopia", "digestiva", "alta"): ("endoscopia", "digestiva", "alta"),
    ("endoscopia",): ("endoscopia", "digestiva", "alta"),
    # Exames de imagem: a TUSS usa as siglas "RX -", "US -", "RM -" e "TC -"
    ("ressonancia", "magnetica"): ("rm",),
    ("ressonancia",): ("rm",),
    ("rnm",): ("rm",),
    ("raio", "x"): ("rx",),
    ("radiografia",): ("rx",),
    ("tomografia", "coerencia", "optica"): ("tomografia", "coerencia", "optica"),
    ("oct",): ("tomografia", "coerencia", "optica"),
    ("tomografia", "computadorizada"): ("tc",),
    ("tomografia",): ("tc",),
    # Articulações: a TUSS codifica RM/US articular "por articulação" (o termo original é mantido)
    ("ombro",): ("ombro", "articular"),
    ("joelho",): ("joelho", "articular"),
    ("cotovelo",): ("cotovelo", "articular"),
    ("punho",): ("punho", "articular"),
    ("tornozelo",): ("tornozelo", "articular"),
    ("quadril",): ("quadril", "articular"),
}
_MAIOR_CHAVE_SINONIMO = max(len(k) for k in SINONIMOS_CLINICOS)


def resolver_caminho_repo(caminho: str | Path) -> Path:
    """Resolve caminhos relativos a partir da raiz clinica-vida-mais/."""
    p = Path(caminho)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class RegistroTUSS:
    id: str
    source: str
    display_name: str
    display_name_norm: str
    vigencia_ativa: bool
    tokens_completos: frozenset = frozenset()
    tokens_nucleo: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RegistroPaciente:
    nome_paciente: str
    carteira_do_plano: str


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def tokenizar_texto_medico(texto: str) -> List[str]:
    if not texto:
        return []
    limpo = remover_acentos(texto.lower())
    tokens = re.findall(r"[a-z0-9]+", limpo)
    return [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]


def tokenizar_descricao_tuss(texto: str) -> List[str]:
    """
    Tokenização para a busca TUSS. Diferente de `tokenizar_texto_medico`, mantém tokens de
    1 caractere (ex.: vitamina "D", proteína "C"), que distinguem procedimentos.
    """
    if not texto:
        return []
    limpo = remover_acentos(texto.lower())
    return [t for t in re.findall(r"[a-z0-9]+", limpo) if t not in STOPWORDS_PT]


def expandir_sinonimos_clinicos(tokens: Sequence[str]) -> List[str]:
    """Substitui abreviações/termos coloquiais pelos termos oficiais (casamento mais longo primeiro)."""
    resultado: List[str] = []
    i = 0
    while i < len(tokens):
        for tamanho in range(min(_MAIOR_CHAVE_SINONIMO, len(tokens) - i), 0, -1):
            chave = tuple(tokens[i:i + tamanho])
            if chave in SINONIMOS_CLINICOS:
                resultado.extend(SINONIMOS_CLINICOS[chave])
                i += tamanho
                break
        else:
            resultado.append(tokens[i])
            i += 1
    return resultado


def _peso_presenca(token: str, conjunto: Set[str]) -> float:
    """
    Quanto `token` está presente em `conjunto`:
      1.0 = igual; 0.9 = aproximado (erro de digitação/OCR, 5+ letras);
      0.8 = parte de uma palavra composta (ex.: 'foto' em 'estereofoto'); 0.0 = ausente.
    Casamento exato sempre vence o aproximado (ex.: 'creatinina' x 'creatina').
    """
    if token in conjunto:
        return 1.0
    if len(token) >= 5 and any(len(t) >= 5 and fuzz.ratio(token, t) >= 85 for t in conjunto):
        return 0.9
    if len(token) >= 4 and any(len(t) > len(token) and token in t for t in conjunto):
        return 0.8
    return 0.0


def separar_descricao_principal(texto: str) -> Tuple[str, str]:
    """Separa o nome principal do detalhamento entre parênteses (que pesa menos na busca)."""
    detalhes = " ".join(re.findall(r"\(([^)]*)\)?", texto or ""))
    principal = re.sub(r"\([^)]*\)?", " ", texto or "")
    return principal, detalhes


def normalizar_texto_comparacao(texto: str) -> str:
    return " ".join(tokenizar_texto_medico(texto))


def extrair_apenas_digitos(valor: Optional[str]) -> str:
    if not valor:
        return ""
    return re.sub(r"\D", "", valor)


def normalizar_ruido_optico_numerico(codigo_ocr: Optional[str]) -> str:
    """Converte letras confundidas pelo OCR em dígitos antes de limpar caracteres."""
    if not codigo_ocr:
        return ""
    convertido = "".join(MAPA_RUIDO_OPTICO.get(ch, ch) for ch in codigo_ocr.strip())
    return re.sub(r"\D", "", convertido)


def remover_zeros_esquerda_excedentes(digitos: str, tamanho: int = 8) -> str:
    """
    Guias impressas costumam preencher o campo de código com zeros à esquerda
    (ex.: '0040302733' -> '40302733'). Só remove quando o excedente é composto
    exclusivamente por zeros; o código bruto original segue preservado no output.
    """
    if len(digitos) > tamanho and set(digitos[:-tamanho]) == {"0"}:
        return digitos[-tamanho:]
    return digitos


# ------------------------------------------------------------------------------
# Nomes em alfabeto coreano (Hangul) -> alfabeto latino, para comparar com o cadastro
# ------------------------------------------------------------------------------
# Cada sílaba Hangul é decomposta deterministicamente em inicial + vogal + final (Unicode),
# romanizada pela Romanização Revisada e reordenada para "Nome Sobrenome", como no cadastro
# (ex.: 강민준 -> "Min-jun Kang"). Sobrenomes usam a grafia popular (Kang, Kim, Lee, Park...).
_HANGUL_INICIAIS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_HANGUL_VOGAIS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_HANGUL_FINAIS = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t"]

SOBRENOMES_COREANOS: Dict[str, str] = {
    "김": "Kim", "이": "Lee", "박": "Park", "최": "Choi", "정": "Jung", "강": "Kang", "조": "Cho",
    "윤": "Yoon", "장": "Jang", "임": "Lim", "한": "Han", "오": "Oh", "서": "Seo", "신": "Shin",
    "권": "Kwon", "황": "Hwang", "안": "Ahn", "송": "Song", "류": "Ryu", "유": "Yoo", "전": "Jeon",
    "홍": "Hong", "고": "Ko", "문": "Moon", "양": "Yang", "손": "Son", "배": "Bae", "백": "Baek",
    "허": "Heo", "노": "Noh", "남": "Nam", "심": "Shim", "하": "Ha", "곽": "Kwak", "성": "Sung",
    "차": "Cha", "주": "Joo", "우": "Woo", "구": "Koo", "민": "Min", "진": "Jin", "나": "Na",
    "엄": "Um", "원": "Won", "천": "Chun", "방": "Bang", "공": "Kong", "현": "Hyun", "변": "Byun",
    "황보": "Hwangbo", "남궁": "Namgoong", "선우": "Sunwoo", "제갈": "Jegal",
}


def contem_hangul(texto: Optional[str]) -> bool:
    return bool(texto) and any("\uac00" <= c <= "\ud7a3" for c in texto)


def _romanizar_silabas_hangul(texto: str) -> List[str]:
    silabas: List[str] = []
    for c in texto:
        indice = ord(c) - 0xAC00
        if 0 <= indice <= 11171:
            inicial, resto = divmod(indice, 21 * 28)
            vogal, final = divmod(resto, 28)
            silabas.append(_HANGUL_INICIAIS[inicial] + _HANGUL_VOGAIS[vogal] + _HANGUL_FINAIS[final])
    return silabas


def romanizar_nome_coreano(nome: str) -> str:
    """Ex.: '강민준' -> 'Min-jun Kang'; '강 태현' -> 'Tae-hyeon Kang'."""
    partes = nome.split()
    if len(partes) >= 2:
        sobrenome_hg, prenome_hg = partes[0], "".join(partes[1:])
    else:
        compacto = partes[0] if partes else ""
        tamanho = 2 if len(compacto) >= 3 and compacto[:2] in SOBRENOMES_COREANOS else 1
        sobrenome_hg, prenome_hg = compacto[:tamanho], compacto[tamanho:]

    sobrenome = SOBRENOMES_COREANOS.get(sobrenome_hg) or "".join(_romanizar_silabas_hangul(sobrenome_hg)).capitalize()
    prenome = "-".join(_romanizar_silabas_hangul(prenome_hg)).capitalize()
    return f"{prenome} {sobrenome}".strip()


def chave_fonetica_coreana(nome_latino: str) -> str:
    """
    Aproxima as grafias de um mesmo nome coreano em alfabeto latino
    (ex.: 'Tae-hyeon' x 'Tae-hyun', 'Ji-hun' x 'Ji-hoon', 'Hyeon-u' x 'Hyun-woo', 'Gang' x 'Kang').
    """
    tokens = []
    for token in remover_acentos(nome_latino.lower()).split():
        t = re.sub(r"[^a-z]", "", token)
        for antes, depois in (("oo", "u"), ("wu", "u"), ("eo", "u"), ("ee", "i"), ("sh", "s"),
                              ("k", "g"), ("p", "b"), ("t", "d")):
            t = t.replace(antes, depois)
        if t:
            tokens.append(t)
    return " ".join(tokens)


class LexicalTUSSEngine:
    """
    Motor de RAG Não-Vetorial (Lexical) em memória para:
      1. Correção e padronização de códigos TUSS (Exato + Regra Visual + Fuzzy + BM25).
      2. Validação cadastral de beneficiários (pacientes_simulados.csv).
    """

    def __init__(
        self,
        tuss_csv_path: str | Path,
        pacientes_csv_path: str | Path,
        hitl_threshold: float = 90.0,
    ) -> None:
        self.tuss_csv_path = resolver_caminho_repo(tuss_csv_path)
        self.pacientes_csv_path = resolver_caminho_repo(pacientes_csv_path)
        self.hitl_threshold = float(hitl_threshold)

        self.tuss_by_id: Dict[str, RegistroTUSS] = {}
        self.tuss_list: List[RegistroTUSS] = []
        self.tuss_ids: List[str] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.idf_tokens: Dict[str, float] = {}
        self.nucleos_norm: List[str] = []

        self.pacientes_by_carteira: Dict[str, RegistroPaciente] = {}
        self.pacientes_list: List[RegistroPaciente] = []

        self._carregar_base_tuss()
        self._carregar_base_pacientes()

    @staticmethod
    def _abrir_csv_seguro(caminho: Path) -> List[Dict[str, str]]:
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo de referência não encontrado: {caminho}")
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with caminho.open("r", encoding=encoding, newline="") as f:
                    sample = f.read(2048)
                    f.seek(0)
                    delimiter = ";" if sample.count(";") > sample.count(",") else ","
                    reader = csv.DictReader(f, delimiter=delimiter)
                    return list(reader)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Não foi possível decodificar o CSV: {caminho}")

    def _carregar_base_tuss(self) -> None:
        linhas = self._abrir_csv_seguro(self.tuss_csv_path)
        corpus_tokens: List[List[str]] = []

        for row in linhas:
            raw_id = extrair_apenas_digitos(row.get("id", ""))
            if not raw_id:
                continue
            tuss_id = raw_id.zfill(8)
            # Aplica .strip() obrigatório conforme especificado para display_name
            display_name = (row.get("display_name") or "").strip()
            source = (row.get("source") or "").strip()
            fim_vigencia = (row.get("extras_fim_vigencia") or "").strip()
            # A base oficial usa "-" para indicar ausência de data de fim de vigência
            vigencia_ativa = fim_vigencia in ("", "-")

            tokens_completos = tokenizar_descricao_tuss(display_name)
            # Núcleo = nome principal (sem parênteses) sem os termos genéricos da TUSS
            nome_principal = re.sub(r"\([^)]*\)?", " ", display_name)
            tokens_nucleo = tuple(
                t for t in tokenizar_descricao_tuss(nome_principal) if t not in TERMOS_GENERICOS_TUSS
            ) or tuple(tokens_completos)

            reg = RegistroTUSS(
                id=tuss_id,
                source=source,
                display_name=display_name,
                display_name_norm=normalizar_texto_comparacao(display_name),
                vigencia_ativa=vigencia_ativa,
                tokens_completos=frozenset(
                    tokens_completos + [a + b for a, b in zip(tokens_completos, tokens_completos[1:])]
                ),
                tokens_nucleo=tokens_nucleo,
            )
            self.tuss_by_id[tuss_id] = reg
            self.tuss_list.append(reg)
            self.tuss_ids.append(tuss_id)
            self.nucleos_norm.append(" ".join(tokens_nucleo))
            corpus_tokens.append(tokens_completos)

        if corpus_tokens:
            self.bm25_index = BM25Okapi(corpus_tokens)
            # IDF próprio: termos raros (ex.: "creatinina") pesam mais que termos comuns (ex.: "total")
            n_docs = len(corpus_tokens)
            df: Dict[str, int] = {}
            for toks in corpus_tokens:
                for t in set(toks):
                    df[t] = df.get(t, 0) + 1
            self.idf_tokens = {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}
            self._idf_maximo = math.log(n_docs + 1) + 1.0

    def _carregar_base_pacientes(self) -> None:
        linhas = self._abrir_csv_seguro(self.pacientes_csv_path)
        for row in linhas:
            nome = (row.get("nome_paciente") or "").strip()
            carteira = extrair_apenas_digitos(row.get("carteira_do_plano", ""))
            if not carteira:
                continue
            carteira_pad = carteira.zfill(17)
            pac = RegistroPaciente(nome_paciente=nome, carteira_do_plano=carteira_pad)
            self.pacientes_by_carteira[carteira_pad] = pac
            self.pacientes_list.append(pac)

    def _idf(self, token: str) -> float:
        return self.idf_tokens.get(token, self._idf_maximo)

    def _pontuar_descricao(
        self,
        tokens_consulta: Sequence[str],
        reg: RegistroTUSS,
        tokens_detalhe: Sequence[str] = (),
    ) -> float:
        """
        Similaridade 0-100 entre a descrição lida (já expandida por sinônimos) e um procedimento TUSS:
          - 50% cobertura da consulta: quanto do que a guia pede está no procedimento (ponderado por IDF;
                termos entre parênteses na guia valem metade);
          - 35% cobertura do núcleo: quanto do nome principal do procedimento foi pedido na guia
                (penaliza termos extras, ex.: "Clearance de uréia" para a consulta "ureia");
          - 15% termo principal: a primeira palavra da guia coincide com a do procedimento.
        """
        principais = [t for t in dict.fromkeys(tokens_consulta) if t not in TERMOS_GENERICOS_TUSS]
        detalhes = [
            t for t in dict.fromkeys(tokens_detalhe)
            if t not in TERMOS_GENERICOS_TUSS and t not in principais
        ]
        if not principais:
            principais, detalhes = detalhes, []
        if not principais:
            return 0.0

        pesos = [(t, self._idf(t)) for t in principais] + [(t, 0.5 * self._idf(t)) for t in detalhes]
        peso_total = sum(p for _, p in pesos)
        cobertura_q = sum(p * _peso_presenca(t, reg.tokens_completos) for t, p in pesos) / peso_total

        conjunto_consulta = set(principais) | set(detalhes)
        nucleo = list(dict.fromkeys(reg.tokens_nucleo))
        peso_n = sum(self._idf(t) for t in nucleo)
        cobertura_n = (
            sum(self._idf(t) * _peso_presenca(t, conjunto_consulta) for t in nucleo) / peso_n
            if peso_n > 0 else 0.0
        )

        termo_principal = _peso_presenca(principais[0], {nucleo[0]}) if nucleo else 0.0
        return 100.0 * (0.50 * cobertura_q + 0.35 * cobertura_n + 0.15 * termo_principal)

    def _candidatos_por_descricao(self, tokens_consulta: Sequence[str], limite: int = 50) -> set[int]:
        """Recuperação de candidatos: BM25 sobre o nome completo + fuzzy sobre o nome principal."""
        candidatos: set[int] = set()
        if not tokens_consulta:
            return candidatos
        if self.bm25_index is not None:
            scores = self.bm25_index.get_scores(list(tokens_consulta))
            candidatos.update(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limite])
        for _, _, idx in process.extract(
            " ".join(tokens_consulta), self.nucleos_norm, scorer=fuzz.token_set_ratio, limit=30
        ):
            candidatos.add(idx)
        return candidatos

    def _avaliar_similaridade_descricao(self, desc_ocr_norm: str, reg: RegistroTUSS) -> float:
        if not desc_ocr_norm or not reg.display_name_norm:
            return 0.0
        return float(fuzz.token_set_ratio(desc_ocr_norm, reg.display_name_norm))

    def validar_procedimento(self, proc: ProcedimentoOCRRaw) -> ProcedimentoValidadoOutput:
        codigo_bruto = (proc.codigo_tuss_ocr or "").strip()
        desc_bruta = (proc.descricao_procedimento_ocr or "").strip()
        desc_norm = normalizar_texto_comparacao(desc_bruta)

        digitos_diretos = remover_zeros_esquerda_excedentes(extrair_apenas_digitos(codigo_bruto))
        digitos_opticos = remover_zeros_esquerda_excedentes(normalizar_ruido_optico_numerico(codigo_bruto))

        # ------------------------------------------------------------------
        # PASSO 1: Match Exato Direto (EXATO)
        # ------------------------------------------------------------------
        if len(digitos_diretos) == 8 and digitos_diretos in self.tuss_by_id:
            reg = self.tuss_by_id[digitos_diretos]
            sim_desc = self._avaliar_similaridade_descricao(desc_norm, reg)
            score_final = 100.0 if (not desc_norm or sim_desc >= 35.0) else 91.0
            requer_hitl = score_final < self.hitl_threshold or not reg.vigencia_ativa
            motivo = None if not requer_hitl else "Código exato, mas descrição divergente ou vigência encerrada."

            return ProcedimentoValidadoOutput(
                sequencial_item=proc.sequencial_item,
                codigo_tuss_ocr_bruto=codigo_bruto,
                codigo_tuss_corrigido=reg.id,
                descricao_ocr_bruta=desc_bruta,
                descricao_tuss_oficial=reg.display_name,
                source_tuss=reg.source,
                vigencia_ativa=reg.vigencia_ativa,
                quantidade=proc.quantidade,
                valor_unitario=proc.valor_unitario_ocr,
                score_confianca_lexical=round(score_final, 2),
                score_fuzzy_codigo=100.0,
                score_bm25_descricao=round(sim_desc, 2),
                metodo_correcao=MetodoCorrecaoTUSS.EXATO,
                requer_revisao_humana=requer_hitl,
                motivo_revisao=motivo,
            )

        # ------------------------------------------------------------------
        # PASSO 2: Correção por Regra de Ruído Óptico (REGRA_VISUAL)
        # ------------------------------------------------------------------
        if len(digitos_opticos) == 8 and digitos_opticos in self.tuss_by_id:
            reg = self.tuss_by_id[digitos_opticos]
            sim_desc = self._avaliar_similaridade_descricao(desc_norm, reg)
            if not desc_norm:
                score_final = 92.0
            elif sim_desc >= 50.0:
                score_final = round(92.0 + (sim_desc - 50.0) * 0.14, 2)
            else:
                score_final = round(78.0 + sim_desc * 0.2, 2)

            requer_hitl = score_final < self.hitl_threshold or not reg.vigencia_ativa
            motivo = (
                f"Ruído óptico corrigido ({codigo_bruto} -> {reg.id}), mas baixa aderência na descrição ({sim_desc:.1f}%)."
                if requer_hitl
                else None
            )

            return ProcedimentoValidadoOutput(
                sequencial_item=proc.sequencial_item,
                codigo_tuss_ocr_bruto=codigo_bruto,
                codigo_tuss_corrigido=reg.id,
                descricao_ocr_bruta=desc_bruta,
                descricao_tuss_oficial=reg.display_name,
                source_tuss=reg.source,
                vigencia_ativa=reg.vigencia_ativa,
                quantidade=proc.quantidade,
                valor_unitario=proc.valor_unitario_ocr,
                score_confianca_lexical=round(min(score_final, 99.0), 2),
                score_fuzzy_codigo=95.0,
                score_bm25_descricao=round(sim_desc, 2),
                metodo_correcao=MetodoCorrecaoTUSS.REGRA_VISUAL,
                requer_revisao_humana=requer_hitl,
                motivo_revisao=motivo,
            )

        # ------------------------------------------------------------------
        # PASSO 3: Busca Híbrida Fuzzy (Código) + BM25 (Descrição) (FUZZY_BM25)
        # ------------------------------------------------------------------
        desc_principal, desc_detalhes = separar_descricao_principal(desc_bruta)
        tokens_desc = expandir_sinonimos_clinicos(tokenizar_descricao_tuss(desc_principal))
        tokens_detalhe = expandir_sinonimos_clinicos(tokenizar_descricao_tuss(desc_detalhes))
        candidatos_indices: set[int] = self._candidatos_por_descricao(tokens_desc + tokens_detalhe)

        if digitos_opticos and self.tuss_ids:
            top_fuzzy = process.extract(
                digitos_opticos,
                self.tuss_ids,
                scorer=fuzz.ratio,
                limit=35,
            )
            for _, _, idx in top_fuzzy:
                candidatos_indices.add(idx)

        melhor_reg: Optional[RegistroTUSS] = None
        melhor_score_final = 0.0
        melhor_score_codigo = 0.0
        melhor_score_desc = 0.0

        tem_codigo = len(digitos_opticos) >= 4
        tem_desc = len(tokens_desc) + len(tokens_detalhe) >= 1

        for idx in candidatos_indices:
            reg = self.tuss_list[idx]
            s_codigo = float(fuzz.ratio(digitos_opticos, reg.id)) if tem_codigo else 0.0

            s_desc = self._pontuar_descricao(tokens_desc, reg, tokens_detalhe) if tem_desc else 0.0

            if tem_codigo and tem_desc:
                s_total = 0.60 * s_codigo + 0.40 * s_desc
            elif tem_codigo:
                s_total = 0.88 * s_codigo
            else:
                s_total = 0.88 * s_desc

            if s_total > melhor_score_final:
                melhor_score_final = s_total
                melhor_reg = reg
                melhor_score_codigo = s_codigo
                melhor_score_desc = s_desc

        if melhor_reg is None or melhor_score_final < 55.0:
            return ProcedimentoValidadoOutput(
                sequencial_item=proc.sequencial_item,
                codigo_tuss_ocr_bruto=codigo_bruto,
                codigo_tuss_corrigido=None,
                descricao_ocr_bruta=desc_bruta,
                descricao_tuss_oficial=None,
                source_tuss=None,
                vigencia_ativa=False,
                quantidade=proc.quantidade,
                valor_unitario=proc.valor_unitario_ocr,
                score_confianca_lexical=round(melhor_score_final, 2),
                score_fuzzy_codigo=round(melhor_score_codigo, 2),
                score_bm25_descricao=round(melhor_score_desc, 2),
                metodo_correcao=MetodoCorrecaoTUSS.NAO_ENCONTRADO,
                requer_revisao_humana=True,
                motivo_revisao="Procedimento não localizado com confiança mínima na tabela TUSS.",
            )

        score_arredondado = round(min(melhor_score_final, 98.5), 2)
        requer_hitl = score_arredondado < self.hitl_threshold or not melhor_reg.vigencia_ativa
        motivo = (
            f"Match aproximado (Fuzzy+BM25 = {score_arredondado:.1f}% < {self.hitl_threshold:.0f}%). Validar antes do faturamento."
            if requer_hitl
            else None
        )

        return ProcedimentoValidadoOutput(
            sequencial_item=proc.sequencial_item,
            codigo_tuss_ocr_bruto=codigo_bruto,
            codigo_tuss_corrigido=melhor_reg.id,
            descricao_ocr_bruta=desc_bruta,
            descricao_tuss_oficial=melhor_reg.display_name,
            source_tuss=melhor_reg.source,
            vigencia_ativa=melhor_reg.vigencia_ativa,
            quantidade=proc.quantidade,
            valor_unitario=proc.valor_unitario_ocr,
            score_confianca_lexical=score_arredondado,
            score_fuzzy_codigo=round(melhor_score_codigo, 2),
            score_bm25_descricao=round(melhor_score_desc, 2),
            metodo_correcao=MetodoCorrecaoTUSS.FUZZY_BM25,
            requer_revisao_humana=requer_hitl,
            motivo_revisao=motivo,
        )

    def validar_paciente(
        self,
        nome_ocr: Optional[str],
        carteira_ocr: Optional[str],
    ) -> PacienteOutput:
        """
        Valida o beneficiário em pacientes_simulados.csv.
        Retorna o nome/carteirinha como lidos pelo OCR e, se localizado, os valores do cadastro.
        """
        carteira_norm = normalizar_ruido_optico_numerico(carteira_ocr)
        nome_norm = normalizar_texto_comparacao(nome_ocr or "")
        # Nome em Hangul: romaniza e compara pela chave fonética com a grafia latina do cadastro
        nome_coreano = contem_hangul(nome_ocr)
        if nome_coreano:
            nome_norm = chave_fonetica_coreana(romanizar_nome_coreano(nome_ocr or ""))
        lido = dict(
            nome_paciente_ocr_bruto=(nome_ocr or "").strip() or None,
            carteira_plano_ocr_bruta=(carteira_ocr or "").strip() or None,
        )

        if len(carteira_norm) == 17 and carteira_norm in self.pacientes_by_carteira:
            pac = self.pacientes_by_carteira[carteira_norm]
            return PacienteOutput(
                paciente_validado_cadastro=True,
                metodo_match_paciente="EXATO_CARTEIRA",
                score_match_paciente=100.0,
                nome_paciente_cadastro=pac.nome_paciente,
                carteira_plano_cadastro=pac.carteira_do_plano,
                **lido,
            )

        melhor_pac: Optional[RegistroPaciente] = None
        melhor_score = 0.0

        for pac in self.pacientes_list:
            s_cart = float(fuzz.ratio(carteira_norm, pac.carteira_do_plano)) if carteira_norm else 0.0
            if not nome_norm:
                s_nome = 0.0
            elif nome_coreano:
                s_nome = float(fuzz.token_sort_ratio(nome_norm, chave_fonetica_coreana(pac.nome_paciente)))
            else:
                s_nome = float(fuzz.token_set_ratio(nome_norm, normalizar_texto_comparacao(pac.nome_paciente)))
            if carteira_norm and nome_norm:
                s_comb = 0.65 * s_cart + 0.35 * s_nome
            elif carteira_norm:
                s_comb = s_cart
            else:
                s_comb = s_nome

            if s_comb > melhor_score:
                melhor_score = s_comb
                melhor_pac = pac

        if melhor_pac is not None and melhor_score >= 85.0:
            return PacienteOutput(
                paciente_validado_cadastro=True,
                metodo_match_paciente="FUZZY_CARTEIRA_NOME",
                score_match_paciente=round(melhor_score, 2),
                nome_paciente_cadastro=melhor_pac.nome_paciente,
                carteira_plano_cadastro=melhor_pac.carteira_do_plano,
                **lido,
            )

        return PacienteOutput(
            paciente_validado_cadastro=False,
            metodo_match_paciente="NAO_LOCALIZADO",
            score_match_paciente=round(melhor_score, 2),
            **lido,
        )

    def consultar_top_candidatos(self, termo_ou_codigo: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Utilitário de apoio para as Tools do ADK e conferência manual do gabarito."""
        termo = termo_ou_codigo.strip()
        digitos = remover_zeros_esquerda_excedentes(normalizar_ruido_optico_numerico(termo))
        resultados: List[Tuple[str, str, float]] = []

        # Só trata como código se o termo for majoritariamente numérico (ex.: '10101O12'):
        # textos como 'Ultrassom' também geram "dígitos" pela regra de ruído óptico.
        alfanumericos = re.sub(r"[^0-9A-Za-z]", "", termo)
        parece_codigo = (
            len(digitos) >= 6
            and len(extrair_apenas_digitos(termo)) >= 0.5 * max(len(alfanumericos), 1)
        )
        if parece_codigo:
            matches = process.extract(digitos, self.tuss_ids, scorer=fuzz.ratio, limit=top_k)
            for cod, score, _ in matches:
                reg = self.tuss_by_id[cod]
                resultados.append((reg.id, reg.display_name, round(float(score), 2)))
            return resultados

        principal, detalhes = separar_descricao_principal(termo)
        tokens = expandir_sinonimos_clinicos(tokenizar_descricao_tuss(principal))
        tokens_det = expandir_sinonimos_clinicos(tokenizar_descricao_tuss(detalhes))
        pontuados = [
            (self._pontuar_descricao(tokens, self.tuss_list[idx], tokens_det), idx)
            for idx in self._candidatos_por_descricao(tokens + tokens_det)
        ]
        for score, idx in sorted(pontuados, reverse=True)[:top_k]:
            reg = self.tuss_list[idx]
            resultados.append((reg.id, reg.display_name, round(score, 2)))
        return resultados


# ==============================================================================
# CLI RÁPIDO PARA CONFERÊNCIA DO GABARITO MANUAL CONTRA OS CSVs OFICIAIS
# Rode da raiz clinica-vida-mais/:
#   python -m ia_ocr_rag.core.lexical_rag --buscar "consulta"
#   python -m ia_ocr_rag.core.lexical_rag --tuss "10101O12" --desc "Consulta em consultorio"
# ==============================================================================
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Validador rápido TUSS / Pacientes (Clínica Vida+)")
    parser.add_argument("--tuss", type=str, default="", help="Código TUSS lido na guia")
    parser.add_argument("--desc", type=str, default="", help="Descrição lida na guia")
    parser.add_argument("--buscar", type=str, default="", help="Busca livre por termo ou código na TUSS")
    parser.add_argument("--carteira", type=str, default="", help="Carteirinha do paciente (17 dígitos)")
    parser.add_argument("--paciente", type=str, default="", help="Nome do paciente")
    args = parser.parse_args()

    engine = LexicalTUSSEngine(
        tuss_csv_path=os.getenv(
            "TUSS_CSV_PATH",
            "ia_ocr_rag/data/reference/tuss-22_20260902.xlsx - tuss-22_20260902.csv",
        ),
        pacientes_csv_path=os.getenv(
            "PACIENTES_CSV_PATH",
            "ia_ocr_rag/data/reference/pacientes_simulados.csv",
        ),
    )
    print(f"[OK] Base TUSS: {len(engine.tuss_list)} procedimentos | Pacientes: {len(engine.pacientes_list)}")

    if args.buscar:
        print(f"\nTop 5 candidatos para '{args.buscar}':")
        for cod, nome, sc in engine.consultar_top_candidatos(args.buscar, top_k=5):
            print(f"  - {cod} | {nome} (score: {sc}%)")

    if args.tuss or args.desc:
        res = engine.validar_procedimento(
            ProcedimentoOCRRaw(
                sequencial_item=1,
                codigo_tuss_ocr=args.tuss,
                descricao_procedimento_ocr=args.desc,
            )
        )
        print("\nResultado da Validação TUSS:")
        print(res.model_dump_json(indent=2))

    if args.carteira or args.paciente:
        pac_res = engine.validar_paciente(args.paciente, args.carteira)
        print("\nResultado da Validação do Paciente:")
        print(pac_res.model_dump_json(indent=2))
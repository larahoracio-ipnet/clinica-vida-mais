from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
import numpy as np
from PIL import Image

from ia_ocr_rag.core.resiliencia_gemini import cadeia_de_modelos, executar_com_fallback, timeout_ms
from ia_ocr_rag.core.schemas import GuiaOCRRaw, ProcedimentoOCRRaw
from ia_ocr_rag.core.verificacao_marcacao import verificar_formulario_marcacao

logger = logging.getLogger(__name__)

PROMPT_EXTRACAO_TISS = """
Você é um especialista em auditoria de contas médicas e faturamento padrão TISS/TUSS no Brasil.
Analise a imagem desta guia médica e extraia os dados estruturados rigorosamente conforme o schema JSON solicitado.

REGRAS CRÍTICAS DE EXTRAÇÃO:
1. FIDELIDADE ÓPTICA (NÃO ADIVINHE CÓDIGOS): Extraia o `codigo_tuss_ocr` e a `descricao_procedimento_ocr` exatamente como aparecem impressos/manuscritos na guia. Se houver caracteres ambíguos ou borrados, transcreva-os fielmente sem tentar corrigir por conta própria (a camada de RAG Lexical fará a padronização oficial).
2. GRADE DE PROCEDIMENTOS: Extraia todos os procedimentos/exames solicitados ou executados (em tabela ou em lista), numerando `sequencial_item` a partir de 1. Em formulários de marcação, siga a regra 7.
3. IDENTIFICAÇÃO DO BENEFICIÁRIO: Extraia o nome completo do paciente (`nome_paciente_ocr`) e o número da carteirinha (`carteira_plano_ocr`, geralmente 17 dígitos).
4. PROFISSIONAL E GUIA: Extraia número da guia (prestador/operadora), registro ANS (6 dígitos), data do atendimento, nome do profissional, sigla do conselho (ex: CRM), número do conselho, UF, CID-10 e caráter do atendimento, quando presentes.
5. CAMPOS AUSENTES: Se algum campo não estiver visível ou preenchido na guia, retorne null (não invente informações).
6. PROCEDIMENTO SEM CÓDIGO: Se um procedimento estiver listado apenas pela descrição, sem código TUSS na guia, retorne `codigo_tuss_ocr` como string vazia "" (nunca "N/A", "-" ou texto equivalente) e extraia normalmente a descrição.
7. FORMULÁRIO DE MARCAÇÃO (CHECKLIST): Se a guia for um formulário com uma lista pré-impressa de exames e um marcador ao lado de cada opção (bolinha, caixa, quadrado ou parênteses), um exame só foi solicitado se o seu marcador estiver PREENCHIDO, pintado, riscado ou com "X"/"✓".
   - Extraia SOMENTE os itens marcados. Ignore os itens cujo marcador está vazio, mesmo que tenham código impresso.
   - Examine o marcador de CADA opção, uma a uma, em TODAS as colunas e seções do formulário; não pare na primeira coluna.
   - O marcador de um exame fica imediatamente à ESQUERDA do nome, na MESMA linha. Não atribua a um exame a marcação da linha de cima ou de baixo.
   - Ordem: seção por seção, de cima para baixo; dentro de cada seção, a coluna da esquerda inteira e depois a da direita.
   - Esta regra NÃO se aplica a tabelas ou listas em que todos os itens são pedidos (ex.: tabela numerada de exames, lista "Solicito:"); nesses casos, um visto ou traço ao lado do código não muda nada: extraia todos.
""".strip()


def eh_pdf(caminho: Path) -> bool:
    return caminho.suffix.lower() == ".pdf"


def renderizar_primeira_pagina_pdf(caminho: Path) -> Image.Image:
    """Converte a 1ª página do PDF em imagem (usado só pelo fallback offline, que não lê PDF)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(caminho))
    try:
        return pdf[0].render(scale=2).to_pil().convert("RGB")
    finally:
        pdf.close()


class GuiaOCRExtractor:
    """
    Extrator estruturado de guias médicas TISS/TUSS:
      - Primário: Gemini multimodal com Structured Outputs (Pydantic), com novas tentativas
        e troca automática de modelo (GEMINI_MODEL_NAME -> GEMINI_FALLBACK_MODELS).
      - Secundário (Contingência / Risco Zero): RapidOCR local via ONNX Runtime.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.modelos = cadeia_de_modelos(model_name)
        self.model_name = self.modelos[0]
        self.max_retries = max_retries  # tentativas por modelo (None = GEMINI_TENTATIVAS_POR_MODELO)
        self._gemini_client: Optional[genai.Client] = None
        self._rapid_ocr = None

        if self.api_key and not self.api_key.startswith("cole_sua_chave"):
            # Timeout evita que a guia fique travada quando a API está sobrecarregada;
            # o estouro conta como erro transitório (nova tentativa / próximo modelo).
            self._gemini_client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=timeout_ms()),
            )

    def _get_rapid_ocr(self):
        if self._rapid_ocr is None:
            from rapidocr import RapidOCR
            self._rapid_ocr = RapidOCR()
        return self._rapid_ocr

    def _extrair_com_gemini(self, caminho_imagem: Path) -> Tuple[GuiaOCRRaw, str]:
        """Retorna (extração, modelo que respondeu), percorrendo a cadeia de modelos."""
        if self._gemini_client is None:
            raise RuntimeError("GEMINI_API_KEY não configurada ou inválida.")

        if eh_pdf(caminho_imagem):
            # O Gemini lê PDF nativamente (todas as páginas), sem conversão para imagem
            conteudo_guia = types.Part.from_bytes(data=caminho_imagem.read_bytes(), mime_type="application/pdf")
        else:
            with Image.open(caminho_imagem) as img:
                conteudo_guia = img.convert("RGB")

        def chamar(modelo: str) -> GuiaOCRRaw:
            response = self._gemini_client.models.generate_content(
                model=modelo,
                contents=[PROMPT_EXTRACAO_TISS, conteudo_guia],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GuiaOCRRaw,
                    temperature=0.0,
                ),
            )
            if not response.text:
                raise ValueError("Resposta vazia retornada pelo Gemini.")
            return GuiaOCRRaw.model_validate_json(response.text)

        return executar_com_fallback(chamar, self.modelos, self.max_retries)

    def _extrair_com_rapidocr_offline(self, caminho_imagem: Path) -> GuiaOCRRaw:
        """
        Parser determinístico de contingência sobre os blocos de texto extraídos
        pelo RapidOCR local (garante funcionamento 100% offline na demo).
        """
        ocr_engine = self._get_rapid_ocr()
        entrada = np.array(renderizar_primeira_pagina_pdf(caminho_imagem)) if eh_pdf(caminho_imagem) else str(caminho_imagem)
        resultado_ocr = ocr_engine(entrada)

        # rapidocr 3.x retorna RapidOCROutput; os textos reconhecidos ficam em .txts
        linhas_texto: List[str] = [
            str(texto).strip() for texto in (resultado_ocr.txts or ()) if str(texto).strip()
        ]

        texto_completo = "\n".join(linhas_texto)

        # Carteirinha de 17 dígitos (aceita eventuais ruídos de letra no meio)
        match_carteira = re.search(r"\b[0-9OoIlZzSsB]{15,19}\b", texto_completo)
        carteira_ocr = match_carteira.group(0) if match_carteira else None

        # Data DD/MM/YYYY ou YYYY-MM-DD
        match_data = re.search(r"\b(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})\b", texto_completo)
        data_atendimento = match_data.group(1) if match_data else None

        # Registro ANS (6 dígitos próximos a ANS ou isolados)
        match_ans = re.search(r"(?:ANS[:\s-]*)(\d{6})\b", texto_completo, re.IGNORECASE)
        registro_ans = match_ans.group(1) if match_ans else None

        # Conselho profissional (ex: CRM RJ 123456)
        match_conselho = re.search(
            r"\b(CRM|CRO|CREFITO|CRP|CRN|COREN)[\s/-]*([A-Z]{2})?[\s:-]*(\d{4,8})\b",
            texto_completo,
            re.IGNORECASE,
        )
        conselho = match_conselho.group(1).upper() if match_conselho else None
        uf_conselho = match_conselho.group(2).upper() if (match_conselho and match_conselho.group(2)) else None
        num_conselho = match_conselho.group(3) if match_conselho else None

        # Nome do paciente (heurística após rótulos comuns de guia TISS)
        nome_paciente: Optional[str] = None
        for idx, linha in enumerate(linhas_texto):
            if re.search(r"(nome\s+do\s+benefici[aá]rio|paciente)", linha, re.IGNORECASE):
                limpo = re.sub(r".*?(benefici[aá]rio|paciente)[:\s-]*", "", linha, flags=re.IGNORECASE).strip()
                if len(limpo) >= 5 and not any(ch.isdigit() for ch in limpo):
                    nome_paciente = limpo
                    break
                if idx + 1 < len(linhas_texto):
                    prox = linhas_texto[idx + 1].strip()
                    if len(prox) >= 5 and not any(ch.isdigit() for ch in prox):
                        nome_paciente = prox
                        break

        # Procedimentos TUSS (padrão 8 caracteres alfanuméricos/numéricos iniciando com dígitos médicos)
        procedimentos: List[ProcedimentoOCRRaw] = []
        codigos_vistos: set[str] = set()
        padrao_tuss = re.compile(r"\b([1-9][0-9OoIlZzSsB]{6,7})\b")

        for idx, linha in enumerate(linhas_texto):
            for match in padrao_tuss.finditer(linha):
                cod_candidato = match.group(1)
                if cod_candidato in codigos_vistos:
                    continue
                codigos_vistos.add(cod_candidato)

                desc_candidata = linha[match.end():].strip(" -:|")
                if len(desc_candidata) < 4 and idx + 1 < len(linhas_texto):
                    desc_candidata = linhas_texto[idx + 1].strip()

                procedimentos.append(
                    ProcedimentoOCRRaw(
                        sequencial_item=len(procedimentos) + 1,
                        codigo_tuss_ocr=cod_candidato,
                        descricao_procedimento_ocr=desc_candidata or "PROCEDIMENTO EXTRAIDO VIA OCR OFFLINE",
                        quantidade=1,
                    )
                )

        return GuiaOCRRaw(
            registro_ans=registro_ans,
            data_atendimento=data_atendimento,
            nome_paciente_ocr=nome_paciente,
            carteira_plano_ocr=carteira_ocr,
            conselho_profissional_ocr=conselho,
            numero_conselho_ocr=num_conselho,
            uf_conselho_ocr=uf_conselho,
            procedimentos=procedimentos,
        )

    def extrair_guia(
        self,
        caminho_imagem: str | Path,
        force_offline: bool = False,
    ) -> Tuple[GuiaOCRRaw, str]:
        """
        Retorna (GuiaOCRRaw, nome_do_motor_utilizado).
        Etapas: Gemini multimodal (com modelos reserva) -> fallback RapidOCR offline ->
        verificação óptica de formulário de marcação (bolinhas pintadas), se for checklist.
        """
        path = Path(caminho_imagem)
        if not path.exists():
            raise FileNotFoundError(f"Imagem da guia não encontrada: {path}")

        guia_raw: Optional[GuiaOCRRaw] = None
        motor = "rapidocr_offline_fallback"
        if not force_offline and self._gemini_client is not None:
            try:
                guia_raw, motor = self._extrair_com_gemini(path)
            except Exception as exc:
                logger.warning("Acionando fallback offline RapidOCR para '%s': %s", path.name, exc)
        if guia_raw is None:
            guia_raw = self._extrair_com_rapidocr_offline(path)

        # Formulário de marcação: as bolinhas pintadas definem quais exames foram pedidos
        imagem = renderizar_primeira_pagina_pdf(path) if eh_pdf(path) else path
        guia_raw = verificar_formulario_marcacao(imagem, guia_raw, self._get_rapid_ocr)
        return guia_raw, motor
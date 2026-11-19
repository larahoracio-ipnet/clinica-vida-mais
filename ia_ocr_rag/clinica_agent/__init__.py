import sys
from pathlib import Path

# O `adk web ia_ocr_rag` coloca apenas ia_ocr_rag/ no sys.path; os imports
# absolutos `ia_ocr_rag.core...` exigem a raiz do repositório clinica-vida-mais/.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import agent  # noqa: E402

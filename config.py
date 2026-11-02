"""Configuración de la mini app.

Todo vive en esta carpeta: datos, base vectorial y código. No hay claves ni
servicios externos.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# Ingesta
# ---------------------------------------------------------------------------
COLLECTION_NAME = "mini_helpdesk"
ODT_EXTENSION = ".odt"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# ---------------------------------------------------------------------------
# Recuperación
# ---------------------------------------------------------------------------
TOP_K = 3

# Búsqueda híbrida: vectorial + coincidencia exacta de términos. Sirve para
# siglas y nombres propios, que los embeddings pequeños tienden a diluir.
HYBRID_SEARCH = True
HYBRID_FETCH_K = 10
HYBRID_MAX_TERMINOS = 4
HYBRID_BONUS_TERMINO = 0.10
HYBRID_BONUS_MAX = 0.20

# ---------------------------------------------------------------------------
# Respuesta extractiva
# ---------------------------------------------------------------------------
# La confianza combina similitud semántica y cuántos términos de la consulta
# aparecen en el fragmento. Por debajo del umbral se responde "no encontrado".
CONFIDENCE_THRESHOLD = 0.45

# Recorte del fragmento que se muestra como respuesta.
MAX_RESPUESTA_CHARS = 900

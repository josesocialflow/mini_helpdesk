"""Acceso a Chroma y búsqueda híbrida.

- Vectorial: similitud semántica con `all-MiniLM-L6-v2` (ONNX, local y gratis).
- Léxica: coincidencia exacta de términos con regex tolerante a acentos.
- Fusión: la similitud manda y la coincidencia añade un pequeño bonus.

La parte léxica existe por un motivo concreto: los embeddings pequeños pierden
siglas y nombres propios. Con `HYBRID_SEARCH = False` se puede comparar.
"""

import re
from functools import lru_cache
from typing import Any, cast

from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction, QueryResult
from chromadb.errors import InvalidArgumentError
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    HYBRID_BONUS_MAX,
    HYBRID_BONUS_TERMINO,
    HYBRID_FETCH_K,
    HYBRID_MAX_TERMINOS,
    HYBRID_SEARCH,
    TOP_K,
)
from models import Candidato

# Palabras funcionales que no discriminan en la búsqueda léxica.
PALABRAS_VACIAS = frozenset(
    {
        "a", "al", "algo", "algun", "alguna", "alguno", "como", "cómo",
        "con", "contra", "cual", "cuál", "cuando", "cuándo", "de", "del",
        "desde", "donde", "dónde", "durante", "el", "ella", "ellas", "ellos",
        "en", "entre", "era", "es", "esa", "ese", "eso", "esta", "este",
        "esto", "estos", "fue", "ha", "hay", "la", "las", "le", "les", "lo",
        "los", "mas", "más", "me", "mi", "mis", "mucho", "muy", "no", "nos",
        "o", "os", "otra", "otro", "para", "pero", "poco", "por", "porque",
        "que", "qué", "se", "ser", "si", "sí", "sin", "sobre", "su", "sus",
        "también", "te", "tiene", "tienen", "tu", "tus", "un", "una", "uno",
        "unos", "y", "ya", "yo",
    }
)

# Cada vocal admite su forma acentuada: "formulario" no necesita, pero
# "validacion" encuentra "validación".
_EQUIVALENCIAS = {
    "a": "[aá]",
    "e": "[eé]",
    "i": "[ií]",
    "o": "[oó]",
    "u": "[uúü]",
}


@lru_cache(maxsize=1)
def obtener_coleccion() -> Collection:
    """Abre (o crea) la colección persistente con el modelo local de Chroma."""
    cliente = PersistentClient(path=str(CHROMA_DIR))

    return cliente.get_or_create_collection(
        name=COLLECTION_NAME,
        # Chroma tipa el parámetro como EmbeddingFunction[Embeddable], pero
        # esta función trabaja con textos; el cast evita un falso positivo.
        embedding_function=cast(
            "EmbeddingFunction[Any]",
            ONNXMiniLM_L6_V2(
                preferred_providers=["CPUExecutionProvider"]
            ),
        ),
        metadata={"hnsw:space": "cosine"},
    )


def contar_fragmentos() -> int:
    """Número de fragmentos indexados."""
    return obtener_coleccion().count()


def borrar_documento(documento: str) -> None:
    """Elimina los fragmentos de un documento antes de reindexarlo."""
    obtener_coleccion().delete(where={"documento": documento})


def borrar_coleccion() -> None:
    """Borra la colección completa (para `ingest.py --reset`)."""
    cliente = PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME in {coleccion.name for coleccion in cliente.list_collections()}:
        cliente.delete_collection(COLLECTION_NAME)
    obtener_coleccion.cache_clear()


def terminos_clave(consulta: str) -> list[str]:
    """Términos que merecen búsqueda léxica: sin vacías, mínimo 3 letras."""
    terminos: list[str] = []
    vistos: set[str] = set()

    for bruto in re.findall(r"[^\W\d_]+", consulta):
        palabra = bruto.lower()

        if palabra in PALABRAS_VACIAS or palabra in vistos:
            continue
        if len(palabra) < 3:
            continue

        vistos.add(palabra)
        terminos.append(palabra)

        if len(terminos) == HYBRID_MAX_TERMINOS:
            break

    return terminos


def _patron(termino: str) -> str | None:
    """Regex con límites de palabra y vocales acentuadas opcionales."""
    if not termino.isalnum():
        return None

    partes = [_EQUIVALENCIAS.get(letra, re.escape(letra)) for letra in termino]
    return r"(?i)\b" + "".join(partes) + r"\b"


def _fila(resultados: QueryResult, indice: int = 0) -> list[dict[str, Any]]:
    """Convierte un resultado de Chroma en diccionarios."""
    documentos = resultados["documents"]
    metadatos = resultados["metadatas"]
    distancias = resultados["distances"]

    if documentos is None or metadatos is None or distancias is None:
        raise ValueError("La consulta no devolvió documentos, metadatos ni distancias.")

    return [
        {
            "id": id_,
            "documento": str((metadata or {}).get("documento", "desconocido")),
            "fragmento": (documento or "").strip() or "(fragmento vacío)",
            "distancia": float(distancia) if distancia is not None else None,
        }
        for id_, documento, metadata, distancia in zip(
            resultados["ids"][indice],
            documentos[indice],
            metadatos[indice],
            distancias[indice],
        )
    ]


def _listas_lexicas(
    coleccion: Collection,
    consulta: str,
    k: int,
) -> list[list[dict[str, Any]]]:
    """
    Ranking vectorial restringido a los fragmentos que contienen cada término.

    Al filtrar con regex, los fragmentos con la palabra se ordenan por
    similitud entre ellos, no en un orden arbitrario.
    """
    listas: list[list[dict[str, Any]]] = []

    for termino in terminos_clave(consulta):
        patron = _patron(termino)
        if patron is None:
            continue

        try:
            resultados = coleccion.query(
                query_texts=[consulta],
                n_results=k,
                where_document={"$regex": patron},
                include=["documents", "metadatas", "distances"],
            )
        except (InvalidArgumentError, TypeError):
            # Versión de Chroma sin soporte de regex: solo búsqueda vectorial.
            return []

        listas.append(_fila(resultados))

    return listas


def _fusionar(
    vectoriales: list[dict[str, Any]],
    lexicas: list[list[dict[str, Any]]],
    k: int,
) -> list[dict[str, Any]]:
    """
    Fusiona por puntuación: similitud + bonus por término exacto.

    `puntos = (1 - distancia) + 0.10 * min(términos_coincidentes, 2)`
    """
    acumulado: dict[str, dict[str, Any]] = {}

    def registrar(item: dict[str, Any], termino: int | None = None) -> None:
        entrada = acumulado.setdefault(
            item["id"], {"item": dict(item), "terminos": set()}
        )

        distancia = item["distancia"]
        actual = entrada["item"]["distancia"]
        if distancia is not None and (actual is None or distancia < actual):
            entrada["item"]["distancia"] = distancia

        if termino is not None:
            entrada["terminos"].add(termino)

    for item in vectoriales:
        registrar(item)

    for indice, lista in enumerate(lexicas):
        for item in lista:
            registrar(item, termino=indice)

    def puntos(entrada: dict[str, Any]) -> float:
        distancia = entrada["item"]["distancia"]
        similitud = 1.0 - distancia if distancia is not None else 0.0
        bonus = min(
            len(entrada["terminos"]) * HYBRID_BONUS_TERMINO,
            HYBRID_BONUS_MAX,
        )
        return similitud + bonus

    ordenados = sorted(
        acumulado.values(),
        key=lambda entrada: (
            -puntos(entrada),
            entrada["item"]["distancia"]
            if entrada["item"]["distancia"] is not None
            else float("inf"),
            entrada["item"]["id"],
        ),
    )

    return [entrada["item"] for entrada in ordenados[:k]]


def buscar_candidatos(consulta: str, k: int = TOP_K) -> list[Candidato]:
    """Devuelve los `k` fragmentos más relevantes para la consulta."""
    consulta = consulta.strip()
    if not consulta:
        raise ValueError("La consulta no puede estar vacía.")

    coleccion = obtener_coleccion()
    total = coleccion.count()
    if total == 0:
        return []

    resultados = coleccion.query(
        query_texts=[consulta],
        n_results=min(max(k * 3, HYBRID_FETCH_K), total),
        include=["documents", "metadatas", "distances"],
    )

    vectoriales = _fila(resultados)
    lexicas = _listas_lexicas(coleccion, consulta, k) if HYBRID_SEARCH else []

    return [Candidato(**item) for item in _fusionar(vectoriales, lexicas, k)]


if __name__ == "__main__":
    print(f"Fragmentos en la colección: {contar_fragmentos()}")

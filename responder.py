"""Respuesta extractiva: sin LLM, sin claves y sin llamadas de red.

La respuesta es el fragmento más relevante, tal cual está en la
documentación, con su documento de origen. La confianza es una heurística
explicable:

    confianza = 0.7 * similitud + 0.3 * cobertura de términos

- `similitud` = 1 - distancia coseno.
- `cobertura` = proporción de términos de la consulta presentes en el
  fragmento (sin distinguir mayúsculas ni acentos).

Si no llega al umbral, se responde "no encontrado" en lugar de inventar.
"""

import unicodedata

from config import CONFIDENCE_THRESHOLD, MAX_RESPUESTA_CHARS
from models import Respuesta
from vectorstore import buscar_candidatos, terminos_clave


def _sin_acentos(texto: str) -> str:
    """Minúsculas y sin diacríticos, para comparar términos."""
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(
        letra for letra in descompuesto if not unicodedata.combining(letra)
    )


def _cobertura(consulta: str, fragmento: str) -> float:
    """Proporción de términos de la consulta que aparecen en el fragmento."""
    terminos = terminos_clave(consulta)
    if not terminos:
        return 0.0

    texto = _sin_acentos(fragmento)
    encontrados = sum(1 for termino in terminos if _sin_acentos(termino) in texto)

    return encontrados / len(terminos)


def _recortar(texto: str) -> str:
    """Recorta el fragmento sin cortar una palabra por la mitad."""
    if len(texto) <= MAX_RESPUESTA_CHARS:
        return texto

    recorte = texto[:MAX_RESPUESTA_CHARS].rsplit(" ", 1)[0]
    return f"{recorte}…"


def responder(consulta: str) -> Respuesta:
    """Busca y devuelve el fragmento que mejor responde a la consulta."""
    consulta = consulta.strip()
    if not consulta:
        raise ValueError("La consulta no puede estar vacía.")

    candidatos = buscar_candidatos(consulta)

    if not candidatos:
        return Respuesta(
            encontrado=False,
            texto=(
                "No hay documentación indexada. Ejecuta `python ingest.py` "
                "antes de hacer consultas."
            ),
        )

    mejor = candidatos[0]
    similitud = (
        1.0 - mejor.distancia if mejor.distancia is not None else 0.0
    )
    similitud = max(0.0, min(similitud, 1.0))

    cobertura = _cobertura(consulta, mejor.fragmento)
    confianza = round(0.7 * similitud + 0.3 * cobertura, 3)

    justificacion = (
        f"similitud {similitud:.2f} · "
        f"cobertura de términos {cobertura:.0%}"
    )

    if confianza < CONFIDENCE_THRESHOLD:
        return Respuesta(
            encontrado=False,
            texto=(
                "No he encontrado información suficiente en la documentación "
                "para responder con fiabilidad. Prueba con otras palabras o "
                "revisa los fragmentos recuperados."
            ),
            confianza=confianza,
            justificacion=justificacion,
            fuentes=candidatos,
        )

    return Respuesta(
        encontrado=True,
        texto=_recortar(mejor.fragmento),
        confianza=confianza,
        justificacion=justificacion,
        documento=mejor.documento,
        candidato_id=mejor.id,
        fuentes=candidatos,
    )

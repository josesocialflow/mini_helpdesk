"""Estructuras de datos de la mini app.

Se usan dataclasses para no añadir dependencias: son objetos internos y la
validación fuerte no aporta nada aquí.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidato:
    """Fragmento recuperado de Chroma."""

    id: str
    documento: str
    fragmento: str
    distancia: float | None = None


@dataclass(frozen=True)
class Respuesta:
    """Resultado extractivo que se muestra al usuario."""

    encontrado: bool
    texto: str
    confianza: float = 0.0
    justificacion: str = ""
    documento: str | None = None
    candidato_id: str | None = None
    fuentes: list[Candidato] = field(default_factory=list)

"""Ingesta: ODT → texto → fragmentos → Chroma.

La extracción es tolerante a propósito. Algunos ODT llevan etiquetas HTML
sin escapar dentro de los párrafos (`<form>`, `<input>`, `<video>`…), lo que
rompe un parser XML estricto. Antes de parsear se escapan esas etiquetas,
que además interesa conservar como texto: son parte del contenido didáctico.

Uso:

    python ingest.py            # todos los ODT de datos/
    python ingest.py --reset    # borra la colección y vuelve a indexar
"""

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from chromadb.api.types import Metadata
from config import CHUNK_OVERLAP, CHUNK_SIZE, DATOS_DIR, ODT_EXTENSION
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore import (
    borrar_coleccion,
    borrar_documento,
    contar_fragmentos,
    obtener_coleccion,
)

OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
TEXTO_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
TABLA_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"

TAMANO_LOTE = 100

# Etiquetas sin prefijo de espacio de nombres: en un content.xml válido todas
# llevan prefijo (`text:p`, `office:text`…), así que una sin prefijo es HTML
# embebido por error y hay que escaparla antes de parsear.
ETIQUETA_HTML = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:\s[^<>]*?)?)(/?)>")


def _escapar_etiquetas_html(xml: str) -> str:
    """Convierte `<form>` en `&lt;form&gt;` para que el XML sea válido."""

    def reemplazo(coincidencia: re.Match[str]) -> str:
        if ":" in coincidencia.group(2):
            return coincidencia.group(0)
        apertura, nombre, atributos, cierre = coincidencia.groups()
        return f"&lt;{apertura}{nombre}{atributos}{cierre}&gt;"

    return ETIQUETA_HTML.sub(reemplazo, xml)


def _clave(etiqueta: str) -> tuple[str, str]:
    """Separa `{namespace}nombre` en (namespace, nombre)."""
    if etiqueta.startswith("{"):
        espacio, nombre = etiqueta[1:].split("}", 1)
        return espacio, nombre
    return "", etiqueta


def _texto(elemento: ElementTree.Element) -> str:
    """Texto plano del elemento, con espacios normalizados."""
    return " ".join("".join(elemento.itertext()).split())


def _nivel(encabezado: ElementTree.Element) -> int:
    """Nivel del encabezado (1-6), tomado de `text:outline-level`."""
    valor = encabezado.get(f"{{{TEXTO_NS}}}outline-level") or "1"

    try:
        return max(1, min(int(valor), 6))
    except ValueError:
        return 1


def _bloques(nodo: ElementTree.Element) -> list[str]:
    """
    Recorre el documento en orden y devuelve encabezados, párrafos y tablas.

    Las tablas se convierten en filas con `|` para no perder la estructura.
    """
    bloques: list[str] = []

    for hijo in list(nodo):
        espacio, nombre = _clave(hijo.tag)

        if espacio == TEXTO_NS and nombre == "h":
            texto = _texto(hijo)
            if texto:
                bloques.append(f"{'#' * _nivel(hijo)} {texto}")
        elif espacio == TEXTO_NS and nombre == "p":
            texto = _texto(hijo)
            if texto:
                bloques.append(texto)
        elif espacio == TABLA_NS and nombre == "table":
            filas: list[str] = []
            for fila in hijo.iter(f"{{{TABLA_NS}}}table-row"):
                celdas = [
                    _texto(celda)
                    for celda in fila.iter(f"{{{TABLA_NS}}}table-cell")
                ]
                if any(celdas):
                    filas.append(" | ".join(celdas))
            if filas:
                bloques.append("\n".join(filas))
        else:
            bloques.extend(_bloques(hijo))

    return bloques


def extraer_texto_odt(ruta: str | Path) -> str:
    """Extrae el texto de un ODT aunque tenga HTML sin escapar."""
    archivo = Path(ruta).expanduser().resolve()

    if not archivo.is_file():
        raise FileNotFoundError(f"No existe el archivo: {archivo}")

    if archivo.suffix.lower() != ODT_EXTENSION:
        raise ValueError(f"El archivo no es un ODT: {archivo.name}")

    try:
        with zipfile.ZipFile(archivo) as paquete:
            xml = paquete.read("content.xml").decode("utf-8")
    except KeyError as exc:
        raise ValueError(f"El ODT no contiene content.xml: {archivo.name}") from exc

    raiz = ElementTree.fromstring(_escapar_etiquetas_html(xml))
    cuerpo = raiz.find(f".//{{{OFFICE_NS}}}text")

    if cuerpo is None:
        return ""

    return "\n\n".join(_bloques(cuerpo)).strip()


def listar_odt(carpeta: str | Path = DATOS_DIR) -> list[Path]:
    """ODT de una carpeta, ordenados por nombre."""
    directorio = Path(carpeta).expanduser().resolve()
    return sorted(
        archivo
        for archivo in directorio.iterdir()
        if archivo.is_file() and archivo.suffix.lower() == ODT_EXTENSION
    )


def fragmentar_texto(texto: str) -> list[str]:
    """Divide el texto respetando párrafos, líneas y frases."""
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return divisor.split_text(texto)


def ingestar_documento(ruta: str | Path) -> int:
    """Extrae, fragmenta e indexa un ODT. Devuelve el número de fragmentos."""
    archivo = Path(ruta).expanduser().resolve()
    texto = extraer_texto_odt(archivo)

    if not texto:
        return 0

    fragmentos = fragmentar_texto(texto)
    if not fragmentos:
        return 0

    coleccion = obtener_coleccion()
    borrar_documento(archivo.name)

    ids = [f"{archivo.stem}-{indice:04d}" for indice in range(len(fragmentos))]
    metadatos: list[Metadata] = [
        {"documento": archivo.name, "chunk": indice}
        for indice in range(len(fragmentos))
    ]

    for inicio in range(0, len(fragmentos), TAMANO_LOTE):
        fin = inicio + TAMANO_LOTE
        coleccion.upsert(
            ids=ids[inicio:fin],
            documents=fragmentos[inicio:fin],
            metadatas=metadatos[inicio:fin],
        )

    return len(fragmentos)


def ingestar_todos(carpeta: str | Path = DATOS_DIR) -> dict[str, int]:
    """Ingesta todos los ODT de una carpeta."""
    return {ruta.name: ingestar_documento(ruta) for ruta in listar_odt(carpeta)}


def main() -> None:
    """Punto de entrada de la ingesta manual."""
    parser = argparse.ArgumentParser(description="Ingesta ODT en Chroma")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra la colección antes de indexar.",
    )
    args = parser.parse_args()

    rutas = listar_odt()
    if not rutas:
        print(f"No hay ODT en {DATOS_DIR}")
        return

    if args.reset:
        borrar_coleccion()
        print("Colección borrada.")

    total = 0
    for ruta in rutas:
        fragmentos = ingestar_documento(ruta)
        total += fragmentos
        print(f"{ruta.name}: {fragmentos} fragmentos")

    print(f"Total ingerido: {total}")
    print(f"Fragmentos en la colección: {contar_fragmentos()}")


if __name__ == "__main__":
    main()

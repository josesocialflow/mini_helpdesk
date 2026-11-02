"""Interfaz Streamlit del mini helpdesk.

Una sola vista: pregunta, recuperación híbrida y respuesta extractiva.
No hay claves, APIs ni generación: la respuesta es el fragmento de la
documentación que mejor encaja, con su fuente.
"""

import streamlit as st
from config import CONFIDENCE_THRESHOLD, HYBRID_SEARCH, TOP_K
from responder import responder
from vectorstore import contar_fragmentos

st.set_page_config(
    page_title="Mini Helpdesk HTML5",
    page_icon="🔎",
    layout="centered",
)

ESTILOS = """
<style>
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(900px 420px at 12% -10%, rgba(139, 92, 246, .16), transparent 60%),
      radial-gradient(800px 380px at 95% 0%, rgba(14, 165, 233, .12), transparent 55%),
      #0e1016;
  }

  .block-container { padding-top: 1.8rem; max-width: 900px; }

  .hero {
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(148, 163, 184, .18);
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(139, 92, 246, .22), rgba(14, 165, 233, .12));
  }

  .hero h1 { margin: 0; font-size: 1.45rem; letter-spacing: -.02em; }
  .hero p { margin: .2rem 0 0; font-size: .92rem; opacity: .72; }

  [data-testid="stForm"] {
    border: 1px solid rgba(148, 163, 184, .18);
    border-radius: 14px;
    padding: .4rem .2rem;
  }

  .stButton > button,
  .stFormSubmitButton > button {
    border-radius: 10px;
    transition: transform .12s ease, filter .12s ease;
  }

  .stButton > button:hover,
  .stFormSubmitButton > button:hover { transform: translateY(-1px); }

  .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    border: none;
  }
</style>
"""

st.markdown(ESTILOS, unsafe_allow_html=True)


@st.cache_data(ttl=30)
def _fragmentos() -> int:
    """Cuenta los fragmentos sin abrir Chroma en cada interacción."""
    return contar_fragmentos()


def _cabecera() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🔎 Mini Helpdesk HTML5</h1>
          <p>Búsqueda híbrida y respuesta extractiva, cien por cien local.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mostrar_resultado(resultado) -> None:
    st.divider()

    if resultado.encontrado:
        st.progress(
            resultado.confianza,
            text=f"Confianza: {resultado.confianza:.0%}",
        )
        st.markdown("#### Respuesta")
        st.markdown(resultado.texto)
        st.caption(
            f"Fuente: **{resultado.documento}** · `{resultado.candidato_id}` · "
            f"{resultado.justificacion}"
        )
    else:
        st.progress(
            max(resultado.confianza, 0.0),
            text=(
                f"Confianza: {resultado.confianza:.0%} "
                f"(umbral {CONFIDENCE_THRESHOLD:.0%})"
            ),
        )
        st.warning(resultado.texto)
        st.caption(resultado.justificacion)

    with st.expander("Fragmentos recuperados"):
        for numero, candidato in enumerate(resultado.fuentes, start=1):
            distancia = (
                f"{candidato.distancia:.4f}"
                if candidato.distancia is not None
                else "—"
            )
            st.markdown(
                f"**[{numero}] {candidato.documento}** · "
                f"`{candidato.id}` · distancia {distancia}"
            )
            st.caption(candidato.fragmento[:400])


def main() -> None:
    fragmentos = _fragmentos()

    with st.sidebar:
        st.title("🔎 Mini Helpdesk")
        st.caption("Documentación HTML5 en local")

        st.divider()
        st.markdown("**Configuración**")
        st.caption(
            f"Fragmentos indexados: {fragmentos}\n\n"
            "Modelo: `all-MiniLM-L6-v2` (ONNX, local)\n\n"
            f"Búsqueda híbrida: {'sí' if HYBRID_SEARCH else 'no'}\n\n"
            f"Umbral de confianza: {CONFIDENCE_THRESHOLD:.0%}\n\n"
            f"Resultados por consulta: {TOP_K}"
        )

        with st.expander("¿Cómo funciona?"):
            st.markdown(
                "1. Los ODT se convierten a texto y se trocean en fragmentos "
                "de 900 caracteres.\n"
                "2. Cada fragmento se vectoriza en local con "
                "`all-MiniLM-L6-v2` (Chroma + ONNX).\n"
                "3. La búsqueda combina similitud vectorial y coincidencia "
                "exacta de términos (útil para siglas como `WCAG` o `SVG`).\n"
                "4. Se devuelve el fragmento más relevante. Si la confianza "
                "no llega al umbral, se avisa en lugar de inventar."
            )

        st.caption("Sin APIs, sin claves y sin coste.")

        st.divider()
        st.caption(
            "Creado por [jsocialflow.com](https://jsocialflow.com/)"
        )

    _cabecera()

    if fragmentos == 0:
        st.error(
            "La colección está vacía. Ejecuta primero:\n\n"
            "```bash\npython ingest.py\n```"
        )
        return

    with st.form("consulta"):
        pregunta = st.text_input(
            "Tu pregunta",
            placeholder="¿Cómo valido un formulario sin JavaScript?",
        )
        enviar = st.form_submit_button("Buscar", type="primary")

    st.caption(
        "Ejemplos: «diferencia entre canvas y svg» · "
        "«atributos de la etiqueta video» · «qué significa WCAG»"
    )

    if enviar:
        if not pregunta.strip():
            st.warning("Escribe una pregunta antes de buscar.")
        else:
            with st.spinner("Buscando en la documentación…"):
                try:
                    st.session_state["resultado"] = responder(pregunta)
                except Exception as exc:  # noqa: BLE001 - se muestra en la UI
                    st.error(f"No se pudo procesar la consulta: {exc}")
                    st.session_state.pop("resultado", None)

    resultado = st.session_state.get("resultado")
    if resultado:
        _mostrar_resultado(resultado)


main()

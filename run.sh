#!/usr/bin/env bash
# Arranca la app desde su carpeta para que Streamlit lea .streamlit/config.toml.
set -euo pipefail

cd "$(dirname "$0")"

# Usa el entorno virtual local si existe; si no, el Python activo.
if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python
fi

exec "$PYTHON" -m streamlit run app.py "$@"

"""
main.py
========

Entry point. Run locally with `python -m src.dashboard.main`, or deploy
with `gunicorn src.dashboard.main:app`.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from src.dashboard.data import RESULTS_DIR
from src.dashboard.layout import serve_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=str(RESULTS_DIR),
)
app.title = "Clever Hans Dashboard"
app.layout = serve_layout  # callable -> re-read on every page load, picks up new results/ files

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

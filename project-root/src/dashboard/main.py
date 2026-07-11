"""
main.py
========

Entry point. Run locally with `python -m src.dashboard.main`, or deploy
with `gunicorn src.dashboard.main:app`.

Static assets (custom CSS) live in `assets/` next to this module -- Dash
auto-serves that folder. Pre-generated result images live in a separate
`results/` directory two levels up and are served from a dedicated Flask
route rather than Dash's `assets_folder`, so the two concerns (app chrome
vs. notebook-generated artifacts) don't collide.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory

from src.dashboard.data import RESULTS_DIR
from src.dashboard.layout import serve_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Clever Hans Dashboard"
app.layout = serve_layout  # callable -> re-read on every page load, picks up new results/ files


@app.server.route("/results/<path:filename>")
def serve_result_image(filename: str):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

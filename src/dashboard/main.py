"""
main.py
=======

Dashboard entry point. Replaces the old `app.py`, which was a disconnected
placeholder: it built its own Dash app with hard-coded dummy metrics and
placehold.co image URLs, and never used `layout.py` or `callbacks.py` at
all (both of which were already implemented and working).

This file just does what the refactor task called for:

    app.layout = serve_layout()
    register_callbacks(app)
    app.run()

Run with:  python -m src.dashboard.main
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from .layout import serve_layout
from .callbacks import register_callbacks


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.DARKLY,
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
        ],
        suppress_callback_exceptions=True,
    )
    app.title = "CLIP Decision Audit — Clever Hans Dashboard"

    app.layout = serve_layout()
    register_callbacks(app)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)

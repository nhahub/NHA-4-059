"""
Vercel's Python runtime expects a WSGI-callable named `app` (or `handler`)
at the path it builds. Dash's underlying Flask server is exposed as
`app.server` — that's the actual WSGI app Vercel needs to point at, not the
Dash object itself.

This file exists so vercel.json can build src/dashboard/wsgi.py directly.
Update vercel.json's "src" to point here instead of app.py if you hit
"no WSGI callable found" errors during deployment (Day 6 setup task).
"""
from src.dashboard.app import app as dash_app

app = dash_app.server  # Flask server instance — this is what Vercel needs

if __name__ == "__main__":
    app.run()

"""
layout.py
==========

Tabs: Overview (headline stat tiles + comparison table + two Plotly charts)
and one tab per model, each browsing whatever PNGs `results/` currently has
for it. `serve_layout` is re-evaluated on every page load (Dash supports a
callable `app.layout`), so refreshing the page picks up newly generated
images without restarting the server.

Chart styling follows the project's data-viz palette: categorical blue/red
for the "Original vs + Logo" comparison, status green/red for the
mitigation-recovery chart (color there encodes sign, not identity), hairline
gridlines, thin rounded bars, direct value labels at each bar's tip.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from src.dashboard.data import (
    RN50_IMAGES,
    SUPERVISED_RESNET_IMAGES,
    VIT_IMAGES,
    available_images,
    load_comparison_table,
)

# Palette (mirrors assets/style.css's :root tokens -- Plotly figures are
# rendered server-side as static JSON, so they can't read CSS custom
# properties; the light-mode hex values are duplicated here deliberately).
COLOR_ORIGINAL = "#2a78d6"  # categorical slot 1 (blue)
COLOR_LOGO = "#e34948"  # categorical slot 6 (red)
COLOR_GOOD = "#0ca30c"  # status: mitigation recovers accuracy
COLOR_CRITICAL = "#d03b3b"  # status: mitigation does not recover accuracy
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"

CHART_FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=COLOR_TEXT_SECONDARY)


def _stat_tile(label: str, value: str, delta_text: str = "", good: bool = True):
    children = [
        html.Div(label, className="ch-stat-label"),
        html.Div(value, className="ch-stat-value"),
    ]
    if delta_text:
        children.append(html.Div(delta_text, className=f"ch-stat-delta {'good' if good else 'bad'}"))
    return dbc.Col(html.Div(children, className="ch-stat-tile"), width=12, md=4)


def _image_gallery(image_list, empty_message: str):
    images = available_images(image_list)
    if not images:
        return html.Div(empty_message, className="ch-empty-state mt-3")

    cards = [
        dbc.Col(
            html.Div(
                [
                    html.Img(src=f"/results/{fname}", style={"width": "100%", "display": "block"}),
                    html.P(caption, className="ch-image-caption"),
                ],
                className="ch-image-card mb-4",
            ),
            width=12,
            lg=6,
        )
        for fname, caption in images
    ]
    return dbc.Row(cards, className="mt-3")


def _accuracy_figure(df):
    return {
        "data": [
            {
                "x": df["Model"],
                "y": df["Original"],
                "type": "bar",
                "name": "Original",
                "marker": {"color": COLOR_ORIGINAL},
                "width": 0.32,
                "text": [f"{v:.1f}%" for v in df["Original"]],
                "textposition": "outside",
            },
            {
                "x": df["Model"],
                "y": df["+ Logo"],
                "type": "bar",
                "name": "+ Logo",
                "marker": {"color": COLOR_LOGO},
                "width": 0.32,
                "text": [f"{v:.1f}%" for v in df["+ Logo"]],
                "textposition": "outside",
            },
        ],
        "layout": {
            "title": {"text": "Accuracy: original vs. with logo", "font": {**CHART_FONT, "size": 15, "color": "#0b0b0b"}},
            "barmode": "group",
            "bargap": 0.35,
            "font": CHART_FONT,
            "paper_bgcolor": COLOR_SURFACE,
            "plot_bgcolor": COLOR_SURFACE,
            "yaxis": {"title": "Accuracy (%)", "gridcolor": COLOR_GRID, "zerolinecolor": COLOR_GRID, "range": [0, 100]},
            "xaxis": {"gridcolor": COLOR_GRID},
            "legend": {"orientation": "h", "y": -0.18},
            "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
        },
    }


def _recovery_figure(df):
    recovery = df["CH Fix (+Logo)"] - df["+ Logo"]
    colors = [COLOR_GOOD if r > 0 else COLOR_CRITICAL for r in recovery]
    return {
        "data": [
            {
                "x": df["Model"],
                "y": recovery,
                "type": "bar",
                "marker": {"color": colors},
                "width": 0.4,
                "text": [f"{r:+.2f}pt" for r in recovery],
                "textposition": "outside",
            }
        ],
        "layout": {
            "title": {
                "text": "Mitigation recovery (green = helps, red = doesn't)",
                "font": {**CHART_FONT, "size": 15, "color": "#0b0b0b"},
            },
            "font": CHART_FONT,
            "paper_bgcolor": COLOR_SURFACE,
            "plot_bgcolor": COLOR_SURFACE,
            "yaxis": {"title": "CH Fix (+Logo) − raw (+Logo)  [pts]", "gridcolor": COLOR_GRID, "zeroline": True, "zerolinecolor": "#c3c2b7"},
            "xaxis": {"gridcolor": COLOR_GRID},
            "showlegend": False,
            "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
        },
    }


def _overview_tab():
    df = load_comparison_table()
    if df is None:
        return html.Div(
            "No results yet — run the notebook (Sections 0-8) and its push-cell to populate results/.",
            className="ch-empty-state mt-3",
        )

    biggest_effect = df.loc[df["Delta"].idxmin()]
    most_robust = df.loc[df["Delta"].idxmax()]
    recovery = df["CH Fix (+Logo)"] - df["+ Logo"]
    best_recovery_idx = recovery.idxmax()
    best_recovery_model = df.loc[best_recovery_idx, "Model"]
    best_recovery_val = recovery.loc[best_recovery_idx]

    stat_row = dbc.Row(
        [
            _stat_tile("Biggest logo effect", biggest_effect["Model"], f"{biggest_effect['Delta']:+.2f} pts", good=False),
            _stat_tile("Most robust to the logo", most_robust["Model"], f"{most_robust['Delta']:+.2f} pts", good=False),
            _stat_tile("Best mitigation recovery", best_recovery_model, f"{best_recovery_val:+.2f} pts", good=best_recovery_val > 0),
        ],
        className="g-3 mb-4",
    )

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
            "padding": "10px 14px",
            "border": "none",
            "borderBottom": f"1px solid {COLOR_GRID}",
        },
        style_header={
            "backgroundColor": "#f9f9f7",
            "color": COLOR_TEXT_SECONDARY,
            "fontWeight": "600",
            "fontSize": "0.85rem",
            "border": "none",
            "borderBottom": f"1px solid {COLOR_GRID}",
        },
        style_data={"backgroundColor": COLOR_SURFACE, "color": "#0b0b0b"},
        style_as_list_view=True,
    )

    return html.Div(
        [
            stat_row,
            html.Div(table, className="ch-table mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(figure=_accuracy_figure(df), config={"displayModeBar": False}),
                            className="ch-chart-card",
                        ),
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        html.Div(
                            dcc.Graph(figure=_recovery_figure(df), config={"displayModeBar": False}),
                            className="ch-chart-card",
                        ),
                        width=12,
                        lg=6,
                    ),
                ]
            ),
        ]
    )


def serve_layout():
    return dbc.Container(
        [
            html.Div(
                [
                    html.Div("Clever Hans Effect Detection", className="ch-title"),
                    html.Div(
                        "CLIP RN50, CLIP ViT-B/32, and a supervised ResNet-50, each tested for "
                        "reliance on a pasted logo as a classification shortcut.",
                        className="ch-subtitle",
                    ),
                ],
                className="ch-header",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(_overview_tab(), label="Overview", tab_id="overview"),
                    dbc.Tab(
                        _image_gallery(RN50_IMAGES, "No CLIP RN50 results found in results/ yet."),
                        label="CLIP RN50",
                        tab_id="rn50",
                    ),
                    dbc.Tab(
                        _image_gallery(
                            VIT_IMAGES,
                            "No CLIP ViT-B/32 results found in results/ yet — "
                            "run Section 6 of the notebook and push.",
                        ),
                        label="CLIP ViT-B/32",
                        tab_id="vit",
                    ),
                    dbc.Tab(
                        _image_gallery(
                            SUPERVISED_RESNET_IMAGES,
                            "No supervised ResNet-50 results found in results/ yet — "
                            "run Section 7 of the notebook and push.",
                        ),
                        label="Supervised ResNet-50",
                        tab_id="resnet",
                    ),
                ],
                active_tab="overview",
            ),
        ],
        fluid=True,
        className="pb-5",
        style={"maxWidth": "1200px"},
    )

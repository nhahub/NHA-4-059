"""
layout.py
==========

Tabs: Overview (three-way comparison table + interactive charts) and one
tab per model, each browsing whatever PNGs `results/` currently has for it.
`serve_layout` is re-evaluated on every page load (Dash supports a callable
`app.layout`), so refreshing the page picks up newly generated images
without restarting the server.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.dashboard.data import (
    RN50_IMAGES,
    SUPERVISED_RESNET_IMAGES,
    VIT_IMAGES,
    available_images,
    load_comparison_table,
)


def _image_gallery(image_list, empty_message: str):
    images = available_images(image_list)
    if not images:
        return dbc.Alert(empty_message, color="secondary", className="mt-3")

    cards = [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src=f"/assets/{fname}", top=True),
                    dbc.CardBody(html.P(caption, className="card-text small")),
                ],
                className="mb-4",
            ),
            width=12,
            lg=6,
        )
        for fname, caption in images
    ]
    return dbc.Row(cards, className="mt-3")


def _overview_tab():
    df = load_comparison_table()
    if df is None:
        return dbc.Alert(
            "No results yet — run the notebook (Sections 0-8) and its "
            "push-cell to populate results/.",
            color="secondary",
            className="mt-3",
        )

    table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, className="mt-3")

    accuracy_fig = {
        "data": [
            {"x": df["Model"], "y": df["Original"], "type": "bar", "name": "Original"},
            {"x": df["Model"], "y": df["+ Logo"], "type": "bar", "name": "+ Logo"},
        ],
        "layout": {
            "barmode": "group",
            "title": "Clever Hans Logo Effect Across Model Families",
            "yaxis": {"title": "Accuracy (%)"},
        },
    }

    recovery = df["CH Fix (+Logo)"] - df["+ Logo"]
    recovery_fig = {
        "data": [
            {
                "x": df["Model"],
                "y": recovery,
                "type": "bar",
                "marker": {"color": ["#2ecc71" if r > 0 else "#e74c3c" for r in recovery]},
            }
        ],
        "layout": {
            "title": "Does Mitigation Recover Accuracy?",
            "yaxis": {"title": "CH Fix (+Logo) − raw (+Logo)  [pts]"},
        },
    }

    return html.Div(
        [
            table,
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=accuracy_fig), width=12, lg=6),
                    dbc.Col(dcc.Graph(figure=recovery_fig), width=12, lg=6),
                ]
            ),
        ]
    )


def serve_layout():
    return dbc.Container(
        [
            html.H1("Clever Hans Effect Detection Dashboard", className="my-4"),
            html.P(
                "CLIP RN50, CLIP ViT-B/32, and a supervised ResNet-50, each "
                "tested for reliance on a pasted logo as a classification shortcut.",
                className="text-muted",
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
    )

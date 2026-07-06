"""
UI Components library for the AI Model Evaluator Dashboard.
Contains only pure UI structures. Data loading and logic are strictly excluded.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import List


def create_kpi_card(title: str, icon_class: str, component_id: str, color: str = "primary") -> dbc.Card:
    """
    Creates a standardized KPI metric card.
    """
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"bi {icon_class} me-2", style={"fontSize": "1.2rem"}),
                html.Span(title, className="text-uppercase small fw-bold text-muted")
            ], className="d-flex align-items-center mb-2"),
            html.H3("---", id=component_id, className=f"mb-0 fw-bold text-{color}")
        ]),
        className="shadow-sm border-0 border-top border-4 border-" + color + " bg-dark text-white h-100"
    )


def create_navigation_panel(classes: List[str]) -> dbc.Card:
    """
    Creates the sidebar navigation and control panel.
    """
    return dbc.Card(
        dbc.CardBody([
            html.H5([html.I(className="bi bi-search me-2"), "Audit Controls"], className="card-title text-primary mb-3"),
            
            html.Div([
                html.Label("Target Class", className="fw-bold small text-muted"),
                dcc.Dropdown(
                    id="class-selector",
                    options=[{"label": c.replace("_", " ").title(), "value": c} for c in classes],
                    placeholder="Select class...",
                    className="mb-3 dash-bootstrap"
                ),
            ]),

            html.Div([
                html.Label("Image Index", className="fw-bold small text-muted"),
                dcc.Dropdown(
                    id="image-selector",
                    placeholder="Select image...",
                    className="mb-3 dash-bootstrap"
                ),
            ]),
            
            dbc.Button(
                [html.I(className="bi bi-arrow-clockwise me-2"), "Manual Refresh"],
                id="refresh-btn",
                color="outline-info",
                size="sm",
                className="w-100 mt-2"
            )
        ]),
        className="bg-dark border-secondary shadow mb-4"
    )


def create_display_panel(title: str, component_id: str, badge_text: str) -> dbc.Card:
    """
    Standard card for image or heatmap display.
    """
    return dbc.Card([
        dbc.CardHeader([
            html.Span(title, className="fw-bold small"),
            dbc.Badge(badge_text, color="secondary", className="ms-2 float-end")
        ], className="bg-transparent border-secondary text-white py-1"),
        dbc.CardBody(
            dcc.Loading(
                type="dot",
                color="#0dcaf0",
                children=html.Img(
                    id=component_id,
                    className="img-fluid rounded",
                    style={"width": "100%", "minHeight": "280px", "objectFit": "contain", "backgroundColor": "#000"}
                )
            ),
            className="p-1"
        )
    ], className="bg-dark border-secondary shadow-sm mb-3")


def create_experiment_panel() -> dbc.Card:
    """
    Controls for the Logo/Text Dependency experiment (Tab 3).
    """
    return dbc.Card(
        dbc.CardBody([
            html.H5("Experiment Settings", className="text-warning mb-3"),
            html.Label("Modification Method", className="fw-bold small text-muted"),
            dbc.RadioItems(
                id="mod-method-selector",
                options=[
                    {"label": "Gaussian Blur", "value": "blur"},
                    {"label": "Background Replace", "value": "replace"},
                    {"label": "Crop & Resize", "value": "crop"},
                ],
                value="blur",
                className="mb-3 text-light small"
            ),
            html.Div(id="experiment-status", className="small text-muted fst-italic")
        ]),
        className="bg-dark border-secondary shadow"
    )


def create_reasoning_panel() -> dbc.Card:
    """
    Displays plain-language model reasoning and confidence metrics.
    """
    return dbc.Card(
        dbc.CardBody([
            html.H6([html.I(className="bi bi-cpu me-2"), "Inference Logic"], className="text-info mb-3"),
            html.Div(id="inference-details", className="text-light small")
        ]),
        className="bg-dark border-info border-start border-4 shadow"
    )

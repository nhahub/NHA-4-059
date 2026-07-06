"""
UI Layout definition for the AI Model Evaluator.
Strictly maps UI components to the Dash Container.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from .components import (
    create_kpi_card, create_navigation_panel, create_display_panel,
    create_experiment_panel, create_reasoning_panel
)

# Constants for standard ImageNet classes (SAD Section 6.1)
TRUCK_CLASSES = [
    "garbage_truck", "moving_van", "pickup_truck", "trailer_truck",
    "beer_truck", "fire_engine", "tow_truck", "minivan"
]

def serve_layout():
    """Returns the complete dashboard layout."""
    return html.Div([
        # System Heartbeat and Data Store
        dcc.Interval(id="auto-refresh", interval=60000, n_intervals=0),
        dcc.Store(id="store-summary-data"),
        
        dbc.Container([
            # Header Block
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("CLIP DECISION AUDIT", className="fw-bold text-white mb-0 mt-4"),
                        html.P("Reveal Hidden Shortcuts in Machine Learning Models", className="text-primary")
                    ], className="border-bottom border-secondary pb-3 mb-4")
                ], width=12)
            ]),

            # Tabbed Interface
            dbc.Tabs([
                
                # Tab 1: Global Performance
                dbc.Tab(label="SYSTEM PERFORMANCE", tab_id="tab-perf", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col(create_kpi_card("Accuracy", "bi-bullseye", "kpi-accuracy", "success"), width=3),
                            dbc.Col(create_kpi_card("Macro F1", "bi-activity", "kpi-f1", "primary"), width=3),
                            dbc.Col(create_kpi_card("Test Images", "bi-images", "kpi-images", "info"), width=3),
                            dbc.Col(create_kpi_card("Class Count", "bi-tags", "kpi-classes", "secondary"), width=3),
                        ], className="g-3 mt-3"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Per-Class F1 Score (Clean)", className="fw-bold"),
                                    dbc.CardBody(dcc.Graph(id="graph-f1-bars", config={'displayModeBar': False}))
                                ], className="bg-dark border-secondary mt-4")
                            ], width=7),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Zero-Shot Confusion Matrix", className="fw-bold"),
                                    dbc.CardBody(dcc.Graph(id="graph-confusion", config={'displayModeBar': False}))
                                ], className="bg-dark border-secondary mt-4")
                            ], width=5),
                        ], className="g-3"),
                    ], className="pb-5")
                ]),

                # Tab 2: XAI Audit Browser
                dbc.Tab(label="XAI AUDIT BROWSER", tab_id="tab-browser", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(create_navigation_panel(TRUCK_CLASSES), className="mt-4"),
                            create_reasoning_panel()
                        ], width=3),
                        
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(create_display_panel("Original Input", "img-clean", "Clean"), width=6),
                                dbc.Col(create_display_panel("Grad-CAM Activation", "img-heatmap", "Primary XAI"), width=6),
                            ], className="mt-4 g-3"),
                            
                            dbc.Card([
                                dbc.CardHeader("Accuracy Degradation Analysis (Delta)", className="fw-bold"),
                                dbc.CardBody(dcc.Graph(id="graph-delta-bars", config={'displayModeBar': False}, style={"height": "300px"}))
                            ], className="bg-dark border-secondary mt-2")
                        ], width=9)
                    ], className="pb-5")
                ]),

                # Tab 3: Shortcut Dependency Experiment
                dbc.Tab(label="SHORTCUT DEPENDENCY", tab_id="tab-shortcut", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(create_experiment_panel(), className="mt-4")
                        ], width=3),
                        
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(create_display_panel("Clean", "exp-img-clean", "A"), width=3),
                                dbc.Col(create_display_panel("A - Heatmap", "exp-heat-clean", "B"), width=3),
                                dbc.Col(create_display_panel("Modified", "exp-img-mod", "C"), width=3),
                                dbc.Col(create_display_panel("C - Heatmap", "exp-heat-mod", "D"), width=3),
                            ], className="mt-4 g-2"),
                            
                            dbc.Alert([
                                html.H4([html.I(className="bi bi-lightning-charge me-2"), "Attention Alignment Score"]),
                                html.P("Ratio of heatmap energy concentrated in suspected shortcut regions.", className="small mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Small("Clean Ratio (B)", className="text-muted"),
                                        html.H3("0.000", id="val-ratio-clean", className="text-info")
                                    ], width=3),
                                    dbc.Col([
                                        html.Small("Modified Ratio (D)", className="text-muted"),
                                        html.H3("0.000", id="val-ratio-mod", className="text-warning")
                                    ], width=3),
                                    dbc.Col([
                                        html.Small("Energy Delta (B-D)", className="text-muted"),
                                        html.H3("0.000", id="val-ratio-delta", className="text-danger")
                                    ], width=6),
                                ])
                            ], color="dark", className="border border-secondary mt-2 shadow-sm")
                        ], width=9)
                    ], className="pb-5")
                ]),

            ], id="main-tabs", active_tab="tab-perf", className="mt-2 border-0")
        ], fluid=True)
    ], className="bg-black min-vh-100")

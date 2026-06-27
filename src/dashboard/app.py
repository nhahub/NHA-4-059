"""
Plotly Dash dashboard — 3 tabs:
  1. Metrics:    class-level accuracy/F1, confusion matrix, accuracy-delta chart
  2. Browser:    image-by-image view with Grad-CAM overlay toggle
  3. Comparison: side-by-side clean vs. logo-inserted view for the same image

Owner: Person E (Dashboard / Frontend Lead)
Day: 2 (skeleton) -> Day 3 (Tab 1 live) -> Day 4 (Tab 2) -> Day 5 (Tab 3)

Run: python src/dashboard/app.py
     Visit http://localhost:8050
"""
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import Input, Output, dcc, html

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = config["dashboard"]["title"]


def load_predictions(split: str) -> pd.DataFrame:
    path = Path(config["paths"]["predictions_out"]) / f"{split}_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Placeholder so the dashboard renders before Person A's pipeline has run
    return pd.DataFrame(columns=["image_path", "true_class", "predicted_class", "confidence", "correct"])


# ---------------- Tab 1: Metrics ----------------
def render_metrics_tab():
    df_orig = load_predictions("original")
    df_logo = load_predictions("logo_variant")

    if df_orig.empty:
        return dbc.Alert(
            "No predictions found yet. Run src/clip_inference/run_zero_shot.py "
            "(Person A) to populate this tab.", color="warning"
        )

    acc_by_class = df_orig.groupby("true_class")["correct"].mean().reset_index()
    fig_acc = px.bar(acc_by_class, x="true_class", y="correct",
                      title="Per-class accuracy (original test set)",
                      labels={"correct": "Accuracy", "true_class": "Class"})
    fig_acc.update_yaxes(range=[0, 1])

    children = [dcc.Graph(figure=fig_acc)]

    if not df_logo.empty:
        orig_acc = df_orig.groupby("true_class")["correct"].mean()
        logo_acc = df_logo.groupby("true_class")["correct"].mean()
        delta = (logo_acc - orig_acc).reset_index()
        delta.columns = ["true_class", "accuracy_delta"]
        fig_delta = px.bar(delta, x="true_class", y="accuracy_delta",
                            title="Accuracy delta: logo-inserted minus original",
                            labels={"accuracy_delta": "Δ Accuracy", "true_class": "Class"},
                            color="accuracy_delta", color_continuous_scale="RdYlGn")
        children.append(dcc.Graph(figure=fig_delta))
    else:
        children.append(dbc.Alert(
            "Logo-variant predictions not found yet — run "
            "src/clip_inference/run_zero_shot.py --dataset logo_variant "
            "to see the accuracy-delta chart here.", color="info"))

    return html.Div(children)


# ---------------- Tab 2: Image Browser ----------------
def render_browser_tab():
    df = load_predictions("original")
    if df.empty:
        return dbc.Alert("No predictions found yet (Tab 1 note applies here too).", color="warning")

    class_options = [{"label": c, "value": c} for c in sorted(df["true_class"].unique())]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Filter by class"),
                dcc.Dropdown(id="browser-class-filter", options=class_options,
                              value=class_options[0]["value"] if class_options else None),
            ], width=4),
            dbc.Col([
                html.Label("Show Grad-CAM overlay"),
                dbc.Switch(id="browser-heatmap-toggle", value=True),
            ], width=4),
        ], className="mb-3"),
        html.Div(id="browser-image-grid"),
    ])


@app.callback(
    Output("browser-image-grid", "children"),
    Input("browser-class-filter", "value"),
    Input("browser-heatmap-toggle", "value"),
)
def update_browser(selected_class, show_heatmap):
    df = load_predictions("original")
    if df.empty or selected_class is None:
        return dbc.Alert("No data to display.", color="warning")

    subset = df[df["true_class"] == selected_class].head(12)
    cards = []
    for _, row in subset.iterrows():
        badge_color = "success" if row["correct"] else "danger"
        cards.append(dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P(Path(row["image_path"]).name, className="small text-truncate"),
                dbc.Badge(f"Pred: {row['predicted_class']} ({row['confidence']:.2f})", color=badge_color),
                # TODO (Person E, Day 4): render the actual image + heatmap
                # overlay here, using outputs/heatmaps/original/<class>/<stem>.npy
                # when show_heatmap is True. A straightforward approach:
                # render the base image with plotly express imshow, then
                # overlay the heatmap as a semi-transparent second imshow trace.
            ])
        ]), width=3))
    return dbc.Row(cards)


# ---------------- Tab 3: Logo Comparison ----------------
def render_comparison_tab():
    return dbc.Alert(
        "TODO (Person E, Day 5): build side-by-side clean vs. logo-inserted "
        "comparison view here. Pick a few representative images (garbage "
        "truck and beer truck are the strongest candidates per the project "
        "doc), show original image + heatmap next to logo-inserted image + "
        "heatmap, with predicted class and confidence labeled on both.",
        color="info"
    )


# ---------------- App layout ----------------
app.layout = dbc.Container([
    html.H1(config["dashboard"]["title"], className="my-4"),
    dcc.Tabs([
        dcc.Tab(label="Metrics", children=[render_metrics_tab()]),
        dcc.Tab(label="Image Browser", children=[render_browser_tab()]),
        dcc.Tab(label="Logo Comparison", children=[render_comparison_tab()]),
    ]),
], fluid=True)


if __name__ == "__main__":
    app.run(debug=True, port=config["dashboard"]["port"])

"""
Logic Engine for the AI Model Evaluator.
Handles all Data loading, Plotting, and UI reactivity.
"""

import logging
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, html
from typing import Dict, List, Tuple

# --- Logging setup ---
logger = logging.getLogger(__name__)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"

def load_data_safe(file_path: Path) -> pd.DataFrame:
    """Loads CSV and returns empty DF on failure."""
    try:
        if file_path.exists():
            return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
    return pd.DataFrame()

def register_callbacks(app):
    """Encapsulates all dashboard callback logic."""

    @app.callback(
        [Output("kpi-accuracy", "children"),
         Output("kpi-f1", "children"),
         Output("kpi-images", "children"),
         Output("kpi-classes", "children"),
         Output("graph-f1-bars", "figure"),
         Output("graph-confusion", "figure"),
         Output("graph-delta-bars", "figure")],
        [Input("auto-refresh", "n_intervals"),
         Input("refresh-btn", "n_clicks")]
    )
    def update_performance_tab(_, __):
        """Refreshes global metrics and charts."""
        # Load Raw Data
        df_summary = load_data_safe(METRICS_DIR / "summary_metrics.csv")
        df_clean = load_data_safe(METRICS_DIR / "metrics_clean.csv")
        df_cm = load_data_safe(METRICS_DIR / "confusion_matrix.csv")
        df_delta = load_data_safe(METRICS_DIR / "accuracy_delta.csv")

        # 1. Parse Summary (Contract: metric,value)
        kpis = {"overall_accuracy": "---", "macro_f1": "---", "total_images": "---", "num_classes": "---"}
        if not df_summary.empty and 'metric' in df_summary.columns:
            lookup = dict(zip(df_summary['metric'], df_summary['value']))
            kpis["overall_accuracy"] = f"{float(lookup.get('overall_accuracy', 0)):.1%}"
            kpis["macro_f1"] = f"{float(lookup.get('macro_f1', 0)):.3f}"
            kpis["total_images"] = f"{int(lookup.get('total_images', 0))}"
            kpis["num_classes"] = f"{int(lookup.get('num_classes', 0))}"

        # 2. Per-Class F1 Chart (go.Bar requirement)
        fig_f1 = go.Figure().update_layout(template="plotly_dark", height=350)
        if not df_clean.empty and 'f1' in df_clean.columns:
            fig_f1.add_trace(go.Bar(x=df_clean['class'], y=df_clean['f1'], marker_color='#3498db'))
            fig_f1.update_layout(margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Class", yaxis_title="F1")

        # 3. Confusion Matrix (px.imshow requirement)
        fig_cm = go.Figure().update_layout(template="plotly_dark", height=350)
        if not df_cm.empty:
            cols = df_cm.columns.tolist()
            fig_cm = px.imshow(df_cm.values, x=cols, y=cols, text_auto=".2f", color_continuous_scale='Blues', template="plotly_dark")
            fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10))

        # 4. Accuracy Delta Chart
        fig_delta = go.Figure().update_layout(template="plotly_dark", height=300)
        if not df_delta.empty:
            fig_delta = px.bar(df_delta, x="class", y="delta", color="method", barmode="group", template="plotly_dark")
            fig_delta.update_layout(margin=dict(l=20, r=20, t=30, b=20))

        return (kpis["overall_accuracy"], kpis["macro_f1"], kpis["total_images"], kpis["num_classes"], 
                fig_f1, fig_cm, fig_delta)

    @app.callback(
        Output("image-selector", "options"),
        Input("class-selector", "value")
    )
    def update_image_list(selected_class):
        """Pops image list for the dropdown based on filesystem walk."""
        if not selected_class: return []
        class_path = DATA_DIR / "clean" / selected_class
        if class_path.exists():
            files = sorted([f.name for f in class_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            return [{"label": f, "value": f} for f in files]
        return []

    @app.callback(
        [Output("img-clean", "src"),
         Output("img-heatmap", "src"),
         Output("inference-details", "children"),
         Output("exp-img-clean", "src"),
         Output("exp-heat-clean", "src"),
         Output("exp-img-mod", "src"),
         Output("exp-heat-mod", "src"),
         Output("val-ratio-clean", "children"),
         Output("val-ratio-mod", "children"),
         Output("val-ratio-delta", "children")],
        [Input("image-selector", "value"),
         Input("mod-method-selector", "value")],
        [State("class-selector", "value")]
    )
    def sync_views_and_ratios(img_name, method, cls):
        """Updates all image views and energy ratio KPIs."""
        if not img_name or not cls:
            return [""]*7 + ["0.000", "0.000", "0.000"]

        stem = Path(img_name).stem
        
        # 1. Image Route Construction (Flask Static Assets)
        url_clean = f"/assets/data/clean/{cls}/{img_name}"
        url_heat_c = f"/assets/outputs/heatmaps/clean/{cls}/{stem}.png"
        url_mod = f"/assets/data/logo_{method}/{cls}/{img_name}"
        url_heat_m = f"/assets/outputs/heatmaps/{method}/{cls}/{stem}.png"

        # 2. Reasoning Logic
        df_metrics = load_data_safe(METRICS_DIR / "metrics_clean.csv")
        reasoning = html.Span("Record not found in log.", className="text-muted")
        if not df_metrics.empty:
            row = df_metrics[df_metrics['filename'] == img_name] if 'filename' in df_metrics.columns else pd.DataFrame()
            if not row.empty:
                pred = row.iloc[0].get('predicted_class', 'Unknown').replace("_", " ").title()
                conf = row.iloc[0].get('confidence', 0)
                reasoning = html.Div([
                    html.B(f"Prediction: {pred}"), html.Br(),
                    html.Span(f"Confidence: {conf:.2%}", className="text-info")
                ])

        # 3. Energy Ratio Logic (Official Schema: filename, class, method, energy_ratio)
        df_energy = load_data_safe(METRICS_DIR / "energy_ratio_all.csv")
        r_clean = r_mod = 0.0
        if not df_energy.empty:
            c_val = df_energy[(df_energy['filename'] == img_name) & (df_energy['method'] == 'clean')]
            m_val = df_energy[(df_energy['filename'] == img_name) & (df_energy['method'] == method)]
            r_clean = c_val.iloc[0].get('energy_ratio', 0.0) if not c_val.empty else 0.0
            r_mod = m_val.iloc[0].get('energy_ratio', 0.0) if not m_val.empty else 0.0

        return (url_clean, url_heat_c, reasoning, 
                url_clean, url_heat_c, url_mod, url_heat_m, 
                f"{r_clean:.3f}", f"{r_mod:.3f}", f"{(r_clean - r_mod):+.3f}")

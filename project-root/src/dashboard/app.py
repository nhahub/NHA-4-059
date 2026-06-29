```python
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

# -------------------------------------------------
# Create Dash App
# -------------------------------------------------

app = Dash(__name__)
app.title = "Clever Hans Dashboard"

# -------------------------------------------------
# Dummy Metrics Data
# -------------------------------------------------

classes = ["Cat", "Dog", "Bird", "Car"]

metrics_df = pd.DataFrame({
    "Class": classes,
    "F1 Score": [0.91, 0.84, 0.88, 0.79]
})

fig_f1 = px.bar(
    metrics_df,
    x="Class",
    y="F1 Score",
    title="Per-Class F1 Score",
    text_auto=".2f"
)

confusion = [
    [18, 2, 0, 0],
    [1, 16, 2, 1],
    [0, 1, 17, 2],
    [0, 0, 2, 18]
]

fig_cm = px.imshow(
    confusion,
    x=classes,
    y=classes,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Confusion Matrix"
)

# -------------------------------------------------
# Layout
# -------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "20px"},
    children=[

        html.H1(
            "Clever Hans CLIP Dashboard",
            style={"textAlign": "center"}
        ),

        dcc.Tabs([

            # =====================================
            # TAB 1
            # =====================================

            dcc.Tab(
                label="Metrics",
                children=[

                    html.Br(),

                    dcc.Graph(figure=fig_f1),

                    dcc.Graph(figure=fig_cm),

                ],
            ),

            # =====================================
            # TAB 2
            # =====================================

            dcc.Tab(
                label="Image Browser",
                children=[

                    html.Br(),

                    html.H3("Select Class"),

                    dcc.Dropdown(
                        id="class-dropdown",
                        options=[
                            {"label": c, "value": c}
                            for c in classes
                        ],
                        value="Cat",
                        style={"width": "300px"},
                    ),

                    html.Br(),

                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-around",
                            "marginBottom": "30px",
                        },
                        children=[

                            html.Div([
                                html.H4("Clean Image"),
                                html.Img(
                                    src="https://placehold.co/300x300?text=Clean+Image"
                                ),
                            ]),

                            html.Div([
                                html.H4("Logo Image"),
                                html.Img(
                                    src="https://placehold.co/300x300?text=Logo+Image"
                                ),
                            ]),

                        ],
                    ),

                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-around",
                        },
                        children=[

                            html.Div([
                                html.H4("Heatmap"),
                                html.Img(
                                    src="https://placehold.co/300x300?text=Heatmap"
                                ),
                            ]),

                            html.Div([
                                html.H4("Overlay"),
                                html.Img(
                                    src="https://placehold.co/300x300?text=Overlay"
                                ),
                            ]),

                        ],
                    ),

                    html.Br(),

                ],
            ),

            # =====================================
            # TAB 3
            # =====================================

            dcc.Tab(
                label="Logo Comparison",
                children=[

                    html.Br(),

                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(2, 1fr)",
                            "gap": "20px",
                            "maxWidth": "650px",
                            "margin": "auto",
                        },
                        children=[

                            html.Div([
                                html.H4("Clean"),
                                html.Img(
                                    src="https://placehold.co/250x250?text=Clean"
                                ),
                            ]),

                            html.Div([
                                html.H4("Logo Blur"),
                                html.Img(
                                    src="https://placehold.co/250x250?text=Blur"
                                ),
                            ]),

                            html.Div([
                                html.H4("Logo Replace"),
                                html.Img(
                                    src="https://placehold.co/250x250?text=Replace"
                                ),
                            ]),

                            html.Div([
                                html.H4("Logo Crop"),
                                html.Img(
                                    src="https://placehold.co/250x250?text=Crop"
                                ),
                            ]),

                        ],
                    ),

                    html.Br(),

                    html.H2(
                        "Energy Ratio Score",
                        style={"textAlign": "center"},
                    ),

                    html.H1(
                        "0.74",
                        style={
                            "textAlign": "center",
                            "color": "green",
                        },
                    ),

                    html.Br(),

                ],
            ),

        ]),
    ],
)

# -------------------------------------------------
# Run App
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
```

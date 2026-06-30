"""
==============================================================
Clever Hans CLIP Dashboard

File:
    src/dashboard/components.py

Author:
    Hassan Mostafa

Description
-----------
Reusable dashboard UI components.

This file contains all visual components used
throughout the dashboard.

NO CALLBACKS HERE.

NO DATA LOADING HERE.

Callbacks live inside callbacks.py

Layout lives inside layout.py

==============================================================
"""

from dash import html
from dash import dcc
import dash_bootstrap_components as dbc


# ============================================================
# Colors
# ============================================================

BACKGROUND = "#101820"

CARD_COLOR = "#1B263B"

PRIMARY = "#00C896"

SECONDARY = "#3A506B"

TEXT = "#F8F9FA"

BORDER = "#2C3E50"

WARNING = "#F4D03F"

SUCCESS = "#2ECC71"

DANGER = "#E74C3C"


# ============================================================
# Shared Styles
# ============================================================

CARD_STYLE = {

    "backgroundColor": CARD_COLOR,

    "border": f"1px solid {BORDER}",

    "borderRadius": "14px",

    "padding": "18px",

    "marginBottom": "20px",

    "boxShadow": "0px 3px 10px rgba(0,0,0,.35)"

}


TITLE_STYLE = {

    "color": TEXT,

    "fontWeight": "bold",

    "marginBottom": "10px"

}


CENTER_STYLE = {

    "textAlign": "center"

}


IMAGE_STYLE = {

    "width": "100%",

    "borderRadius": "10px",

    "border": "1px solid #333"

}


# ============================================================
# Header
# ============================================================

def create_header():

    return dbc.Navbar(

        dbc.Container(

            [

                dbc.Row(

                    [

                        dbc.Col(

                            html.Div(

                                [

                                    html.H2(

                                        "Clever Hans CLIP Dashboard",

                                        className="mb-0",

                                        style={

                                            "color": TEXT,

                                            "fontWeight": "bold"

                                        }

                                    ),

                                    html.Small(

                                        "Reveal Hidden Decision Patterns",

                                        style={

                                            "color": "#BFC9CA"

                                        }

                                    )

                                ]

                            ),

                            width=10

                        ),

                        dbc.Col(

                            dbc.Button(

                                "Refresh",

                                id="refresh-button",

                                color="success",

                                n_clicks=0

                            ),

                            width=2,

                            className="text-end"

                        )

                    ],

                    align="center"

                )

            ],

            fluid=True

        ),

        color="#111827",

        dark=True,

        className="mb-4"

    )


# ============================================================
# Footer
# ============================================================

def create_footer():

    return html.Div(

        [

            html.Hr(),

            html.P(

                "Clever Hans CLIP Project © 2026",

                style={

                    "textAlign": "center",

                    "color": "#AAAAAA"

                }

            )

        ]

    )


# ============================================================
# Summary Card
# ============================================================

def summary_card(

        title,

        value,

        card_id,

        color=PRIMARY

):

    return dbc.Card(

        dbc.CardBody(

            [

                html.H6(

                    title,

                    style={

                        "color": "#BBBBBB"

                    }

                ),

                html.H2(

                    value,

                    id=card_id,

                    style={

                        "color": color,

                        "fontWeight": "bold"

                    }

                )

            ]

        ),

        style=CARD_STYLE

    )
#part2
# ============================================================
# Summary Panel
# ============================================================

def create_summary_cards():

    return dbc.Row(

        [

            dbc.Col(

                summary_card(

                    "Overall Accuracy",

                    "--",

                    "overall-accuracy"

                ),

                md=3

            ),

            dbc.Col(

                summary_card(

                    "Macro F1",

                    "--",

                    "macro-f1"

                ),

                md=3

            ),

            dbc.Col(

                summary_card(

                    "Test Images",

                    "--",

                    "test-images"

                ),

                md=3

            ),

            dbc.Col(

                summary_card(

                    "Classes",

                    "--",

                    "num-classes"

                ),

                md=3

            )

        ]

    )
  # ============================================================
# Loading Spinner
# ============================================================

def create_loading(component):

    return dcc.Loading(

        type="circle",

        color=PRIMARY,

        fullscreen=False,

        children=component

    )


# ============================================================
# Empty State
# ============================================================

def create_empty_state(message="Waiting for data..."):

    return html.Div(

        [

            html.H4(

                "No Data Available",

                style={

                    "color": "#CCCCCC",

                    "textAlign": "center"

                }

            ),

            html.P(

                message,

                style={

                    "color": "#888888",

                    "textAlign": "center"

                }

            )

        ],

        style={

            "padding": "60px"

        }

    )


# ============================================================
# Refresh Status
# ============================================================

def create_refresh_status():

    return html.Div(

        id="refresh-status",

        children="",

        style={

            "color": SUCCESS,

            "marginBottom": "10px",

            "fontWeight": "bold"

        }

    )


# ============================================================
# Auto Refresh Timer
# ============================================================

def create_interval():

    return dcc.Interval(

        id="auto-refresh",

        interval=5000,

        n_intervals=0

    )


# ============================================================
# Per-Class F1 Chart
# ============================================================

def create_f1_chart():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H4(

                    "Per-Class F1 Score",

                    style=TITLE_STYLE

                ),

                create_loading(

                    dcc.Graph(

                        id="f1-chart",

                        config={

                            "displaylogo": False,

                            "responsive": True

                        },

                        style={

                            "height": "500px"

                        }

                    )

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Confusion Matrix
# ============================================================

def create_confusion_matrix():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H4(

                    "Confusion Matrix",

                    style=TITLE_STYLE

                ),

                create_loading(

                    dcc.Graph(

                        id="confusion-matrix",

                        config={

                            "displaylogo": False,

                            "responsive": True

                        },

                        style={

                            "height": "700px"

                        }

                    )

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Metrics Tab
# ============================================================

def create_tab1():

    return dbc.Container(

        [

            create_interval(),

            create_refresh_status(),

            create_summary_cards(),

            html.Br(),

            create_f1_chart(),

            html.Br(),

            create_confusion_matrix(),

        ],

        fluid=True

    )
#part3
# ============================================================
# Class Dropdown
# ============================================================

def create_class_dropdown():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Select Class",

                    style=TITLE_STYLE

                ),

                dcc.Dropdown(

                    id="class-dropdown",

                    placeholder="Choose a class...",

                    clearable=False,

                    searchable=True

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Image Selector
# ============================================================

def create_image_selector():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Select Image",

                    style=TITLE_STYLE

                ),

                dcc.Dropdown(

                    id="image-dropdown",

                    placeholder="Choose an image...",

                    clearable=False,

                    searchable=True

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Original Image Panel
# ============================================================

def create_original_image():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Original Image",

                    style=TITLE_STYLE

                ),

                create_loading(

                    html.Img(

                        id="original-image",

                        style=IMAGE_STYLE

                    )

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Heatmap Panel
# ============================================================

def create_heatmap_image():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Grad-CAM Heatmap",

                    style=TITLE_STYLE

                ),

                create_loading(

                    html.Img(

                        id="heatmap-image",

                        style=IMAGE_STYLE

                    )

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Image Information
# ============================================================

def create_image_information():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Image Information",

                    style=TITLE_STYLE

                ),

                html.Div(

                    id="image-information",

                    children="",

                    style={

                        "color": TEXT,

                        "fontSize": "15px",

                        "lineHeight": "30px"

                    }

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Tab 2
# ============================================================

def create_tab2():

    return dbc.Container(

        [

            dbc.Row(

                [

                    dbc.Col(

                        create_class_dropdown(),

                        md=6

                    ),

                    dbc.Col(

                        create_image_selector(),

                        md=6

                    )

                ]

            ),

            html.Br(),

            dbc.Row(

                [

                    dbc.Col(

                        create_original_image(),

                        md=6

                    ),

                    dbc.Col(

                        create_heatmap_image(),

                        md=6

                    )

                ]

            ),

            html.Br(),

            create_image_information()

        ],

        fluid=True

    )
#part4
# ============================================================
# Comparison Method Dropdown
# ============================================================

def create_method_selector():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Modification Method",

                    style=TITLE_STYLE

                ),

                dcc.Dropdown(

                    id="method-dropdown",

                    options=[

                        {
                            "label": "Gaussian Blur",
                            "value": "blur"
                        },

                        {
                            "label": "Background Replace",
                            "value": "replace"
                        },

                        {
                            "label": "Crop & Resize",
                            "value": "crop"
                        }

                    ],

                    value="blur",

                    clearable=False

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Comparison Image Card
# ============================================================

def comparison_image(title, component_id):

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    title,

                    style=TITLE_STYLE

                ),

                create_loading(

                    html.Img(

                        id=component_id,

                        style=IMAGE_STYLE

                    )

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Energy Ratio Card
# ============================================================

def create_energy_ratio_card():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Energy Ratio",

                    style=TITLE_STYLE

                ),

                dbc.Row(

                    [

                        dbc.Col(

                            [

                                html.Small(

                                    "Clean",

                                    style={

                                        "color": "#BBBBBB"

                                    }

                                ),

                                html.H3(

                                    "--",

                                    id="energy-clean",

                                    style={

                                        "color": SUCCESS,

                                        "fontWeight": "bold"

                                    }

                                )

                            ],

                            md=6

                        ),

                        dbc.Col(

                            [

                                html.Small(

                                    "Modified",

                                    style={

                                        "color": "#BBBBBB"

                                    }

                                ),

                                html.H3(

                                    "--",

                                    id="energy-modified",

                                    style={

                                        "color": WARNING,

                                        "fontWeight": "bold"

                                    }

                                )

                            ],

                            md=6

                        )

                    ]

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Plain Language Caption
# ============================================================

def create_attention_caption():

    return dbc.Card(

        dbc.CardBody(

            [

                html.H5(

                    "Attention Summary",

                    style=TITLE_STYLE

                ),

                html.Div(

                    id="attention-summary",

                    children="",

                    style={

                        "color": TEXT,

                        "fontSize": "16px",

                        "lineHeight": "30px"

                    }

                )

            ]

        ),

        style=CARD_STYLE

    )


# ============================================================
# Tab 3
# ============================================================

def create_tab3():

    return dbc.Container(

        [

            create_method_selector(),

            html.Br(),

            dbc.Row(

                [

                    dbc.Col(

                        comparison_image(

                            "Original Image",

                            "comparison-original"

                        ),

                        md=6

                    ),

                    dbc.Col(

                        comparison_image(

                            "Clean Heatmap",

                            "comparison-clean"

                        ),

                        md=6

                    )

                ]

            ),

            html.Br(),

            dbc.Row(

                [

                    dbc.Col(

                        comparison_image(

                            "Modified Image",

                            "comparison-modified"

                        ),

                        md=6

                    ),

                    dbc.Col(

                        comparison_image(

                            "Modified Heatmap",

                            "comparison-modified-heatmap"

                        ),

                        md=6

                    )

                ]

            ),

            html.Br(),

            create_energy_ratio_card(),

            html.Br(),

            create_attention_caption()

        ],

        fluid=True

    )
#part5

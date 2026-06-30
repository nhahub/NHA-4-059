"""
==============================================================
Clever Hans CLIP Dashboard

File:
    src/dashboard/layout.py

Author:
    Hassan Mostafa

Description
-----------
Main dashboard layout.

This file assembles all reusable components into
the final dashboard interface.

No callbacks here.
No data loading here.

==============================================================
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from components import (
    create_header,
    create_footer,
    create_tab1,
    create_tab2,
    create_tab3,
)


# ============================================================
# Dashboard Layout
# ============================================================

def create_layout():

    return dbc.Container(

        [

            # =====================================================
            # Hidden Stores
            # =====================================================

            dcc.Store(id="metrics-store"),

            dcc.Store(id="summary-store"),

            dcc.Store(id="confusion-store"),

            dcc.Store(id="image-store"),

            dcc.Store(id="comparison-store"),

            # =====================================================
            # Header
            # =====================================================

            create_header(),

            # =====================================================
            # Main Tabs
            # =====================================================

            dcc.Tabs(

                id="main-tabs",

                value="tab-metrics",

                children=[

                    # -------------------------------------------------
                    # TAB 1
                    # -------------------------------------------------

                    dcc.Tab(

                        label="Metrics",

                        value="tab-metrics",

                        children=[

                            html.Br(),

                            create_tab1()

                        ],

                    ),

                    # -------------------------------------------------
                    # TAB 2
                    # -------------------------------------------------

                    dcc.Tab(

                        label="Image Browser",

                        value="tab-browser",

                        children=[

                            html.Br(),

                            create_tab2()

                        ],

                    ),

                    # -------------------------------------------------
                    # TAB 3
                    # -------------------------------------------------

                    dcc.Tab(

                        label="Logo Comparison",

                        value="tab-comparison",

                        children=[

                            html.Br(),

                            create_tab3()

                        ],

                    ),

                ],

            ),

            html.Br(),

            # =====================================================
            # Footer
            # =====================================================

            create_footer(),

        ],

        fluid=True,

        style={

            "minHeight": "100vh",

            "padding": "20px",

            "backgroundColor": "#101820",

        }

    )

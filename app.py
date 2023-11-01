import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from dash_bootstrap_components import Modal, ModalHeader, ModalBody, ModalFooter
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import math
from concurrent.futures import ThreadPoolExecutor
import uuid
import pandas as pd
import random
import numpy as np
import json
from flask import Flask, request
import os
import copy

executor = ThreadPoolExecutor(max_workers=4)  # Define the number of worker threads

load_figure_template(["darkly"])

server = Flask(__name__)
# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], server=server)

# server-side data store
user_data_store = {}

# Default data structure
default_data = {
    'global_settings': {
        'r_cycle': 14,
        'r_cost': 8,
        'k_cost': 0.18
    },
    'items': [],
    'day': 1,
    'is_initialized': False
}

def get_user_data(uuid):
    # Check if uuid is in user_data_store
    if uuid not in user_data_store:
        # If not, create a new entry with default data
        user_data_store[uuid] = copy.deepcopy(default_data)
    return user_data_store.get(uuid, {...})  # returns default data if uuid not found

def set_user_data(uuid, data):
    user_data_store[uuid] = data

#Shutdown and Startup Procedure to save data
def on_shutdown():
    with open('data/user_data.json', 'w') as f:
        json.dump(user_data_store, f)

def load_data():
    if os.path.exists('data/user_data.json'):
        with open('data/user_data.json', 'r') as f:
            return json.load(f)
    return {}

user_data_store = load_data()


def create_global_settings(r_cycle=14, r_cost=8, k_cost=0.18):
    return {
        'r_cycle': r_cycle,
        'r_cost': r_cost,
        'k_cost': k_cost
    }

def create_inventory_item(usage_rate, lead_time, item_cost, pna, safety_allowance, standard_pack, global_settings, hits_per_month):
    daily_ur = usage_rate / 30
    pna_days = pna / daily_ur
    op = calculate_op(usage_rate, lead_time, safety_allowance)
    pna_days_frm_op = ((pna - op) / usage_rate) * 30
    lp = calculate_lp(usage_rate, global_settings, op)
    eoq = calculate_eoq(usage_rate, item_cost, global_settings)
    oq = calculate_oq(eoq, usage_rate, global_settings)
    soq = calculate_soq(pna_days, lp, op, oq, standard_pack)
    surplus_line = calculate_surplus_line(lp, eoq)
    cp = calculate_critical_point(usage_rate, lead_time)
    proposed_pna = pna + soq # This is where the PNA would bump up to if an order is placed
    pro_pna_days_frm_op = ((proposed_pna - op) / usage_rate) * 30 #conv to days from op
    no_pna_days_frm_op = ((0 - op) / usage_rate) * 30

    return {
        'usage_rate': usage_rate, 'lead_time': lead_time, 'item_cost': item_cost, 'pna': pna, 'safety_allowance': safety_allowance, 'standard_pack': standard_pack,
        'daily_ur': daily_ur, 'pna_days': pna_days, 'op': op, 'pna_days_frm_op': pna_days_frm_op, 'lp': lp, 'eoq': eoq, 'oq': oq, 'soq': soq,
        'surplus_line': surplus_line, 'cp': cp, 'proposed_pna': proposed_pna, 'pro_pna_days_frm_op': pro_pna_days_frm_op, 'no_pna_days_frm_op': no_pna_days_frm_op,
        'hits_per_month': hits_per_month
    }

def calculate_op(usage_rate, lead_time, safety_allowance):
    safety_stock = (usage_rate * lead_time * safety_allowance) / 30
    return usage_rate * (lead_time / 30) + safety_stock

def calculate_lp(usage_rate, global_settings, op):
    return (usage_rate * (global_settings['r_cycle'] / 30)) + op

def calculate_eoq(usage_rate, item_cost, global_settings):
    return math.sqrt((24 * global_settings['r_cost'] * usage_rate) / (global_settings['k_cost'] * item_cost))

def calculate_oq(eoq, usage_rate, global_settings):
    oq_min = max(0.5 * usage_rate, global_settings['r_cycle'] / 30 * usage_rate)
    oq_max = 12 * usage_rate
    return max(min(eoq, oq_max), oq_min)

def calculate_soq(pna, lp, op, oq, standard_pack):
    if pna > lp:
        return 0
    elif pna > op:
        return round(oq / standard_pack) * standard_pack
    else:
        return round((oq + op - pna) / standard_pack) * standard_pack

def calculate_surplus_line(lp, eoq):
    return lp + eoq

def calculate_critical_point(usage_rate, lead_time):
    return usage_rate * lead_time

def initial_graph(store_data=None):
    if not store_data or 'items' not in store_data or not store_data['items']:
        fig = px.scatter(title="Inventory Simulation")
        fig.update_layout(xaxis=dict(autorange=True, title="Items"), yaxis=dict(autorange=True), title="PNA (Days from OP)")
    else:
        fig = update_graph_based_on_items(store_data['items'], store_data['global_settings'])
    return fig

def simulate_daily_hits(hpm):
    daily_hit_rate = hpm / 30.0  # Convert monthly rate to daily
    return np.random.poisson(daily_hit_rate)  # Generate a Poisson-distributed random number

def simulate_sales(avg_sale_qty, standard_pack):
    # Generate a Poisson-distributed random number, then round to the nearest standard pack
    raw_sales = np.random.poisson(avg_sale_qty)
    return round(raw_sales / standard_pack) * standard_pack

def process_item(item):
    avg_sale_qty = item['usage_rate']/item['hits_per_month']
    
    # Simulate the number of hits for today
    num_hits_today = simulate_daily_hits(item['hits_per_month'])
    
    # Simulate the sale for each hit
    total_sales_today = 0
    for _ in range(num_hits_today):
        total_sales_today += simulate_sales(avg_sale_qty, item['standard_pack'])
        
    # Update PNA
    item['pna'] -= total_sales_today
    
    return update_pna_related_values(item)

def update_pna_related_values(item):
    item['pna_days_frm_op'] = ((item['pna'] - item['op']) / item['usage_rate']) * 30
    # Now, recalculate all the related fields
    item['pna_days'] = item['pna'] / item['daily_ur']
    item['soq'] = calculate_soq(item['pna'], item['lp'], item['op'], item['oq'], item['standard_pack'])
    item['surplus_line'] = calculate_surplus_line(item['lp'], item['eoq'])
    item['proposed_pna'] = item['pna'] + item['soq']
    item['pro_pna_days_frm_op'] = ((item['proposed_pna'] - item['op']) / item['usage_rate']) * 30  # convert to days from op
    
    return item

def update_global_settings(current_settings, review_cycle, r_cost, k_cost):
    current_settings['r_cycle'] = review_cycle
    current_settings['r_cost'] = r_cost
    current_settings['k_cost'] = k_cost
    return current_settings

def update_gs_related_values(item, global_settings):
    # Recalculate all the properties that depend on global settings
    item['lp'] = calculate_lp(item['usage_rate'], global_settings, item['op'])
    item['eoq'] = calculate_eoq(item['usage_rate'], item['item_cost'], global_settings)
    item['oq'] = calculate_oq(item['eoq'], item['usage_rate'], global_settings)
    item['soq'] = calculate_soq(item['pna'], item['lp'], item['op'], item['oq'], item['standard_pack'])
    item['surplus_line'] = calculate_surplus_line(item['lp'], item['eoq'])
    item['proposed_pna'] = item['pna'] + item['soq']
    item['pro_pna_days_frm_op'] = ((item['proposed_pna'] - item['op']) / item['usage_rate']) * 30  # convert to days from op
    return item

def update_graph_based_on_items(items, global_settings):
    df = pd.DataFrame(items)
    # Create a mask for rows where pro_pna_days_frm_op is different from pna_days_frm_op
    mask = df['pro_pna_days_frm_op'] != df['pna_days_frm_op']
    filtered_df = df[mask]
    fig = px.scatter(df, x=df.index+1, y=df['pna_days_frm_op'], title="Inventory Simulation")
    fig.update_traces(marker=dict(color='blue'))
    fig.add_scatter(x=(filtered_df.index + 1), y=filtered_df['pro_pna_days_frm_op'], mode='markers', name='PNA + SOQ Days from OP', marker=dict(color='green', symbol='circle-open'))
    fig.add_scatter(x=df.index+1, y=df['no_pna_days_frm_op'], mode='markers', name='0 PNA', marker=dict(color='red'))
    fig.add_hline(y=0, line_dash="dot", annotation_text="OP")
    fig.add_hline(y=global_settings['r_cycle'], line_dash="dot", annotation_text="LP")
    fig.update_layout(
        xaxis=dict(autorange=True, tickvals=list(range(1, len(items) + 1)), title="Items"),
        yaxis=dict(autorange=True, title="PNA (Days from OP)")
    )
    return fig

app.title = 'IM Sim'

app.layout = dbc.Container(
    [
        # Navigation Bar
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Inventory Management Simulator", href="#")),
            ],
            brand="CEEUS",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Row(
            [
                # Left column for settings and buttons
                dbc.Col(
                    [
                        html.H4("Controls", style={"margin-top": "20px"}),
                        html.Div(
                            [
                                dbc.Button("Start/Pause Simulation", id="start-button", n_clicks=0, color="success"),
                                dbc.Button("Reset Simulation", id="reset-button", n_clicks=0),
                                dbc.Button("Place Purchase Order", id="po-button", n_clicks=0),
                                dbc.Button("Place Custom Order", id="place-custom-order-button", n_clicks=0, color="warning"),
                                dbc.Button("Add Item", id="add-item-button", n_clicks=0, color="info")
                            ],
                            className="vstack gap-2",
                        ),
                        html.Hr(),
                        html.H5("Parameters"),
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroupText("Review Cycle (Days):", style={"margin-bottom": "8px"}),
                                dbc.InputGroupText("R-Cost ($):", style={"margin-bottom": "8px"}),
                                dbc.InputGroupText("K-Cost (%):", style={"margin-bottom": "8px"})
                            ], width=8, style={"padding-right": "2px"}),
                            dbc.Col([
                                dbc.Input(id="review-cycle-input", type="number", value=14, style={"margin-bottom": "8px"}),
                                dbc.Input(id="r-cost-input", type="number", value=8, style={"margin-bottom": "8px"}),
                                dbc.Input(id="k-cost-input", type="number", value=0.18 * 100, style={"margin-bottom": "8px"})
                            ], width=4, style={"padding-left": "2px"})
                        ], style={"margin-top": "8px"}),
                        dbc.Button("Update Parameters", id="update-button", n_clicks=0, color="primary", style={"width": "100%"}),
                        dbc.Label("Simulation Speed (ms):", style={"margin-top": "10px"}),
                        dcc.Slider(id="sim-speed-slider", min=150, max=2000, value=600, marks={150: "Fast", 2000: "Slow"}, step=50),
                        html.Hr(),
                        html.Div(id="day-display", children=f"Day: {1}", style={"margin-top": "20px"}),
                        html.Div(id="sim-status", children="Status: Paused", style={"margin-top": "20px"}),
                        html.Div(id='store-output', style={"margin-top": "20px"}),
                        dcc.Store(id='user-data-store', storage_type='local'),
                        dcc.Store(id='page-load', data=0)
                    ],
                    width=2,
                ),
                # Right column for the graph
                dbc.Col(
                    [
                        dcc.Graph(id="inventory-graph", style={"height": "800px"}, figure=initial_graph()),
                    ],
                    width=10,
                ),
            ]
        ),
        dcc.Interval(id="interval-component", interval=600, n_intervals=0, max_intervals=-1, disabled=True),
        Modal(
            [
                ModalHeader("Add New Item"),
                ModalBody(
                    [
                        dbc.InputGroup([
                                dbc.InputGroupText("Usage Rate:"),
                                dbc.Input(id="usage-rate-input", type="number", value=0)
                            ], style={"margin-bottom": "10px"}
                        ),
                        dbc.InputGroup([
                                dbc.InputGroupText("Lead Time:"),
                                dbc.Input(id="lead-time-input", type="number", value=0)
                            ], style={"margin-bottom": "10px"}
                        ),
                        dbc.InputGroup([
                                dbc.InputGroupText("Item Cost:"),
                                dbc.Input(id="item-cost-input", type="number", value=0)
                            ], style={"margin-bottom": "10px"}
                        ),
                        dbc.InputGroup([
                                dbc.InputGroupText("Initial PNA:"),
                                dbc.Input(id="pna-input", type="number", value=0)
                            ], style={"margin-bottom": "10px"}
                        ),
                        dbc.InputGroup([
                                dbc.InputGroupText("Safety Allowance (%):"),
                                dbc.Input(id="safety-allowance-input", type="number", value=50)
                            ], style={"margin-bottom": "10px"}
                        ),
                        dbc.InputGroup([
                                dbc.InputGroupText("Standard Pack:"),
                                dbc.Input(id="standard-pack-input", type="number", value=0)
                            ], style={"margin-bottom": "10px"}
                        ),
                                dbc.InputGroup([
                                dbc.InputGroupText("Hits Per Month:"),
                                dbc.Input(id="hits-per-month-input", type="number", value=0)
                            ], style={"margin-bottom": "10px"}
                        )
                    ]
                ),
                ModalFooter([
                    dbc.Button("Randomize", id="randomize-button", color="secondary"),
                    dbc.Button("Add item", id="submit-item-button", color="primary")
                ]),
            ],
            id="add-item-modal",
            is_open=False,  # by default, the modal is not open
        ),
        Modal(
            [
                ModalHeader("Place Custom Order"),
                ModalBody(
                    [
                        dbc.Row([  # This is the header row
                            dbc.Col(html.Strong("Item Index"), width=2),
                            dbc.Col(html.Strong("PNA")),
                            dbc.Col(html.Strong("Usage Rate")),
                            dbc.Col(html.Strong("Lead Time")),
                            dbc.Col(html.Strong("OP")),
                            dbc.Col(html.Strong("LP")),
                            dbc.Col(html.Strong("OQ")),
                            dbc.Col(html.Strong("Order Quantity"), width=2)
                        ]),
                        html.Div(id="custom-order-items-div")  # This is where the items will be populated
                    ]
                ),
                ModalFooter([
                    dbc.Button("Cancel", id="cancel-custom-order-button", color="secondary"),
                    dbc.Button("Place Order", id="place-order-button", color="primary")
                ]),
            ],
            id="place-custom-order-modal",
            is_open=False,
            size="lg",
            style={"maxHeight": "70vh", "overflowY": "auto"}  # To ensure scrolling if there are many items
        )
    ],
    fluid=True,
)

# Callback to set a new UUID for each session
@app.callback(
    Output('user-data-store', 'data'),
    Input('user-data-store', 'modified_timestamp'),  # This is an internal property of dcc.Store
    State('user-data-store', 'data'),
    prevent_initial_call=True  # Prevent running on app load
)
def set_uuid(ts, data):
    if ts is None or (data is not None and 'uuid' in data):
        raise dash.exceptions.PreventUpdate
    if not data or 'uuid' not in data or not data['uuid']:
        return {'uuid': str(uuid.uuid4())}
    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output("add-item-modal", "is_open"), Output('store-output', 'children')],
    [Input("add-item-button", "n_clicks"), Input("submit-item-button", "n_clicks")],
    [State("add-item-modal", "is_open"),
     State('user-data-store', 'data')],
)
def toggle_modal(add_clicks, submit_clicks, is_open, data):
    if add_clicks or submit_clicks:
        return not is_open, str(data)
    return is_open, str(data)

@app.callback(
    [Output("place-custom-order-modal", "is_open"),
     Output("custom-order-items-div", "children")],
    [Input("place-custom-order-button", "n_clicks"),
     Input("cancel-custom-order-button", "n_clicks"),
     Input("place-order-button", "n_clicks")],
    [State("user-data-store", "data")]
)
def populate_custom_order_items(n, cancel_clicks, place_order_clicks, client_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Retrieve user's data based on uuid
    uuid = client_data['uuid']
    current_data = get_user_data(uuid)
    items = current_data.get('items', [])

    if not current_data or 'items' not in current_data or not current_data['items']:
        raise dash.exceptions.PreventUpdate

    # Build rows for each item
    item_rows = []
    for index, item in enumerate(items):
        row = dbc.Row([
            dbc.Col(html.P(str(index+1)), width=2),
            dbc.Col(html.P(str(round(item['pna'])))),
            dbc.Col(html.P(str(round(item['usage_rate'])))),
            dbc.Col(html.P(str(round(item['lead_time'])))),
            dbc.Col(html.P(str(round(item['op'])))),
            dbc.Col(html.P(str(round(item['lp'])))),
            dbc.Col(html.P(str(round(item['oq'])))),
            dbc.Col(dbc.Input(value=round(item['soq']), id={"type": "order-quantity", "index": index}))
        ])
        item_rows.append(row)

    if button_id == "place-custom-order-button":  # Corrected button id
        return True, item_rows
    elif button_id in ["cancel-custom-order-button", "place-order-button"]:
        return False, item_rows
    return dash.no_update, item_rows

@app.callback(
    [Output('day-display', 'children'),
     Output('inventory-graph', 'figure'),
     Output('interval-component', 'disabled'),
     Output('start-button', 'children'),
     Output('start-button', 'color'),
     Output('sim-status', 'children'),
     Output('page-load', 'data'),
     Output('interval-component', 'interval')],
    [Input('interval-component', 'n_intervals'),
     Input('start-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input("submit-item-button", "n_clicks"),
     Input("update-button", "n_clicks"),
     Input("po-button", 'n_clicks'),
     Input("place-order-button", 'n_clicks'),
     Input({"type": "order-quantity", "index": ALL}, "value")],
    [State('interval-component', 'disabled'),
     State('user-data-store', 'data'),
     State('page-load', 'data'),
     State("usage-rate-input", "value"),
     State("lead-time-input", "value"),
     State("item-cost-input", "value"),
     State("pna-input", "value"),
     State("safety-allowance-input", "value"),
     State("standard-pack-input", "value"),
     State("hits-per-month-input", "value"),
     State("review-cycle-input", "value"),
     State("r-cost-input", "value"),
     State("k-cost-input", "value"),
     State("sim-speed-slider", "value")]
)
def combined_callback(n_intervals, start_clicks, reset_clicks, submit_item_clicks, update_button_clicks, po_button, place_order_button, order_quantities, is_disabled, client_data, page_load, usage_rate, lead_time, item_cost, pna, safety_allowance, standard_pack, hits_per_month, review_cycle, r_cost, k_cost, sim_speed):

    ctx = dash.callback_context

    uuid = client_data['uuid']
    current_data = get_user_data(uuid)

    print(uuid, " data: ",current_data)

    if page_load == 0:
        day_count = current_data.get('day', 1)
        day_display = f"Day: {day_count}"  # Or whatever your default should be
        fig = initial_graph(current_data)
        return day_display, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 1, sim_speed  # Set page load to 1
    
    # If interval-component was triggered
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'interval-component.n_intervals':
        # If simulation is not initialized, don't proceed
        if not current_data.get('is_initialized', False):
            raise dash.exceptions.PreventUpdate
        if not current_data or 'items' not in current_data or not current_data['items']:
            raise dash.exceptions.PreventUpdate
        day_count = current_data.get('day', 1)
        current_data['day'] = day_count + 1
        futures = [executor.submit(process_item, item) for item in current_data['items']]
        current_data['items'] = [future.result() for future in futures]
        day_display = f"Day: {day_count}"
        set_user_data(uuid=uuid, data=current_data)
        fig = update_graph_based_on_items(current_data['items'],current_data['global_settings'])
        print(current_data)
        return day_display, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, sim_speed

    # If the start-button was clicked
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'start-button.n_clicks':
        if is_disabled:
            current_data['is_initialized'] = True  # Set to True when starting the simulation
            set_user_data(uuid=uuid, data=current_data)
            return dash.no_update, dash.no_update, False, "Pause Simulation", "warning", "Status: Running", dash.no_update, sim_speed
        else:
            return dash.no_update, dash.no_update, True, "Resume Simulation", "success", "Status: Paused", dash.no_update, sim_speed
        
    # If the reset-button was clicked
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-button.n_clicks':
        # Reset the user data to defaults
        default_data = {
            'global_settings': {'r_cycle': 14, 'r_cost': 8, 'k_cost': 0.18},
            'items': [],
            'day': 1,
            'is_initialized': False
        }
        current_data['day'] = 1
        day_count = current_data.get('day', 1)  # Safely get the 'day'
        day_display = f"Day: {day_count}"
        fig = px.scatter(title="Inventory Simulation")
        set_user_data(uuid=uuid, data=default_data)
        return day_display, fig, True, "Start Simulation", "success", "Status: Paused", dash.no_update, sim_speed
        
    # Handle adding a new item
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'submit-item-button.n_clicks':
        new_item = create_inventory_item(usage_rate, lead_time, item_cost, pna, (safety_allowance/100), standard_pack,current_data['global_settings'], hits_per_month)
        current_data['items'].append(new_item)
        set_user_data(uuid=uuid, data=current_data)
        fig = update_graph_based_on_items(current_data['items'],current_data['global_settings'])
        return dash.no_update, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, sim_speed
    
    # If the update-button was clicked
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'update-button.n_clicks':
        if not current_data or 'global_settings' not in current_data or not current_data['global_settings']:
            raise dash.exceptions.PreventUpdate
        # Update global settings
        current_data['global_settings'] = update_global_settings(current_data['global_settings'], review_cycle, r_cost, k_cost/100)
        set_user_data(uuid=uuid, data=current_data)
        if not current_data or 'items' not in current_data or not current_data['items']:
            raise dash.exceptions.PreventUpdate
        # Update all items based on the new global settings
        for item in current_data['items']:
            item = update_gs_related_values(item, current_data['global_settings'])
        set_user_data(uuid=uuid, data=current_data)
        fig = update_graph_based_on_items(current_data['items'],current_data['global_settings'])
        return dash.no_update, fig, is_disabled, dash.no_update, dash.no_update, dash.no_update, dash.no_update, sim_speed
    
    # If the po-button was clicked
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'po-button.n_clicks':
        if not current_data or 'items' not in current_data or not current_data['items']:
            raise dash.exceptions.PreventUpdate
        # Increase the pna of each item by its soq
        for i, item in enumerate(current_data['items']):
            item['pna'] += item['soq']
            current_data['items'][i] = update_pna_related_values(item)
        set_user_data(uuid=uuid, data=current_data)
        fig = update_graph_based_on_items(current_data['items'], current_data['global_settings'])
        return dash.no_update, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # If the custom po-button was clicked
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == "place-order-button.n_clicks":
        if not current_data or 'items' not in current_data or not current_data['items']:
            raise dash.exceptions.PreventUpdate
        for index, item in enumerate(current_data['items']):
            custom_order_quantity = float(order_quantities[index]) if order_quantities[index] is not None else 0
            item['pna'] += custom_order_quantity
            item = update_pna_related_values(item)
        set_user_data(uuid=uuid, data=current_data)
        #Not sure why graph is not updating after custom po placed
        fig = update_graph_based_on_items(current_data['items'], current_data['global_settings'])
        return dash.no_update, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('review-cycle-input', 'value'),
     Output('r-cost-input', 'value'),
     Output('k-cost-input', 'value')],
    [Input('update-button', 'n_clicks')],
    [State('user-data-store', 'data')]
)
def update_input_values_from_data_store(n_clicks, data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    uuid = data['uuid']
    current_data = get_user_data(uuid)
    # Extract the global settings
    global_settings = current_data.get('global_settings', {})
    return global_settings.get('r_cycle', 14), global_settings.get('r_cost', 8), global_settings.get('k_cost', 0.18) * 100  # Return defaults if not found

@app.callback(
    [Output("usage-rate-input", "value"),
     Output("lead-time-input", "value"),
     Output("item-cost-input", "value"),
     Output("pna-input", "value"),
     Output("safety-allowance-input", "value"),
     Output("standard-pack-input", "value"),
     Output("hits-per-month-input", "value")],
    [Input("randomize-button", "n_clicks")]
)
def randomize_item_values(n):
    if not n:
        raise dash.exceptions.PreventUpdate

    # Generate random values within specified ranges
    random_usage_rate = random.randint(1, 100)
    random_lead_time = abs(round(np.random.normal(30,30)))
    while random_lead_time < 7:
        random_lead_time = abs(round(np.random.normal(30, 30)))
    random_item_cost = abs(round(np.random.normal(100,100)))
    while random_lead_time < 7:
        random_lead_time = abs(round(np.random.normal(30, 30)))
    random_safety_allowance = 50 if random_lead_time < 60 else round(3000 / random_lead_time)
    random_standard_pack = np.random.choice([1, 5, 10, 20, 25, 40, 50], p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    random_pna = round(random_usage_rate * (random_lead_time / 30) + ((random_usage_rate * random_lead_time * random_safety_allowance/100) / 30) + random_standard_pack)
    random_hits_per_month = np.random.poisson(5)

    return random_usage_rate, random_lead_time, random_item_cost, random_pna, random_safety_allowance, random_standard_pack, random_hits_per_month

def save_data():
    with open('data/user_data.json', 'w') as f:
        json.dump(user_data_store, f)

#use curl -X POST http://127.0.0.1:8050/shutdown to save server data
@server.route('/shutdown', methods=['POST'])
def shutdown():
    save_data()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)
import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html, Input, Output, State, callback, ALL, dash_table
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
import base64
import datetime
import io

executor = ThreadPoolExecutor(max_workers=4)  # Define the number of worker threads

load_figure_template(["darkly"])

server = Flask(__name__)
# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], server=server)

# server-side data store
user_data_store = {}

# Default data structure
def get_default_data():
    return {
    'global_settings': {'r_cycle': 14, 'r_cost': 8, 'k_cost': 0.18},
    'items': [],
    'day': 1,
    'is_initialized': False
    }

def get_user_data(uuid):
    # Check if uuid is in user_data_store
    if uuid not in user_data_store:
        # If not, create a new entry with default data
        user_data_store[uuid] = get_default_data()
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

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            print(df)
        else:
            return dbc.Alert("The Input file was of the wrong type.", color="warning", duration=4000)
        
        # Data validation steps
        # Check for the correct number of columns
        if df.shape[1] != 7:
            return dbc.Alert(f"Incorrect number of columns. Expected 7, found {df.shape[1]}.", color="warning", duration=4000)

        # Check for any empty cells
        if df.isnull().values.any():
            return dbc.Alert("One or more cells are empty.", color="warning", duration=4000)

        # Check all values are greater than 0
        if (df <= 0).any().any():
            return dbc.Alert("All values must be greater than 0.", color="warning", duration=4000)

        # Data type validation - checking if all columns contain numbers
        try:
            df = df.apply(pd.to_numeric, errors='coerce')
            if df.isnull().values.any():
                return dbc.Alert("All columns must contain only numbers.", color="warning", duration=4000)
        except Exception as e:
            print(e)
            return dbc.Alert("All columns must contain only numbers.", color="danger", duration=4000)

    except Exception as e:
        print(e)
        return dbc.Alert("There was an error processing this file.", color="danger", duration=4000)

    return dbc.Card(
        dbc.CardBody([
            html.H5(filename, className="card-title"),
            html.H6(datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S'), className="card-subtitle"),
            html.Br(),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'color': 'white'
                },
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white',
                    'border': '1px solid #444'
                },
                style_data={
                    'border': '1px solid #444'
                }
            ),
            html.Hr(className="my-4"),
        ]),
        className="mb-3"
    )

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
    mask = (df['pro_pna_days_frm_op'] != df['pna_days_frm_op']) & (df['pna'] <= df['lp'])
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
            children=[dbc.NavItem(dbc.NavLink("Inventory Management Simulator", href="#")),],
            brand="CEEUS",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Row([
                dbc.Col(
                    [
                        dbc.Card([
                            dbc.CardHeader("Controls"),
                            dbc.CardBody(
                                [
                                    dbc.Button("Start/Pause Simulation", id="start-button", n_clicks=0, color="success"),
                                    dbc.Button("Reset Simulation", id="reset-button", n_clicks=0),
                                    dbc.Button("Place Purchase Order", id="po-button", n_clicks=0),
                                    dbc.Button("Place Custom Order", id="place-custom-order-button", n_clicks=0, color="warning"),
                                    dbc.Button("Add Item", id="add-item-button", n_clicks=0, color="info")
                                ],
                                className="vstack gap-2"
                            )
                        ], style={"margin-top": "10px"}),
                        dbc.Card([
                            dbc.CardHeader("Parameters"),
                            dbc.CardBody(
                                [
                                    dbc.InputGroup([dbc.InputGroupText("Review Cycle (Days):"), dbc.Input(id="review-cycle-input", type="number", value=14)]),
                                    dbc.InputGroup([dbc.InputGroupText("R-Cost ($):"), dbc.Input(id="r-cost-input", type="number", value=8)]),
                                    dbc.InputGroup([dbc.InputGroupText("K-Cost (%):"), dbc.Input(id="k-cost-input", type="number", value=0.18 * 100)]),
                                    dbc.Row(id = "update-params-conf"),
                                    dbc.Button("Update Parameters", id="update-params-button", n_clicks=0, color="primary", className="w-100 mb-2"),
                                    dbc.Label("Simulation Speed (ms):", className="d-block mb-2"),
                                    dcc.Slider(id="sim-speed-slider", min=150, max=2000, value=600, marks={150: "Fast", 2000: "Slow"}, step=50)
                                ],
                                className="vstack gap-2"
                            )
                        ], style={"margin-top": "10px"}),
                        dbc.Card([
                            dbc.CardBody(
                                [
                                        html.Div(id="day-display", children=f"Day: {1}"),
                                        html.Div(id="sim-status", children="Status: Paused"),
                                        html.Div(id='store-output')
                                ],
                                className="vstack gap-2"
                            )
                        ], style={"margin-top": "10px"}),
                        dcc.Store(id='user-data-store', storage_type='local'),
                        dcc.Store(id='page-load', data=0),
                        dcc.Store(id='page-load-2', data=0)
                    ],
                    width=2,
                ),
                # Right column for the graph
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Inventory Graph"),
                        dbc.CardBody([
                            dcc.Graph(id="inventory-graph", style={"height": '80vh'}, figure=initial_graph())
                        ])
                    ])
                ], width=10, style={"margin-top": "10px"})
        ]),
        dcc.Interval(id="interval-component", interval=600, n_intervals=0, max_intervals=-1, disabled=True),
        Modal(
            [
                ModalHeader("Add New Item"),
                ModalBody([
                    dbc.Card([
                        dbc.CardHeader("Manually Add Item"),
                        dbc.CardBody(
                            [
                                dbc.InputGroup([dbc.InputGroupText("Usage Rate:"), dbc.Input(id="usage-rate-input", type="number", value=0)], style={"margin-bottom": "10px"}),
                                dbc.InputGroup([dbc.InputGroupText("Lead Time:"), dbc.Input(id="lead-time-input", type="number", value=0)], style={"margin-bottom": "10px"}),
                                dbc.InputGroup([dbc.InputGroupText("Item Cost:"), dbc.Input(id="item-cost-input", type="number", value=0)], style={"margin-bottom": "10px"}),
                                dbc.InputGroup([dbc.InputGroupText("Initial PNA:"), dbc.Input(id="pna-input", type="number", value=0)], style={"margin-bottom": "10px"}),
                                dbc.InputGroup([dbc.InputGroupText("Safety Allowance (%):"), dbc.Input(id="safety-allowance-input", type="number", value=50)], style={"margin-bottom": "10px"}),
                                dbc.InputGroup([dbc.InputGroupText("Standard Pack:"), dbc.Input(id="standard-pack-input", type="number", value=0)], style={"margin-bottom": "10px"}),
                                dbc.InputGroup([dbc.InputGroupText("Hits Per Month:"), dbc.Input(id="hits-per-month-input", type="number", value=0)], style={"margin-bottom": "10px"}),
                                dbc.Row(id = "add-item-error"),
                                dbc.Button("Randomize", id="randomize-button", color="secondary", className="me-md-2"),
                                dbc.Button("Add item", id="submit-item-button", color="primary")
                            ],
                            className="end"
                        )
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Upload Items"),
                        dbc.CardBody(
                            [
                                dcc.Upload(
                                    id='upload-item',
                                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                                    multiple=True # Allow multiple files to be uploaded
                                ),
                                html.Div(id='output-item-upload')
                            ]
                        )
                    ], style={"margin-top": "10px"}),
                ])
            ],
            id="add-item-modal",
            is_open=False,  # by default, the modal is not open
            style={'maxHeight': 'calc(95vh)', 'overflowY': 'auto'},
            size="lg"
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
            style={'maxHeight': 'calc(95vh)', 'overflowY': 'auto'},
            size="lg"
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
        raise PreventUpdate
    if not data or 'uuid' not in data or not data['uuid']:
        return {'uuid': str(uuid.uuid4())}
    else:
        raise PreventUpdate

@app.callback(
    [Output('day-display', 'children', allow_duplicate=True),
     Output('inventory-graph', 'figure', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals')],
    [State('user-data-store', 'data')],
    prevent_initial_call = True
)
def update_on_interval(n_intervals, client_data):
    if not n_intervals or not client_data:
        raise PreventUpdate

    uuid = client_data.get('uuid')
    current_data = get_user_data(uuid)

    if not current_data.get('is_initialized', False):
        # If simulation is not initialized, don't proceed
        raise PreventUpdate

    if not current_data or 'items' not in current_data or not current_data['items']:
        # If there are no items, don't proceed
        raise PreventUpdate

    # Update the day count
    day_count = current_data.get('day', 1) + 1
    current_data['day'] = day_count

    # Process each item
    futures = [executor.submit(process_item, item) for item in current_data['items']]
    current_data['items'] = [future.result() for future in futures]

    # Update the graph based on the processed items
    fig = update_graph_based_on_items(current_data['items'], current_data['global_settings'])

    # Save updated data back to user-data-store (this will depend on your implementation of set_user_data)
    set_user_data(uuid, current_data)

    # Construct the day display string
    day_display = f"Day: {day_count}"

    return day_display, fig

@app.callback(
    [Output('sim-status', 'children', allow_duplicate=True),
     Output('interval-component', 'disabled', allow_duplicate=True),
     Output('start-button', 'children'),
     Output('start-button', 'color')],
    [Input('start-button', 'n_clicks')],
    [State('user-data-store', 'data'),
     State('interval-component', 'disabled')],
    prevent_initial_call = True
)
def toggle_simulation(n_clicks, client_data, is_disabled):
    if not n_clicks or not client_data:
        raise PreventUpdate

    uuid = client_data['uuid']
    current_data = get_user_data(uuid)

    # Toggle the simulation status based on whether it is currently paused or running.
    if is_disabled:
        # If the simulation is currently disabled/paused, start it.
        current_data['is_initialized'] = True
        set_user_data(uuid, current_data)
        return "Status: Running", False, "Pause Simulation", "warning"
    else:
        # If the simulation is running, pause it.
        return "Status: Paused", True, "Resume Simulation", "success"

@app.callback(
    [Output('day-display', 'children', allow_duplicate=True),
     Output('inventory-graph', 'figure', allow_duplicate=True),
     Output('sim-status', 'children', allow_duplicate=True),
     Output('user-data-store', 'data', allow_duplicate=True)],  # To update the stored data
    [Input('reset-button', 'n_clicks')],
    [State('user-data-store', 'data')],
    prevent_initial_call=True
)
def reset_simulation(n_clicks, client_data):
    if n_clicks:
        # Reset the day count and other necessary parts of `client_data` to their defaults
        default_data = get_default_data()
        default_data['day'] = 1

        # Prepare the initial state of the graph
        fig = initial_graph()

        # Return the updated outputs
        return f"Day: {default_data['day']}", fig, "Status: Paused", default_data

    raise PreventUpdate

@app.callback(
    [Output("add-item-modal", "is_open"),
     Output('add-item-error', 'children'),
     Output('inventory-graph', 'figure', allow_duplicate=True)],
    [Input("add-item-button", "n_clicks"),
     Input("submit-item-button", "n_clicks")],
    [State("add-item-modal", "is_open"),
     State("usage-rate-input", "value"),
     State("lead-time-input", "value"),
     State("item-cost-input", "value"),
     State("pna-input", "value"),
     State("safety-allowance-input", "value"),
     State("standard-pack-input", "value"),
     State("hits-per-month-input", "value"),
     State('user-data-store', 'data')],
    prevent_initial_call=True
)
def handle_add_item_and_update_graph(add_clicks, submit_clicks, is_open, usage_rate, lead_time, item_cost, pna, safety_allowance, standard_pack, hits_per_month, client_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, dash.no_update, dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "add-item-button":
        # Toggle modal without validating inputs
        return not is_open, dash.no_update, dash.no_update

    if button_id == "submit-item-button":
        # Validate inputs
        all_inputs = [usage_rate, lead_time, item_cost, pna, safety_allowance, standard_pack, hits_per_month]
        if any(i is None for i in all_inputs):
            return is_open, dbc.Alert("All fields must be filled out!", color="warning", duration=4000), dash.no_update

        if any(i <= 0 for i in [usage_rate, lead_time, item_cost, safety_allowance, standard_pack, hits_per_month]):
            return is_open, dbc.Alert("All item parameters except PNA must be greater than 0!", color="warning", duration=4000), dash.no_update

        # Add the new item
        uuid = client_data['uuid']
        current_data = get_user_data(uuid)
        new_item = create_inventory_item(usage_rate, lead_time, item_cost, pna, safety_allowance/100, standard_pack, current_data['global_settings'], hits_per_month)
        current_data['items'].append(new_item)
        set_user_data(uuid=uuid, data=current_data)

        # Update the graph
        fig = update_graph_based_on_items(current_data['items'], current_data['global_settings'])

        # Close the modal and clear any error message, then return the updated graph
        return False, None, fig

    raise PreventUpdate

@app.callback(
    [Output('update-params-conf', 'children'),
     Output('inventory-graph', 'figure', allow_duplicate=True)],  # Assuming this is the ID of your graph
    [Input('update-params-button', 'n_clicks')],
    [State('review-cycle-input', 'value'),
     State('r-cost-input', 'value'),
     State('k-cost-input', 'value'),
     State('user-data-store', 'data')],
    prevent_initial_call=True
)
def update_parameters(n_clicks, review_cycle, r_cost, k_cost, client_data):
    if not n_clicks:
        raise PreventUpdate

    # Input validation
    if review_cycle is None or r_cost is None or k_cost is None:
        return dbc.Alert("Please fill out all parameters!", color="warning", duration=4000), dash.no_update

    if review_cycle <= 0 or r_cost <= 0 or k_cost <= 0:
        return dbc.Alert("All parameters must be greater than 0!", color="danger", duration=4000), dash.no_update

    # Extract the UUID from client_data and get the current data
    uuid = client_data.get('uuid')
    if not uuid:
        raise PreventUpdate

    current_data = get_user_data(uuid)
    if 'global_settings' not in current_data:
        return dbc.Alert("Global settings not found.", color="danger", duration=4000), dash.no_update

    # Update global settings with new values
    current_data['global_settings'] = update_global_settings(
        current_data['global_settings'],
        review_cycle,
        r_cost,
        k_cost / 100  # Assuming k_cost was provided as a percentage and needs to be converted to a decimal
    )

    # Save the updated global settings back to the user's data store
    set_user_data(uuid, current_data)

    # Update all items based on the new global settings
    if 'items' in current_data:
        for item in current_data['items']:
            item = update_gs_related_values(item, current_data['global_settings'])

        set_user_data(uuid, current_data)
        # Now update the graph based on the updated items
        fig = update_graph_based_on_items(current_data['items'], current_data['global_settings'])
    else:
        fig = dash.no_update  # Or set to an empty figure if no items exist

    # Provide user feedback
    return dbc.Alert("Parameters updated successfully!", color="success", duration=4000), fig


@app.callback(
    Output('inventory-graph', 'figure', allow_duplicate=True),
    [Input("po-button", "n_clicks")],
    [State('user-data-store', 'data')],
    prevent_initial_call=True
)
def handle_purchase_order(n_clicks, client_data):
    if not n_clicks:
        raise PreventUpdate
    
    # Extract the UUID from client_data and get the current data
    uuid = client_data.get('uuid')
    if not uuid:
        raise PreventUpdate

    current_data = get_user_data(uuid)
    if not current_data or 'items' not in current_data or not current_data['items']:
        raise PreventUpdate

    # Increase the PNA of each item by its SOQ and update item details accordingly
    for i, item in enumerate(current_data['items']):
        item['pna'] += item['soq']
        current_data['items'][i] = update_pna_related_values(item)
    
    # Save the updated item data back to the user's data store
    set_user_data(uuid, current_data)

    # Now update the graph based on the updated items
    fig = update_graph_based_on_items(current_data['items'], current_data['global_settings'])
    
    return fig

@app.callback(
    [Output('inventory-graph', 'figure', allow_duplicate=True),
     Output('custom-order-items-div', 'children'),
     Output("place-custom-order-modal", "is_open"),
     Output('sim-status', 'children', allow_duplicate=True),
     Output('interval-component', 'disabled', allow_duplicate=True)],
    [Input("place-custom-order-button", "n_clicks"),
     Input("cancel-custom-order-button", "n_clicks"),
     Input("place-order-button", "n_clicks")],
    [State({"type": "order-quantity", "index": ALL}, "value"),
     State('user-data-store', 'data'),
     State('interval-component', 'disabled')],
    prevent_initial_call=True
)
def handle_custom_order_and_modal_actions(place_order_clicks, cancel_clicks, submit_clicks, order_quantities, client_data, is_interval_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        # No button clicked yet, nothing to update
        return dash.no_update, dash.no_update, False, dash.no_update, is_interval_disabled

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    uuid = client_data.get('uuid')

    # If there's no UUID, we cannot proceed
    if not uuid:
        return dash.no_update, dash.no_update, False, dash.no_update, is_interval_disabled

    current_data = get_user_data(uuid)
    items = current_data.get('items', [])
    item_rows = [build_row(index, item) for index, item in enumerate(items)]

    if button_id == "place-custom-order-button":
        # Open the modal and pause the simulation
        return dash.no_update, item_rows, True, "Status: Paused", True

    elif button_id == "cancel-custom-order-button":
        # Close the modal and resume simulation if it was paused
        return dash.no_update, dash.no_update, False, dash.no_update, dash.no_update

    elif button_id == "place-order-button":
        if not items:
            # If there are no items, keep the modal open
            return dash.no_update, dbc.Alert("No items available for ordering.", color="warning"), True, dash.no_update, True

        error_message = None
        for index, quantity in enumerate(order_quantities):
            if quantity is None or float(quantity) < 0:
                error_message = f"Order quantity for item {index + 1} cannot be less than zero."
                item_rows = update_custom_order_items_div(items, order_quantities, error_index=index)
                return dash.no_update, item_rows, True, dash.no_update, True

        for index, quantity in enumerate(order_quantities):
            custom_order_quantity = float(quantity)
            item = items[index]
            item['pna'] += custom_order_quantity
            items[index] = update_pna_related_values(item)

        set_user_data(uuid, current_data)
        fig = update_graph_based_on_items(items, current_data['global_settings'])

        # Close the modal and resume simulation
        return fig, item_rows, False, "Status: Running", False

    # If the callback was not triggered by any of the above buttons
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def update_custom_order_items_div(items, order_quantities, error_index=None):
    item_rows = []
    for index, item in enumerate(items):
        row = build_row(index, item)
        if index == error_index:
            # Add an error message or styling to the row
            row.children.append(dbc.Alert("Invalid quantity!", color="danger", duration=4000))
        item_rows.append(row)
    return item_rows

def build_row(index, item):
    # This helper function builds a single row of item data
    return dbc.Row([
        dbc.Col(html.P(str(index+1)), width=2),
        dbc.Col(html.P(str(round(item['pna'])))),
        dbc.Col(html.P(str(round(item['usage_rate'])))),
        dbc.Col(html.P(str(round(item['lead_time'])))),
        dbc.Col(html.P(str(round(item['op'])))),
        dbc.Col(html.P(str(round(item['lp'])))),
        dbc.Col(html.P(str(round(item['oq'])))),
        dbc.Col(dbc.Input(value=round(item['soq']), id={"type": "order-quantity", "index": index}))
    ])

@app.callback(
    [Output('review-cycle-input', 'value'),
     Output('r-cost-input', 'value'),
     Output('k-cost-input', 'value'),
     Output('page-load-2', 'data'),],
    [Input('update-params-button', 'n_clicks')],
    [State('user-data-store', 'data'),
     State('page-load-2', 'data')]
)
def update_input_values_from_data_store(n_clicks, data, page_load):
    if page_load == 0:
        uuid = data['uuid']
        current_data = get_user_data(uuid)
        global_settings = current_data.get('global_settings', {})
        return global_settings.get('r_cycle', 14), global_settings.get('r_cost', 8), global_settings.get('k_cost', 0.18) * 100, 1  # Return defaults if not found
    if not n_clicks:
        raise PreventUpdate
    uuid = data['uuid']
    current_data = get_user_data(uuid)
    # Extract the global settings
    global_settings = current_data.get('global_settings', {})
    return global_settings.get('r_cycle', 14), global_settings.get('r_cost', 8), global_settings.get('k_cost', 0.18) * 100, dash.no_update  # Return defaults if not found

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
        raise PreventUpdate

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

@app.callback(
    [Output('day-display', 'children'),
     Output('inventory-graph', 'figure'),
     Output('page-load', 'data')],  # Assuming 'page-load' is a dcc.Store component
    [Input('page-load', 'data')],   # Triggers when 'page-load' data changes
    [State('user-data-store', 'data')]
)
def handle_page_load(page_load, client_data):
    if page_load == 0:
        # Assuming get_user_data function fetches the current user's data
        current_data = get_user_data(client_data['uuid']) if client_data and 'uuid' in client_data else {'day': 1}
        day_count = current_data.get('day', 1)
        fig = initial_graph(current_data)  # Assuming initial_graph is a function that creates the initial graph figure

        # Update 'page-load' to 1 so this initialization doesn't run again
        return f"Day: {day_count}", fig, 1
    else:
        # If 'page-load' is not 0, do not update anything
        raise PreventUpdate

@callback(Output('output-item-upload', 'children'),
    Input('upload-item', 'contents'),
    State('upload-item', 'filename'),
    State('upload-item', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

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
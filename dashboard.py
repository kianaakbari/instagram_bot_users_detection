import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import io
from sklearn.externals import joblib
import requests
import json
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
import datetime
import numpy as np

colors = ['lightslategray',] * 5
colors[1] = 'crimson'
bgcolor = '#ffece4';

#load data
dataset = pd.read_csv("data/dataset.csv")
cleaned_dataset = dataset.applymap(lambda x: np.nan if x == 999999999999 else x)
cleaned_dataset = cleaned_dataset.applymap(lambda x: np.nan if x == 1000000000000 else x)
cleaned_dataset = cleaned_dataset.applymap(lambda x: 1 if x == True else x)
cleaned_dataset = cleaned_dataset.applymap(lambda x: 0 if x == False else x)
columns = dataset.columns
model = joblib.load('data/RandomForestModel.pkl')
database = pd.DataFrame()
database_rows = []
try:
    database = pd.read_csv("data/database.csv")
    database_rows = database.to_dict('records')
except:
    pass

features = ['followers', 'followings', 'has_biography', 'is_private', 'has_name',
       'posts', 'has_profile_picture', 'bio_lenght',
       'following_to_follower_ratio', 'following_to_post_ratio',
       'follower_to_post_ratio', 'name_only_alphabet',
       'LCSofNameAndUsername_to_name_ratio', 'number_in_username_ratio']
feature_importance = [ 0.07573864, 0.1799273 , 0.01377027, 0.08454097, 0.00086077,
       0.04679525, 0.04519827, 0.02238473, 0.33663927, 0.15096226,
       0.02790126, 0.00136808, 0.00434537, 0.00956756]

#parameters
add_row_nclick = 0
temp_data = 0


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.Tabs(
                [
                    dbc.Tab(label="Query", tab_id="database_tab"),
                    dbc.Tab(label="Analytics", tab_id="model_tab"),
                    dbc.Tab(label="Home", tab_id="predict_tab"),
                ],
                id="nav_tabs",
                card=True,
                active_tab="model_tab",
            style={"marginBottom":"18px"})
        ),
    ],
    brand="Instagram Bot Accounts",
    brand_href="#",
    sticky="top",
)

bar_figure = {"data": [
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        "0",
                        "0",
                    ],
                    marker={
                        "color": "#97151c",
                        "line": {
                            "color": "#97151c",
                            "width": 1,
                        },
                    },
                    name="Calibre Index Fund",
                ),
            ],
            "layout": go.Layout(
                plot_bgcolor=bgcolor,
                paper_bgcolor=bgcolor,
                autosize=False,
                bargap=0.35,
                font={"family": "Raleway", "size": 15},
                height=257,
                hovermode="closest",
                # legend={
                #     "x": -0.0228945952895,
                #     "y": -0.189563896463,
                #     "orientation": "h",
                #     "yanchor": "top",
                # },
                margin={
                    "r": 0,
                    "t": 10,
                    "b": 50,
                    "l": 30,
                },
                # showlegend=True,
                title="",
                width=276,
                xaxis={
                    "autorange": True,
                    # "range": [-0.5, 4.5],
                    "showline": True,
                    "title": "",
                    "type": "category",
                },
                yaxis={
                    "autorange": True,
                    # "range": [0, 22.9789473684],
                    "showgrid": True,
                    "showline": True,
                    "title": "",
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

bar_figure_3 = {"data": [
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        0,
                        0,
                    ],
                    marker={
                        "color": "#97151c",
                        "line": {
                            "color": "#97151c",
                            "width": 1,
                        },
                    },
                    name="Calibre Index Fund",
                ),
            ],
            "layout": go.Layout(
                plot_bgcolor=bgcolor,
                paper_bgcolor=bgcolor,
                autosize=False,
                bargap=0.35,
                font={"family": "Raleway", "size": 15},
                height=350,
                hovermode="closest",
                # legend={
                #     "x": -0.0228945952895,
                #     "y": -0.189563896463,
                #     "orientation": "h",
                #     "yanchor": "top",
                # },
                margin={
                    "r": 0,
                    "t": 10,
                    "b": 50,
                    "l": 30,
                },
                # showlegend=True,
                title="",
                width=340,
                xaxis={
                    "autorange": True,
                    # "range": [-0.5, 4.5],
                    "showline": True,
                    "title": "",
                    "type": "category",
                },
                yaxis={
                    "autorange": True,
                    # "range": [0, 22.9789473684],
                    "showgrid": True,
                    "showline": True,
                    "title": "",
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

bar_figure_1 = {"data": [
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        0,
                        0,
                    ],
                    marker={
                        "color": "#97151c",
                        "line": {
                            "color": "#97151c",
                            "width": 1,
                        },
                    },
                    name="Calibre Index Fund",
                ),
            ],
            "layout": go.Layout(
                plot_bgcolor=bgcolor,
                paper_bgcolor=bgcolor,
                autosize=False,
                bargap=0.35,
                font={"family": "Raleway", "size": 15},
                height=350,
                hovermode="closest",
                # legend={
                #     "x": -0.0228945952895,
                #     "y": -0.189563896463,
                #     "orientation": "h",
                #     "yanchor": "top",
                # },
                margin={
                    "r": 0,
                    "t": 10,
                    "b": 50,
                    "l": 30,
                },
                # showlegend=True,
                title="",
                width=340,
                xaxis={
                    "autorange": True,
                    # "range": [-0.5, 4.5],
                    "showline": True,
                    "title": "",
                    "type": "category",
                },
                yaxis={
                    "autorange": True,
                    # "range": [0, 22.9789473684],
                    "showgrid": True,
                    "showline": True,
                    "title": "",
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

bar_figure_2 = {
            "data": [
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        0,
                        0,
                    ],
                    marker={
                        "color": "#97151c",
                        "line": {
                            "color": "#97151c",
                            "width": 1,
                        },
                    },
                    name="#0",
                ),
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        0,
                        0,
                    ],
                    marker={
                        "color": "#dddddd",
                        "line": {
                            "color": "#d4d2d2",
                            "width": 1,
                        },
                    },
                    name="#1",
                ),
            ],
            "layout": go.Layout(
                plot_bgcolor=bgcolor,
                paper_bgcolor=bgcolor,
                autosize=True,
                bargap=0.35,
                font={"family": "Raleway", "size": 15},
                height=350,
                hovermode="closest",
                # legend={
                #     "x": -0.0228945952895,
                #     "y": -0.189563896463,
                #     "orientation": "h",
                #     "yanchor": "top",
                # },
                margin={
                    "r": 0,
                    "t": 10,
                    "b": 50,
                    "l": 30,
                },
                # showlegend=True,
                title="",
                width=400,
                xaxis={
                    "autorange": True,
                    # "range": [-0.5, 4.5],
                    "showline": True,
                    "title": "",
                    "type": "category",
                },
                yaxis={
                    "autorange": True,
                    # "range": [0, 22.9789473684],
                    "showgrid": True,
                    "showline": True,
                    "title": "",
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }

predict_page = [
                dbc.Row(dbc.Col(html.H2("Input"), md=12), style={"marginTop":"5%"}),
                dbc.Row([
                    dbc.Col([
                        # html.H6("upload a csv or excel file of suspicious usernames:", style={'marginTop': '8%'}),
                        html.Div([
                            dbc.Row([
                                dbc.Col(
                                    [
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select File')
                                        ]),
                                        style={
                                            # 'width': '50%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            # 'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            # 'margin': '10px'
                                        },
                                    ),])], style={'marginTop': '4%'}, form=True),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([dcc.Input(value="no file chosen!", type="text", className="file-path validate")], className="file-path-wrapper")
                                ], md=6),
                                dbc.Col(md=2),
                                dbc.Col([
                                    html.Button(["upload", html.I("file_upload", className="material-icons right")], id="upload-btn", type="submit", className="btn waves-effect waves-light", style={"marginLeft":"20px", "width":"130px"}, n_clicks_timestamp=0)
                                ], md=4)
                            ],  style={'marginTop': '4%'}, form=True)], className="file-field input-field"),
                            # ,method="POST", action="http://localhost:8050/uploader", encType="multipart/form-data")]
                    ], md=5),
                    # html.H6("enter a single username:", style={'marginTop': '12%'}),
                    dbc.Col([],md=1),
                    dbc.Col([
                        html.H6("enter a single username:", style={"marginTop":"4%"}, id="h6"),
                        dbc.Row([
                            dbc.Col([
                                # html.Div([
                                dcc.Input(id="username", type="text", className="validate"),
                                # ], className="input-field"),
                            ], md=6),
                            dbc.Col(md=5),
                            dbc.Col([
                                html.Button([
                                    "Predict",
                                    html.I("send", className="material-icons right")
                                ], id="predict-btn", className="btn waves-effect waves-light", type="submit", style={"width":"127px", "marginTop":"-13px"}, n_clicks_timestamp=0),
                            ],md=1, className="file-field input-field")
                        ], style={'marginTop': '6%'}, form=True),
                    ],md=5,)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H2("Summarized Result"),
                        dbc.Row([
                            dbc.Col([
                                dash_table.DataTable(id='fake-table', data=[{} for _ in range(6)], columns=[{"name": i, "id": i} for i in ['username', 'is_fake']], fixed_rows={'headers': True, 'data': 0}, style_cell={'width': '100px', 'textAlign': 'left', 'backgroundColor': '#efffefd1'}, style_header={'backgroundColor':'#fafafa'},)
                            ], md=6),
                            dbc.Col([
                                dcc.Graph(
                                    id="bar_graph",
                                    figure=bar_figure,
                                    config={"displayModeBar": False},
                                ),
                            ], md=6)
                        ], form=True),
                    ], md=5),
                    dbc.Col(md=1),
                    dbc.Col([
                        html.H2("Detailed Result"),
                        # dcc.Graph(
                        #     figure={"data": [{"x": [1, 2, 3], "y": [1, 4, 9]}]}
                        # ),
                        ##efffefd1
                        dash_table.DataTable(id='upload-table', data=[{} for _ in range(6)], columns=[{"name": i, "id": i} for i in columns], fixed_rows={'headers': True, 'data': 0}, style_cell={'width': '150px', 'textAlign': 'left', 'backgroundColor': '#efffefd1'}, style_header={'backgroundColor':'#fafafa'}, style_table={'maxHeight': '227px', 'width': '587px'})
                    ], md=5),
                ])
            ]

feature_importance_table = html.Div([
    html.Td([
    html.Thead([
        html.Tr([
            html.Th("bootstrap",),
            html.Th("class_weight",),
            html.Th("criterion",),
            html.Th("max_depth",),
            html.Th("max_features",),
            html.Th("max_leaf_nodes",),
            html.Th("min_impurity_decrease",),
            html.Th("min_impurity_split",),
            html.Th("min_samples_split",),
            html.Th("min_weight_fraction_leaf",),
            html.Th("n_estimators",),
            html.Th("n_jobs",),
            html.Th("oob_score",),
            html.Th("random_state",),
            html.Th("verbose",),
            html.Th("warm_start",),
        ],)
    ], className=""),

    html.Tbody([
        html.Tr([
            html.Td("True",),
            html.Td("None",),
            html.Td("gini",),
            html.Td("50",),
            html.Td("auto",),
            html.Td("None",),
            html.Td("0.0",),
            html.Td("1",),
            html.Td("2",),
            html.Td("0.0",),
            html.Td("200",),
            html.Td("None",),
            html.Td("False",),
            html.Td("None",),
            html.Td("0",),
            html.Td("False",),
        ],),
    ],),
], className="ui inverted teal table")
], style={"width":"800px",  "overflow":"auto", "marginLeft":"-13px"},)

precision_table = html.Div([
    html.Td([
    html.Thead([
        html.Tr([
            html.Th("random forest",),
            html.Th("k nearest neighbour",),
            html.Th("naive bayes",),
            html.Th("gaussian naive bayes",),
            html.Th("support vector machine",),
            html.Th("logistic regression",),
        ],)
    ], className=""),

    html.Tbody([
        html.Tr([
            html.Td("0.9875",),
            html.Td("0.9833333333333333",),
            html.Td("0.9333333333333333",),
            html.Td("0.8375",),
            html.Td("0.7375",),
            html.Td("0.4625",),
        ],),
    ],),
], className="ui inverted teal table" )
], style={"width":"800px",  "overflow":"auto", "marginTop":"16px", "marginLeft":"-13px"},)

model_page = [
    html.Div([
        dbc.Row([
            dcc.Dropdown(
                id='ddl_x',
                options=[{'label': i, 'value': i} for i in (features+['is_fake'])],
                value='followings',
                style={'width':'200px'}
            ),
            dcc.Dropdown(
                id='ddl_y',
                options=[{'label': i, 'value': i} for i in (features+['is_fake'])],
                value='posts',
                style={'width':'200px', 'marginLeft':'10px'}
            ),
        ],),
        html.Div([
            dcc.Graph(id='graph1')
        ], style={'width':'100%', 'display':'inline-block', "border":"1px solid #e0cccc"})
    ], style={"marginTop":"5%"}),

    dbc.Row([
        dbc.Col([
            feature_importance_table,
            precision_table
        ], md=8),
        dbc.Col([
            html.Img(src="static/2.png", className="cover")
        ], md=4),
    ], style={"marginTop" : "5%"}),

    html.Div(
        dcc.Graph(
            figure={
                "data": [go.Bar(
                x=feature_importance,
                y=features,
                orientation='h',),
                ],
                "layout": go.Layout(
                    paper_bgcolor="#f5f2f2",
                    plot_bgcolor="#f5f2f2",
                    margin={'l': 275, 'b': 40, 't': 10, 'r': 0},
                    height=350
                )
            },
            config={"displayModeBar": False}
        ), style={'width':'100%', 'display':'inline-block', "marginTop":"2.2%", "marginBottom":"5%", "border":"1px solid #e0cccc"}),

    html.Div(style={"marginBottom":"5%"}),
]

database_page = [
    html.Div([
        dash_table.DataTable(
            id='database-table',
            columns=[
                {"name": i, "id": i} for i in columns
            ],
            data=database_rows,
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            # row_selectable="multi",
            row_deletable=True,
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 10,
            fixed_rows={'headers': True, 'data': 0},
            style_cell={'width': '150px', 'textAlign': 'left'},
            style_table={'maxHeight': '700px', 'width': '1211px',},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            export_format="xlsx",
            export_headers='display',
        ),
        dbc.Row([
            html.Button('Add Row', id='add-row', n_clicks=0, className="btn waves-effect waves-light table_btn", style={"color":"black"}),
            html.Button("Save Changes", id="save-btn", type="submit", className="btn waves-effect waves-light table_btn"),
        ], id="table_btn_row")

    ],style = {"marginTop" : "5%"}),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dcc.Dropdown(
                    id="fig1_col",
                    options=[
                        {'label': 'followers', 'value': 'followers'},
                        {'label': 'followings', 'value': 'followings'},
                        {'label': 'posts', 'value': 'posts'},
                        {'label': 'biography length', 'value': 'bio_lenght'},
                        {'label': 'following to follower', 'value': 'following_to_follower_ratio'},
                        {'label': 'following to post', 'value': 'following_to_post_ratio'},
                        {'label': 'follower to post', 'value': 'follower_to_post_ratio'},
                        # {'label': 'LCS of name and username to name', 'value': 'LCSofNameAndUsername_to_name_ratio'},
                        {'label': 'usernameNo to usernameLength', 'value': 'number_in_username_ratio'},
                    ]
                , placeholder="column", className="drop"),
                dcc.Dropdown(
                    id="fig1_op",
                    options=[
                        {'label': 'mean', 'value': 'mean'},
                        {'label': 'median', 'value': 'med'},
                    ]
                ,placeholder="operation", className="drop"),
            ]),
            dcc.Graph(
                id="bar_graph_1",
                figure=bar_figure_1,
                config={"displayModeBar": False},
            ),
        ], md=4),

        dbc.Col([
            dbc.Row([
                dcc.Dropdown(
                    id="fig2_col",
                    options=[
                        {'label': 'has biography', 'value': 'has_biography'},
                        {'label': 'is private', 'value': 'is_private'},
                        {'label': 'has name', 'value': 'has_name'},
                        {'label': 'has profile', 'value': 'has_profile_picture'},
                        {'label': 'name only alpha', 'value': 'name_only_alphabet'},
                    ]
                , placeholder="column", className="drop"),
            ]),
            dcc.Graph(
                id="bar_graph_2",
                figure=bar_figure_2,
                config={"displayModeBar": False},
            ),
        ], md=4),

        dbc.Col([
            dbc.Row([
                dcc.Dropdown(className="drop", style={"visibility":"hidden"}),
            ]),
            dcc.Graph(
                id="bar_graph_3",
                figure=bar_figure_3,
                config={"displayModeBar": False},
            ),
        ], md=4),
    ], style={"marginTop" : "4%"})

]

body = dbc.Container(
        # html.Div[predict_page, html.Div([html.Div(database_page), html.Div(model_page)], style={"display":"none"})]        ,
        id="main_container")

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/icon?family=Material+Icons',
    # {
    #     'href': 'https://fonts.googleapis.com/icon?family=Material+Icons',
    #     'rel': 'stylesheet',
    #     'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
    #     'crossorigin': 'anonymous'
    # }
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    navbar, body,
    html.Div([html.Div(database_page), html.Div(model_page), html.Div(predict_page)
              ], style={"display": "none"}),
    html.H6(str(temp_data), id="hidden-div", style={"display": "none"})
])


def LCSubStr(X, Y, m, n):
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i - 1] == Y[j - 1]):
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result


def extract_user_info(username):
    r = requests.get('https://www.instagram.com/' + str(username) + '/?__a=1')
    if(r.ok):
        result = json.loads(r.text or r.content)

        user = result['graphql']['user']
        followers = user['edge_followed_by']['count']
        followings = user['edge_follow']['count']
        has_biography = not(user['biography'] == "")
        is_private = user['is_private']
        has_name = not(user['full_name'] == "")
        posts = user['edge_owner_to_timeline_media']['count']
        has_profile_picture = (0 if '44884218_345707102882519_2446069589734326272_n.jpg' in user['profile_pic_url'] else 1)
        bio_lenght = len(user['biography'])
        if(followers):
            following_to_follower_ratio = followings / followers
        else:
            following_to_follower_ratio = 999999999999
        if(posts):
            following_to_post_ratio = followings / posts
            follower_to_post_ratio = followers / posts
        else:
            following_to_post_ratio = 999999999999
            follower_to_post_ratio = 999999999999

        name_only_alphabet = 1
        name = user['full_name']
        name = str(name)
        username = str(username)
        for c in name:
            if not(ord(c) in set(list(range(65,91)) + list(range(97,123)) + list(range(1570,1611)))):
                name_only_alphabet = 0
                break

        if(name):
            parsed_username = username.lower().replace(".", "").replace("_", "").replace(" ", "")
            name = name.lower().replace(" ", "")
            LCSofNameAndUsername_to_name_ratio = LCSubStr(parsed_username, name, len(parsed_username), len(name)) / len(name)
        else:
            LCSofNameAndUsername_to_name_ratio = 0
        numberCount = 0
        for c in username:
            if ord(c) in range(48,58):
                numberCount += 1

        number_in_username_ratio = numberCount/len(username)

        return({'username':username, 'followers':followers, 'followings':followings, 'has_biography':has_biography, 'is_private':is_private, 'has_name':has_name, 'posts':posts, 'has_profile_picture':has_profile_picture, 'bio_lenght':bio_lenght, 'following_to_follower_ratio':following_to_follower_ratio, 'following_to_post_ratio':following_to_post_ratio, 'follower_to_post_ratio':follower_to_post_ratio, 'name_only_alphabet':name_only_alphabet, 'LCSofNameAndUsername_to_name_ratio':LCSofNameAndUsername_to_name_ratio, 'number_in_username_ratio':number_in_username_ratio})
    return []


@app.callback(
    Output('graph1', 'figure'),
    [Input('ddl_x', 'value'),
     Input('ddl_y', 'value')]
)
def update_output(ddl_x_value, ddl_y_value):
    if(ddl_x_value is None or ddl_y_value is None):
        raise PreventUpdate
    figure={
        'data': [
            go.Scatter(
                x=cleaned_dataset[ddl_x_value],
                y=cleaned_dataset[ddl_y_value],
                mode='markers',
                marker={'size': 15,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}
                        },
            )
        ],
        'layout':
            go.Layout(
                plot_bgcolor="#f1f1f1",
                paper_bgcolor='#f1f1f1',
                height= 350,
                hovermode= 'closest',
                xaxis={
                    'title': ddl_x_value,
                    'automargin': True
                },
                yaxis={
                    'title': ddl_y_value,
                    'automargin': True
                },
                margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            )

    }
    return figure


@app.callback(
    [Output('upload-table', 'data'),
     Output('fake-table', 'data'),
     Output('bar_graph', 'figure')],
    [Input('upload-btn', 'n_clicks'),
     Input('predict-btn', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('predict-btn', 'n_clicks_timestamp'),
     State('upload-btn', 'n_clicks_timestamp'),
     State('username', 'value')])
def update_output(upload_clicks, submit_clicks, contents, filename, submit_ts, upload_ts, username):
    global database, database_rows
    if upload_clicks is None and submit_clicks is None:
        raise PreventUpdate
    if upload_ts > submit_ts:
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            file = []
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                file = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')), header=None)
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                file = pd.read_excel(io.BytesIO(decoded), header=None)
            else:
                return []

            usernames = list(file[0])

            df = pd.DataFrame([], columns=['username', 'followers', 'followings', 'has_biography', 'is_private', 'has_name',
                                           'posts', 'has_profile_picture', 'bio_lenght',
                                           'following_to_follower_ratio', 'following_to_post_ratio',
                                           'follower_to_post_ratio', 'name_only_alphabet',
                                           'LCSofNameAndUsername_to_name_ratio', 'number_in_username_ratio'])

            for username in usernames:
                info = extract_user_info(username)
                if info:
                    df = df.append(info, ignore_index=True, sort=False)

            if df.shape[0] != 0:
                temp = df.drop(df.username.name, axis=1)
                results = list(model.predict(temp))
                df['is_fake'] = results
                df = df.applymap(lambda x: 1 if x == True else x)
                df = df.applymap(lambda x: 0 if x == False else x)
                df = df.applymap(lambda x: np.nan if x == 999999999999 else x)

                db = pd.DataFrame()
                try:
                    db = pd.read_csv("data/database.csv")
                    db = db[~db['username'].isin(usernames)]
                except:
                    pass

                database = db.append(df, ignore_index=True, sort=False)
                database_rows = database.to_dict('records')
                database.to_csv("data/database.csv", index=False)

                prediction = pd.DataFrame()
                prediction['username'] = usernames
                prediction['is_fake'] = results

                # counts = df['is_fake'].value_counts()
                fake_count = df[df.is_fake==1].shape[0]
                legal_count = df[df.is_fake==0].shape[0]

                figure = {"data": [
                            go.Bar(
                                x=[
                                    "legal",
                                    "bot",
                                ],
                                y=[
                                    legal_count,
                                    fake_count,
                                ],
                                marker={
                                    "color": "#97151c",
                                    "line": {
                                        "color": "#97151c",
                                        "width": 1,
                                    },
                                },
                                name="Calibre Index Fund",
                            ),
                        ],
                        "layout": go.Layout(
                            plot_bgcolor=bgcolor,
                            paper_bgcolor=bgcolor,
                            autosize=False,
                            bargap=0.35,
                            font={"family": "Raleway", "size": 15},
                            height=257,
                            hovermode="closest",
                            # legend={
                            #     "x": -0.0228945952895,
                            #     "y": -0.189563896463,
                            #     "orientation": "h",
                            #     "yanchor": "top",
                            # },
                            margin={
                                "r": 0,
                                "t": 10,
                                "b": 50,
                                "l": 30,
                            },
                            # showlegend=True,
                            title="",
                            width=276,
                            xaxis={
                                "autorange": True,
                                # "range": [-0.5, 4.5],
                                "showline": True,
                                "title": "",
                                "type": "category",
                            },
                            yaxis={
                                "autorange": True,
                                # "range": [0, 22.9789473684],
                                "showgrid": True,
                                "showline": True,
                                "title": "",
                                "type": "linear",
                                "zeroline": False,
                            },
                        ),
                    }
                return df.to_dict('records'), prediction.to_dict('records'), figure
    elif submit_ts > upload_ts:
        if username is not None:
            df = pd.DataFrame([],
                              columns=['username', 'followers', 'followings', 'has_biography', 'is_private', 'has_name',
                                       'posts', 'has_profile_picture', 'bio_lenght',
                                       'following_to_follower_ratio', 'following_to_post_ratio',
                                       'follower_to_post_ratio', 'name_only_alphabet',
                                       'LCSofNameAndUsername_to_name_ratio', 'number_in_username_ratio'])

            info = extract_user_info(username)
            if info:
                df = df.append(info, ignore_index=True, sort=False)

            if df.shape[0] != 0:
                print(df.shape[0])
                temp = df.drop(df.username.name, axis=1)
                results = list(model.predict(temp))
                df['is_fake'] = results
                df = df.applymap(lambda x: 1 if x == True else x)
                df = df.applymap(lambda x: 0 if x == False else x)
                df = df.applymap(lambda x: np.nan if x == 999999999999 else x)

                db = pd.DataFrame()
                try:
                    db = pd.read_csv("data/database.csv")
                    db = db[db['username'] != username]
                except:
                    pass

                database = db.append(df, ignore_index=True, sort=False)
                database_rows = database.to_dict('records')
                database.to_csv("data/database.csv", index=False)

                prediction = pd.DataFrame()
                prediction['username'] = [username]
                prediction['is_fake'] = results

                counts = df['is_fake'].value_counts()
                fake_count = df[df.is_fake == 1].shape[0]
                legal_count = df[df.is_fake == 0].shape[0]

                figure = {"data": [
                    go.Bar(
                        x=[
                            "legal",
                            "bot",
                        ],
                        y=[
                            legal_count,
                            fake_count,
                        ],
                        marker={
                            "color": "#97151c",
                            "line": {
                                "color": "#97151c",
                                "width": 1,
                            },
                        },
                        name="Calibre Index Fund",
                    ),
                ],
                    "layout": go.Layout(
                        plot_bgcolor=bgcolor,
                        paper_bgcolor=bgcolor,
                        autosize=False,
                        bargap=0.35,
                        font={"family": "Raleway", "size": 15},
                        height=257,
                        hovermode="closest",
                        # legend={
                        #     "x": -0.0228945952895,
                        #     "y": -0.189563896463,
                        #     "orientation": "h",
                        #     "yanchor": "top",
                        # },
                        margin={
                            "r": 0,
                            "t": 10,
                            "b": 50,
                            "l": 30,
                        },
                        # showlegend=True,
                        title="",
                        width=276,
                        xaxis={
                            "autorange": True,
                            # "range": [-0.5, 4.5],
                            "showline": True,
                            "title": "",
                            "type": "category",
                        },
                        yaxis={
                            "autorange": True,
                            # "range": [0, 22.9789473684],
                            "showgrid": True,
                            "showline": True,
                            "title": "",
                            "type": "linear",
                            "zeroline": False,
                        },
                    ),
                }
                return df.to_dict('records'), prediction.to_dict('records'), figure
    return [{} for _ in range(6)], [{} for _ in range(6)], bar_figure


@app.callback(
    Output('database-table', 'data'),
    [Input('add-row', 'n_clicks'),
     Input("nav_tabs", "active_tab")],
    [State('database-table', 'data'),
     State('database-table', 'columns')])
def add_row(n_clicks, at, rows, col):
    global add_row_nclick, database, database_rows
    if n_clicks > 0 and add_row_nclick != n_clicks:
        print(n_clicks, add_row_nclick)
        add_row_nclick = n_clicks
        rows.append({c['id']: '' for c in col})
        database_rows = rows
    elif(at == "database_tab"):
        try:
            database = pd.read_csv("data/database.csv")
            database_rows = database.to_dict('records')
        except:
            pass
    else:
        raise PreventUpdate

    return database_rows


@app.callback(Output("hidden-div", "children"),
              [Input("save-btn", "n_clicks")],
              [State("database-table", 'data')])
def save(nc, rows):
    if nc is None:
        raise PreventUpdate
    new_db = pd.DataFrame(rows)
    global database, database_rows
    database = new_db
    database_rows = database.to_dict('records')
    new_db.to_csv("data/database.csv", index=False)
    global temp_data
    temp_data = 0 if temp_data else 1
    return str(temp_data)


@app.callback(Output("main_container", "children"),
              [Input("nav_tabs", "active_tab")])
def switch_tab(at):
    rows = database_rows
    if at == "predict_tab":
        return predict_page
    elif at == "model_tab":
        return model_page
    elif at == "database_tab":
        return database_page
    return predict_page, rows


@app.callback(
    Output('bar_graph_1', 'figure'),
    [Input('fig1_op', 'value'),
     Input('fig1_col', 'value'),
     Input("hidden-div", "children")])
def update_fig1(op, col, hid):
    fig = bar_figure_1
    if(op and col and op == "med"):
        fig = {"data": [
            go.Bar(
                x=[
                    "legal",
                    "bot",
                ],
                y=[
                    database[database.is_fake == 0][col].median(skipna=True),
                    database[database.is_fake == 1][col].median(skipna=True),
                ],
                marker={
                    "color": "#97151c",
                    "line": {
                        "color": "#97151c",
                        "width": 1,
                    },
                },
                name="Calibre Index Fund",
            ),
        ],
            "layout": go.Layout(
                plot_bgcolor=bgcolor,
                paper_bgcolor=bgcolor,
                autosize=False,
                bargap=0.35,
                font={"family": "Raleway", "size": 15},
                height=350,
                hovermode="closest",
                # legend={
                #     "x": -0.0228945952895,
                #     "y": -0.189563896463,
                #     "orientation": "h",
                #     "yanchor": "top",
                # },
                margin={
                    "r": 0,
                    "t": 10,
                    "b": 50,
                    "l": 30,
                },
                # showlegend=True,
                title="",
                width=340,
                xaxis={
                    "autorange": True,
                    # "range": [-0.5, 4.5],
                    "showline": True,
                    "title": "",
                    "type": "category",
                },
                yaxis={
                    "autorange": True,
                    # "range": [0, 22.9789473684],
                    "showgrid": True,
                    "showline": True,
                    "title": "",
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }
    elif(op and col and op == "mean"):
        fig = {"data": [
            go.Bar(
                x=[
                    "legal",
                    "bot",
                ],
                y=[
                    database[database.is_fake == 0][col].mean(skipna = True),
                    database[database.is_fake == 1][col].mean(skipna = True),
                ],
                marker={
                    "color": "#97151c",
                    "line": {
                        "color": "#97151c",
                        "width": 1,
                    },
                },
                name="Calibre Index Fund",
            ),
        ],
        "layout": go.Layout(
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor,
            autosize=False,
            bargap=0.35,
            font={"family": "Raleway", "size": 15},
            height=350,
            hovermode="closest",
            # legend={
            #     "x": -0.0228945952895,
            #     "y": -0.189563896463,
            #     "orientation": "h",
            #     "yanchor": "top",
            # },
            margin={
                "r": 0,
                "t": 10,
                "b": 50,
                "l": 30,
            },
            # showlegend=True,
            title="",
            width=340,
            xaxis={
                "autorange": True,
                # "range": [-0.5, 4.5],
                "showline": True,
                "title": "",
                "type": "category",
            },
            yaxis={
                "autorange": True,
                # "range": [0, 22.9789473684],
                "showgrid": True,
                "showline": True,
                "title": "",
                "type": "linear",
                "zeroline": False,
            },
        ),
    }
    return fig


@app.callback(
    Output('bar_graph_2', 'figure'),
    [Input('fig2_col', 'value'),
     Input("hidden-div", "children")])
def update_fig2(col, hid):
    fig = bar_figure_2
    if(col):
        fig = {
            "data": [
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        database[(database[col]==0) & (database.is_fake==0)].shape[0],
                        database[(database[col]==0) & (database.is_fake==1)].shape[0],
                    ],
                    marker={
                        "color": "#97151c",
                        "line": {
                            "color": "#97151c",
                            "width": 1,
                        },
                    },
                    name="#0",
                ),
                go.Bar(
                    x=[
                        "legal",
                        "bot",
                    ],
                    y=[
                        database[(database[col] == 1) & (database.is_fake == 0)].shape[0],
                        database[(database[col] == 1) & (database.is_fake == 1)].shape[0],
                    ],
                    marker={
                        "color": "#dddddd",
                        "line": {
                            "color": "#d4d2d2",
                            "width": 1,
                        },
                    },
                    name="#1",
                ),
            ],
            "layout": go.Layout(
                plot_bgcolor=bgcolor,
                paper_bgcolor=bgcolor,
                autosize=True,
                bargap=0.35,
                font={"family": "Raleway", "size": 15},
                height=350,
                hovermode="closest",
                # legend={
                #     "x": -0.0228945952895,
                #     "y": -0.189563896463,
                #     "orientation": "h",
                #     "yanchor": "top",
                # },
                margin={
                    "r": 0,
                    "t": 10,
                    "b": 50,
                    "l": 30,
                },
                # showlegend=True,
                title="",
                width=400,
                xaxis={
                    "autorange": True,
                    # "range": [-0.5, 4.5],
                    "showline": True,
                    "title": "",
                    "type": "category",
                },
                yaxis={
                    "autorange": True,
                    # "range": [0, 22.9789473684],
                    "showgrid": True,
                    "showline": True,
                    "title": "",
                    "type": "linear",
                    "zeroline": False,
                },
            ),
        }
    return fig


@app.callback(
    Output('bar_graph_3', 'figure'),
    [Input("hidden-div", "children")])
def update_fig3(hid):
    fig = bar_figure_3
    if(database.shape[0]!=0):
        fig = {"data": [
                    go.Bar(
                        x=[
                            "legal",
                            "bot",
                        ],
                        y=[
                            database[database.is_fake == 0].shape[0],
                            database[database.is_fake == 1].shape[0],
                        ],
                        marker={
                            "color": "#97151c",
                            "line": {
                                "color": "#97151c",
                                "width": 1,
                            },
                        },
                        name="Calibre Index Fund",
                    ),
                ],
                "layout": go.Layout(
                    plot_bgcolor=bgcolor,
                    paper_bgcolor=bgcolor,
                    autosize=False,
                    bargap=0.35,
                    font={"family": "Raleway", "size": 15},
                    height=350,
                    hovermode="closest",
                    # legend={
                    #     "x": -0.0228945952895,
                    #     "y": -0.189563896463,
                    #     "orientation": "h",
                    #     "yanchor": "top",
                    # },
                    margin={
                        "r": 0,
                        "t": 10,
                        "b": 50,
                        "l": 30,
                    },
                    # showlegend=True,
                    title="",
                    width=340,
                    xaxis={
                        "autorange": True,
                        # "range": [-0.5, 4.5],
                        "showline": True,
                        "title": "",
                        "type": "category",
                    },
                    yaxis={
                        "autorange": True,
                        # "range": [0, 22.9789473684],
                        "showgrid": True,
                        "showline": True,
                        "title": "",
                        "type": "linear",
                        "zeroline": False,
                    },
                ),
            }
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

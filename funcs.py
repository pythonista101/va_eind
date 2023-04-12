from streamlit import *
import folium
from streamlit_folium import folium_static
import pandas as pd
import xgboost as xgb
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
import seaborn as sns
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = (df.index.isocalendar().week).astype("int64")
    return df


def get_prediction_from_1d_timeseries(start, end, df, obj_colname):
    # pullung features out of dt-index
    df = create_features(df)

    # X-y split
    X = df[['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
    y = df[obj_colname]

    pred_df = pd.DataFrame()
    index_list = pd.date_range(start, end, freq='D')
    pred_df.index = index_list
    pred_df = create_features(pred_df)

    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                             n_estimators=1000,
                             objective='reg:linear',
                             max_depth=3,
                             learning_rate=0.01)
    model = model.fit(X, y, verbose=100)

    preddy = pd.DataFrame({'Temp': model.predict(pred_df)})
    preddy.index = index_list

    return preddy


def get_prediction_df(start, end, df):
    build_df = pd.DataFrame()
    build_df.index = pd.date_range(start, end, freq='D')

    for city in df.columns:
        build_df[city] = get_prediction_from_1d_timeseries(start, end, df[[city]], city)

    return build_df







def linechart_choice(build_df, best_choice, gevraagde_temp):
    # best_choice = 'Amsterdam'
    # gevraagde_temp = 20


    # create an empty figure
    fig = go.Figure()

    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000000', '#008080']

    for colname, color in zip(build_df, colors):
        y=build_df[colname]

        fig.add_trace(go.Scatter(
            x=build_df.index,
            y=y,
            mode='lines+markers',
            name=colname,
            line=dict(color=color),
            visible=('legendonly' if colname != best_choice else None)))


    fig.update_layout(
        title='Temperatuur per bestemming',
        xaxis=dict(title='Tijd (gehele verblijf)'),
        yaxis=dict(title='Temperatuur °C')
    )



    fig.add_shape(type='line',
                  yref='y',
                  xref='paper',
                  y0=gevraagde_temp,
                  y1=gevraagde_temp,
                  x0=0,
                  x1=1,
                  line=dict(color='grey', width=3, dash='dashdot'),
                  name='Line')

    # Show the legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='grey', width=3, dash='dashdot'),
                             showlegend=True, name='Gevraagde temp'))
    fig.update_layout(template='gridon')


    fig.update_layout(
        plot_bgcolor='#A7E4F2'
    )


    return fig


def statplot(df, city):
    test = df[[city]]
    test['m'] = test.index.month
    x = test.groupby('m')[city].agg(['mean', 'std'])

    test1 = df[[city]]
    test1['m'] = test1.index.month_name()
    y = test1.groupby('m')[city].agg(['mean', 'std'])
    plot_df = x.merge(y.reset_index().drop(columns='std'), on='mean')

    return px.bar(plot_df, x='m', y='mean', error_y='std', barmode='group').add_trace(
        px.line(plot_df, x='m', y='mean').data[0]).update_layout(yaxis_title='Temperatuur °C', xaxis_title='Maand',
                                                                 title=f'Klimaat in {city}')




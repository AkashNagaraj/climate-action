import numpy as np                   # for multi-dimensional containers 
import pandas as pd                  # for DataFrames
import plotly.graph_objects as go    # for data visualisation
import plotly.express as px
import sys

def plot_temp_data():
    access_token = 'pk.eyJ1Ijoic2hhaGlucm9zdGFtaSIsImEiOiJjbDZmbXhjY28wOG1uM2tyeGhxcWdzMjYwIn0.Yi8ZtU8TVZI7kyibQhqFzA'
    px.set_mapbox_access_token(access_token)
    data = pd.read_csv('data/city_avg_temp.csv')
    data[data["avg_temp"]<0]=0

    fig = px.scatter_mapbox(
    data, lat="Lat", lon="Long",
    size="avg_temp", size_max=15,
    color="avg_temp", color_continuous_scale=px.colors.sequential.Oranges,
    hover_name="city",
    mapbox_style='light', zoom=1
    )
    fig.layout.coloraxis.showscale = False
    fig.show()

    fig = px.scatter_mapbox(
    data, lat="Lat", lon="Long",
    size="avg_temp", size_max=15,
    color="avg_temp", color_continuous_scale=px.colors.sequential.Bluered,
    hover_name="city",
    mapbox_style='streets', zoom=1,
    animation_frame="year", animation_group="city"
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    fig.layout.coloraxis.showscale = False
    fig.layout.sliders[0].pad.t = 10
    fig.layout.updatemenus[0].pad.t= 10
    fig.show()


def plot_air_quality_data():
    access_token = 'pk.eyJ1Ijoic2hhaGlucm9zdGFtaSIsImEiOiJjbDZmbXhjY28wOG1uM2tyeGhxcWdzMjYwIn0.Yi8ZtU8TVZI7kyibQhqFzA'
    px.set_mapbox_access_token(access_token)
    data = pd.read_csv('data/monthly_air_quality.csv')
   
    #https://plotly.com/python/builtin-colorscales/
    fig = px.scatter_mapbox(data.head(n=2000), lat="Lat", lon="Long", size="pm2.5", size_max=50, color="co", color_continuous_scale=px.colors.sequential.Rainbow, hover_name="city", mapbox_style='light', zoom=1,animation_frame="date")
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    fig.layout.coloraxis.showscale = False
    fig.layout.sliders[0].pad.t = 5
    fig.layout.updatemenus[0].pad.t = 5
    fig.show()

#plot_air_quality_data()
plot_temp_data()
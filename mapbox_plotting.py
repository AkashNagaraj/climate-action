import numpy as np                   # for multi-dimensional containers 
import pandas as pd                  # for DataFrames
import plotly.graph_objects as go    # for data visualisation
import plotly.express as px

access_token = 'pk.eyJ1Ijoic2hhaGlucm9zdGFtaSIsImEiOiJjbDZmbXhjY28wOG1uM2tyeGhxcWdzMjYwIn0.Yi8ZtU8TVZI7kyibQhqFzA'
px.set_mapbox_access_token(access_token)
data = pd.read_csv('data/city_avg_temp.csv')
data[data["avg_temp"]<0]=0

fig = px.scatter_mapbox(
    data, lat="Lat", lon="Long",
    size="avg_temp", size_max=15,
    color="avg_temp", color_continuous_scale=px.colors.sequential.Pinkyl,
    hover_name="city",
    mapbox_style='dark', zoom=1
)
fig.layout.coloraxis.showscale = False
fig.show()

fig = px.scatter_mapbox(
    data, lat="Lat", lon="Long",
    size="avg_temp", size_max=15,
    color="avg_temp", color_continuous_scale=px.colors.sequential.Pinkyl,
    hover_name="city",
    mapbox_style='dark', zoom=1,
    animation_frame="year", animation_group="city"
)

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
fig.layout.coloraxis.showscale = False
fig.layout.sliders[0].pad.t = 10
fig.layout.updatemenus[0].pad.t= 10
fig.show()



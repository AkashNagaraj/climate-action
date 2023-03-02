# Import libraries
import numpy as np                   
import pandas as pd                 
import plotly.graph_objects as go    
import plotly.express as px
import sys

# Plot the time-series temperature data for cities 
def plot_temp_data():
    access_token = 'pk.eyJ1Ijoic2hhaGlucm9zdGFtaSIsImEiOiJjbDZmbXhjY28wOG1uM2tyeGhxcWdzMjYwIn0.Yi8ZtU8TVZI7kyibQhqFzA'
    px.set_mapbox_access_token(access_token)
    data = pd.read_csv('data/city_avg_temp.csv')
    data[data["avg_temp"]<0]=0

    # Plot a figure
    fig = px.scatter_mapbox(data, lat="Lat", lon="Long", size="avg_temp", size_max=15, color="avg_temp", color_continuous_scale=px.colors.sequential.PuRd, hover_name="city", mapbox_style='dark', zoom=1,title="Initial Temperature")
    fig.layout.coloraxis.showscale = False
    fig.show()

    # Plot the animation
    fig = px.scatter_mapbox(
    data, lat="Lat", lon="Long",
    size="avg_temp", size_max=15,
    color="avg_temp", color_continuous_scale=px.colors.sequential.Bluered,
    hover_name="city",
    mapbox_style='streets', zoom=1,
    animation_frame="year", animation_group="city",title="Temperature Intensity In Major Cities"
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    fig.layout.coloraxis.showscale = False
    fig.layout.sliders[0].pad.t = 10
    fig.layout.updatemenus[0].pad.t= 10
    fig.show()


# Plot the air quality dataset
def plot_air_quality_data():
    access_token = 'pk.eyJ1Ijoic2hhaGlucm9zdGFtaSIsImEiOiJjbDZmbXhjY28wOG1uM2tyeGhxcWdzMjYwIn0.Yi8ZtU8TVZI7kyibQhqFzA'
    px.set_mapbox_access_token(access_token)
    data = pd.read_csv('data/monthly_air_quality.csv')
   
    fig = px.scatter_mapbox(data.head(n=2000), lat="Lat", lon="Long", size="pm2.5", size_max=35, color="co", color_continuous_scale=px.colors.sequential.Rainbow, hover_name="city", mapbox_style='light', zoom=1,animation_frame="date",title="Carbon Monoxide Intensity")
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
    fig.layout.coloraxis.showscale = True
    fig.layout.sliders[0].pad.t = 5
    fig.layout.updatemenus[0].pad.t = 5
    
    fig.update_layout(title_x=0.5,title_y=0.95,margin={"l": 0, "r": 0, "b": 0, "t": 80})
    fig.show()

plot_air_quality_data()
plot_temp_data()

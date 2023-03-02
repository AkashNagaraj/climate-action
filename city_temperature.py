# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn, math, sys

import plotly.graph_objects as go
import plotly.express as px
import ozon3
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from meteostat import Point, Daily, Monthly, Stations

# Get a list of cities from a dataset
def get_lat_long():
    data_dir = "data/worldcities.csv"
    df = pd.read_csv(data_dir)
    return df.head(n=200)

# Use ARIMA(a time series forecasting model) to predict future values
def predict_using_arima(end,data):
    train_size = 0.8
    split_idx = math.ceil(len(data)*train_size)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    #print("Length of training data : {}, leng of testing is : {}".format(len(train_data), len(test_data)))
    
    """
    # Plot the train and test data
    fig,ax = plt.subplots(figsize=(12,8))
    kws = dict(marker='o')
    plt.plot(train_data, label='Train', **kws)
    plt.plot(test_data, label='Test', **kws)
    ax.legend(bbox_to_anchor=[1,1]);
    plt.show()
    """
    
    p,d,q = 1,1,1
    
    predictions = []
    new_end = '2030-1-1'
    required_prediction_size = len(pd.date_range(end,new_end,freq='M'))-1

    for i in range(required_prediction_size):
        model = sm.tsa.arima.ARIMA(data, order=(p,d,q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        data = np.append(data,yhat)

    return predictions, new_end


# Given a city's name, lat, and longitude get the details using the meteostat API
def get_city_specific_data(city_details):
    
    # how many null values should be present in the dataframe
    threshold = 50

    lat, long, city_name = city_details
    stations = Stations()
    stations = stations.nearby(lat, long)
    station = stations.fetch(1)

    start = datetime(2000,1,1)
    end = datetime(2022,12,31)

    # Get daily/mothly data
    data = Monthly(station, start, end)
    data = data.normalize()
    data = data.fetch()
    data = data["tavg"]

    #data.plot(y=['tavg','tmin','tmax'])
    #plt.show()
    
    predictions = []
    
    if(data.isna().sum()<threshold):
        data = data.fillna(method="ffill")
        data = data.fillna(0)
        predictions, new_end = predict_using_arima(end,data)
        complete_temp_data = data.tolist() + predictions

        print("The city details are : {} and the complete_temp_data size : {}".format(city_details,len(complete_temp_data)))
        
        return start, new_end, complete_temp_data
    
    else:

        return 0, 0, []

# Obtain temperature and humidity related data for a specific city
def get_meteostat_data(city_details):
    
    city_data = city_details[["city","lat","lng"]]
    new_dataframe_data = pd.DataFrame()
    all_temp_data, all_date_data, all_lat_data, all_long_data, all_city_data = [],[], [],[],[]
    for idx, row in city_data.iterrows():

        start, end, temp_data = get_city_specific_data((row["lat"],row["lng"],row["city"]))
        if len(temp_data)!=0:
            year_range = pd.date_range(start,end,freq='M')       
            all_temp_data += temp_data
            all_date_data += year_range.tolist()
            all_lat_data += [row["lat"]]*len(year_range)
            all_long_data += [row["lng"]]*len(year_range)
            all_city_data += [row["city"]]*len(year_range)
    
    #print(len(all_temp_data), len(all_date_data), len(all_lat_data),len(all_long_data),len(all_city_data))
    
    new_dataframe_data["avg_temp"] = all_temp_data
    new_dataframe_data["year"] = all_date_data
    new_dataframe_data["Lat"] = all_lat_data
    new_dataframe_data["Long"] = all_long_data
    new_dataframe_data["city"] = all_city_data
            
    df = pd.DataFrame(new_dataframe_data)
    df.to_csv("data/city_avg_temp.csv")

def main():
    city_details = get_lat_long()
    get_meteostat_data(city_details)

if __name__=="__main__":
    main()

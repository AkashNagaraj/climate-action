import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn, math, sys

import plotly.graph_objects as go
import plotly.express as px
import pypopulation
import ozon3
import statsmodels.api as sm


from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima.arima import auto_arima
from datetime import datetime
from meteostat import Point, Daily, Monthly, Stations
from neuralprophet import NeuralProphet

def get_lat_long():
    data_dir = "data/worldcities.csv"
    df = pd.read_csv(data_dir)
    return df.head()


def forecast_to_df(model, steps=12):
    forecast = model.get_forecast(steps=steps)
    pred_df = forecast.conf_int()
    pred_df['pred'] = forecast.predicted_mean
    pred_df.columns = ['lower', 'upper', 'pred']
    return pred_df


def get_pdq_values(train_data, test_data):
    p,d,q = 1,1,1
    history = train_data.values
    predictions = []

    
    for i in range(len(test_data)):
        model = sm.tsa.arima.ARIMA(history, order=(p,d,q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[i]
        history = np.append(history,obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))

    rmse = sqrt(mean_squared_error(test_data, predictions))
    print('Test RMSE: %.3f' % rmse)


def predict_using_arima(end,data):
    train_size = 0.8
    split_idx = math.ceil(len(data)*train_size)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print("Length of training data : {}, leng of testing is : {}".format(len(train_data), len(test_data)))

    """
    # Plot the train and test data
    fig,ax = plt.subplots(figsize=(12,8))
    kws = dict(marker='o')
    plt.plot(train_data, label='Train', **kws)
    plt.plot(test_data, label='Test', **kws)
    ax.legend(bbox_to_anchor=[1,1]);
    plt.show()
    """
    
    # get_pdq_values(train_data, test_data)
    p,d,q = 1,1,1
    
    history = data.values
    predictions = []
    
    required_prediction_size = 2030 - end.year

    for i in range(required_prediction_size):
        model = sm.tsa.arima.ARIMA(history, order=(p,d,q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[i]
        history = np.append(history,obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))

    return predictions


def predict_neural_network(end,df):
    m = NeuralProphet()
    
    df_train, df_val = m.split_df(df, freq='M', valid_p = 0.2)
    
    #metrics = m.fit(df_train, freq='M', validation_df=df_val, plot_live_loss=True)
    
    #future = m.make_future_dataframe(df, periods=24, n_historic_predictions=len(df))
    #forecast = m.predict(future)
    #print(metrics)


def get_city_specific_data(city):
    
    # how many null values should be present in the dataframe
    threshold = 50

    lat, long, city_name = city
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
        print("The below data is for the city :",city_name)
        predictions = predict_using_arima(end,data)
        print("The predictions are :", predictions)
    

def get_meteostat_data(city_details):
    city_data = city_details[["city","lat","lng"]]
    for idx, row in city_data.iterrows():
        get_city_specific_data((row["lat"],row["lng"],row["city"]))


def main():
    city_details = get_lat_long()
    get_meteostat_data(city_details)
    """
    build_dataframe()
    plot_mapbox()
    air_quality()
    compare_temperature_air_quality()
    """

if __name__=="__main__":
    main()

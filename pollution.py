import pandas as pd
import ozon3 as oz
import sys


def single_city_value(city):
    o3 = oz.Ozon3("dbcf6b562d074d43fbe2ae147957240210619015")

    ID = o3.get_city_station_options(city)
    #print(ID)
    air_quality_data = o3.get_city_air(city)
    historical_data = o3.get_historical_data(city).fillna(0)
    row_size, column_size = historical_data.shape
    city_name = [city]*row_size

    #print(air_quality_data["latitude"].tolist()[0])

    Lat = [air_quality_data["latitude"].tolist()[0]]*row_size
    Long = [air_quality_data["longitude"].tolist()[0]]*row_size
    historical_data["city"] = city_name
    historical_data["Lat"] = Lat
    historical_data["Long"] = Long
    
    return historical_data


def get_pollution_data(city_list):
    
    total_climate_data = []
    for idx,city in enumerate(city_list):
        if(idx%5==0):
            print("Completed : {}",idx)
        try:
            total_climate_data.append(single_city_value(city))
        except:
            print("City data doesnt exist : ",city)
            continue


    result = pd.concat(total_climate_data)
    result.to_csv("data/air_quality.csv")


def read_data():
    df = pd.read_csv("data/city_avg_temp.csv")
    unique_city_list = df["city"].unique().tolist()
    return unique_city_list


def main():
    city_list = read_data()
    count = 25
    get_pollution_data(city_list[:count])


if __name__=="__main__":
    main()

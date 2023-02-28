import pandas as pd
import ozon3 as oz
import sys


def single_city_value(city):
    o3 = oz.Ozon3("dbcf6b562d074d43fbe2ae147957240210619015")

    ID = o3.get_city_station_options(city)
    #print(ID)
    air_quality_data = o3.get_city_air(city)
    historical_data = o3.get_historical_data(city).ffill() #fillna(0)
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


def  modify_dataframe():
    o3 = oz.Ozon3("dbcf6b562d074d43fbe2ae147957240210619015")

    data = pd.read_csv('data/air_quality.csv')
    data = data.sort_values("date")
    data = data.ffill()
    data = data.fillna(0)

    data["date"] =pd.to_datetime(data["date"])
    new_df = data.groupby([data["date"].dt.year,data["date"].dt.month,'city']).agg({'co':'sum','pm2.5':'sum'})

    new_df.to_csv("data/monthly_air_quality.csv")
    new_df = pd.read_csv("data/monthly_air_quality.csv")
    
    for city in data["city"].unique():
        air_quality_data = o3.get_city_air(city)
        Lat = air_quality_data["latitude"].tolist()[0]
        Long = air_quality_data["longitude"].tolist()[0]
        new_df.loc[new_df["city"]==city,"Lat"] = Lat
        new_df.loc[new_df["city"]==city,"Long"] = Long

    new_df["date"] = new_df["date"].astype(str)+"-"+new_df["date.1"].astype(str)
    new_df.to_csv("data/monthly_air_quality.csv")

def main():
    """
    city_list = read_data()
    count = 40
    get_pollution_data(city_list[:count])
    print("Modifying the dataframe")
    """
    modify_dataframe()

if __name__=="__main__":
    main()

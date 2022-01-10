import pandas as pd
from datetime import datetime
import numpy as np
from meteostat import Stations, Daily, Point
from pandas.io.sql import DatabaseError
import haversine as hs
from haversine import Unit

def dates_encoder(X):
    
    # Make sure that flight date is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "flight_date"] = pd.to_datetime(X['flight_date'])
    
    # Encode the date information from the flight_date column
    
    X.loc[:, 'year'] = X['flight_date'].dt.year
    X.loc[:, 'month'] = X['flight_date'].dt.month
    X.loc[:, 'day'] = X['flight_date'].dt.day
    X.loc[:, 'weekday'] = X['flight_date'].dt.weekday
    X.loc[:, 'week'] = X['flight_date'].dt.isocalendar().week
    X.loc[:, 'number_of_days'] = X['flight_date'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["flight_date"])

def merge_path(X):
    
    X = X.copy()
    X.loc[:, 'path'] = X['from'] + '_' + X['to']
    
    return X.drop(columns = ['from','to'])


def merge_stock_data(X):

    '''
the data X needs to be indexed by the flights date column, 
the date format is year-month-day.
The stock data used here was taken from Finance.Yahoo 
of American Airlines Group Inc. (AAL) stocks

    ''' 
    X = X.copy()
    stocks_data = pd.read_csv('../data/stock_external_data.csv')
    # take only closing price of the stock
    stocks_data = stocks_data[['Date','Close']]
    stocks_data.rename(columns={"Date": "flight_date"}, inplace=True)
    merged = pd.merge(X, stocks_data, how="outer", on=["flight_date"])

    return merged

def get_distance(X):
    url="https://raw.githubusercontent.com/ravisurdhar/flight_delays/master/airports.csv"
    us_airports=pd.read_csv(url)
    us_airports['coordinate_from'] = list(zip(us_airports.LATITUDE, us_airports.LONGITUDE))
    us_airports = us_airports[['IATA_CODE','coordinate_from']]
    us_airports['coordinate_to'] = us_airports['coordinate_from']
    us_airports = us_airports.rename(columns={'IATA_CODE':'from'})
    data = pd.merge(X, us_airports, on=['from'])
    us_airports = us_airports.rename(columns={'from':'to'})
    data = pd.merge(data, us_airports, on=['to'])
    data = data.rename(columns={'coordinate_from_x':'coordinate_from', 'coordinate_to_y':'coordinate_to'})
    data = data.drop(labels=['coordinate_to_x', 'coordinate_from_y'], axis=1)
    list_coordinates = list(zip(data['coordinate_from'].tolist(), data['coordinate_to'].tolist()))
    dist=[]
    for i in range(len(list_coordinates)):
        dist.append(hs.haversine(list_coordinates[i][0], list_coordinates[i][1],unit=Unit.KILOMETERS))
    data['distance'] = dist
    data = data.drop(labels=['coordinate_from', 'coordinate_to'], axis=1)
    return data

def scrap_weather(X, start, end):
    url="https://raw.githubusercontent.com/ravisurdhar/flight_delays/master/airports.csv"
    us_airports=pd.read_csv(url)
    airports = np.unique((np.append(X['from'].unique(), X['to'].unique())))
    airport_localisation = us_airports[us_airports['IATA_CODE'].isin(airports)]
    lats = airport_localisation['LATITUDE'].tolist()
    longs = airport_localisation['LONGITUDE'].tolist()
    iatas = airports.tolist()
    list_geo = list(zip(lats,longs))
    iatas = airports.tolist()
    list_data=[]

    for i in range(len(list_geo)):
        point = Point(list_geo[i][0], list_geo[i][1])
        data = Daily(point, start, end)
        df = data.fetch()
        df = df.drop(labels=['snow', 'wdir', 'wpgt', 'pres', 'tsun'], axis=1)
        list_data.append(df.reset_index())
    
    weather_data = pd.concat(list_data, axis=0)
    iatas_repeated = [iata for iata in iatas for i in range(list_data[0].shape[0])]
    weather_data['airport_code'] = iatas_repeated
    return weather_data

def merge_weather_data(X, weather_data):
    X = X.copy()
    weather_data = weather_data[['time','tavg','prcp','wspd','airport_code']]

    weather_data_from = weather_data.rename(columns={"time": "flight_date",
                                 "airport_code": "from",
                                 "tavg": "tavg_from",
                                 "prcp": "prcp_from",
                                 "wspd": "wspd_from"})

    weather_data_to = weather_data.rename(columns={"time": "flight_date",
                                 "airport_code": "to",
                                 "tavg": "tavg_to",
                                 "prcp": "prcp_to",
                                 "wspd": "wspd_to"})                            
    merged_from = pd.merge(X, weather_data_from, on=["flight_date", "from"])
    merged = pd.merge(merged_from, weather_data_to, on=["flight_date", "to"])
    
    return merged

def interpolate_missing_values(X, column, rename):
    X[rename] = X[column].interpolate(method='polynomial', order=2).values
    X = X.drop(labels=[column], axis=1)
    X = X.rename(columns={rename:column})
    return X
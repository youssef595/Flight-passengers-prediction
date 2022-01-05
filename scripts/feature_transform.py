import pandas as pd
from datetime import datetime

from pandas.io.sql import DatabaseError

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

def merge_weather_data(X):
    X = X.copy()
    weather_data = pd.read_csv('../data/weather_data.csv')
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











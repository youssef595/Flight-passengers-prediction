import pandas as pd
from datetime import datetime

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
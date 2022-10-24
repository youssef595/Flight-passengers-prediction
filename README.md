# Number-of-Flight-Passengers-Prediction

## The Task

The goal of the challenge is to predict the number of passengers per plane on some flights in the US. The data is provided to us by a single company. This is a supervised regression problem.

From the company point of view, the interest of this challenge is to be able to evaluate the percentage of no-show reservations, in order to properly calibrate overbooking.

Some passengers make reservation but do not show up on the flight, leading to empty seats in the plane. Estimating the number of passengers effectively boarding the plane is thus important for the company. The left-out data has dates that come after the training data, so a time series approach is possible.

## The Data
The training data is made available as a dataframe, whose columns are:

- flight_date: the flight's takeoff day
- from: the IATA code of the departure airport
- to: the IATA code of the arrival airport
- avg_weeks: average number of weeks between booking and flight date, across passengers
- std_weeks: standard deviation of number of weeks between booking and flight date, across passengers. You can think of the last two columns as being outputs of another, unknown ML pipeline.

The target variable is a transformation of the number of passengers boarding the plane, and is named `target` in the training dataframe.

The main difficulty of the challenge is the limited number of information per flight (the data is "thin", it has few columns). The participants are encouraged to enrich the data with other sources, e.g. weather, holiday calendar, etc.

## Scoring
The performance of the prediction will be quantified on left-out data, using the RMSE (Root Mean Squared Error).
The test data is available under the same format as the training data, minus the target column.

In order to be able to see the plotly plots in the EDA notebook paste the repo link and explore the notebook in this [website](https://nbviewer.org/)

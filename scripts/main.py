import pandas as pd
from feature_transform import dates_encoder, merge_path, \
    merge_temperature_data

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

X_train = pd.read_csv('data/flights_train.csv',
                      parse_dates=['flight_date'])
X_test = pd.read_csv('data/flights_Xtest.csv',
                     parse_dates=['flight_date'])

data_merger = FunctionTransformer(merge_temperature_data)

X_train = data_merger.fit_transform(X_train)
X_test = data_merger.fit_transform(X_test)

X_train = merge_path(X_train)
X_test = merge_path(X_test)

X_train = dates_encoder(X_train)
X_test = dates_encoder(X_test)

categorical_encoder = OneHotEncoder(handle_unknown='ignore',
                                    sparse=False)
categorical_cols = ['path']

preprocessor = make_column_transformer((categorical_encoder, categorical_cols),
                                       remainder='passthrough')

regressor_rf = RandomForestRegressor(n_estimators=10, max_depth=10,
                                     n_jobs=4)
regressor_lgb = LGBMRegressor(
    boosting_type='gbdt',
    class_weight=None,
    colsample_bytree=1.0,
    importance_type='split',
    learning_rate=0.1,
    max_depth=-1,
    min_child_samples=20,
    min_child_weight=0.001,
    min_split_gain=0.0,
    n_estimators=100,
    n_jobs=-1,
    num_leaves=31,
    objective=None,
    random_state=2326,
    reg_alpha=0.0,
    reg_lambda=0.0,
    silent='warn',
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=0,
    )
regressor_cb = CatBoostRegressor(loss_function='RMSE')
regressor_xgb = XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=1,
    enable_categorical=False,
    gamma=0,
    gpu_id=-1,
    importance_type=None,
    interaction_constraints='',
    learning_rate=0.30,
    max_delta_step=0,
    max_depth=6,
    min_child_weight=1,
    monotone_constraints='()',
    n_estimators=100,
    n_jobs=-1,
    num_parallel_tree=1,
    objective='reg:squarederror',
    predictor='auto',
    random_state=5589,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=1,
    tree_method='auto',
    validate_parameters=1,
    verbosity=0,
    )

pipe_line_rf = make_pipeline(preprocessor, regressor_rf)
pipe_line_lgb = make_pipeline(preprocessor, regressor_lgb)
pipe_line_cb = make_pipeline(preprocessor, regressor_cb)
pipe_line_xgb = make_pipeline(preprocessor, regressor_xgb)

vr = VotingRegressor([('pipe_cb', pipe_line_cb), ('pipe_xgb',
                     pipe_line_xgb), ('pipe_lgb', pipe_line_lgb),
                     ('pipe_rf', pipe_line_rf)])

X = X_train.drop('target', axis=1)
y = X_train.target

pred = vr.fit(X, y).predict(X_test)

submission = pd.DataFrame(pred)
submission.to_csv('submissions/FPX_submission.csv', index=False,
                  header=False)

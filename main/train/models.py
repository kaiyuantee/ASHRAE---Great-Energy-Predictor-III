import pandas as pd
import numpy as np
import keras
import keras.backend as K
from keras import Input, Model, models
from keras.layers import Dense, Dropout, Embedding, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from .utils import memory_reducer
from sklearn.preprocessing import LabelEncoder

first = [2, 3, 4, 5, 6]
second = [8, 9, 10, 11, 12]
category_cols = ['building_id', 'site_id', 'primary_use',
                 'is_holiday', 'meter', 'speed_beaufort']
feature_cols = ['square_feet', 'year_built'] + \
               ['hour', 'weekday', 'building_median'] + \
               ['air_temperature', 'cloud_coverage',
                'dew_temperature', 'precip_depth_1_hr',
                'sea_level_pressure','wind_direction',
                'wind_speed']


class LightGBM(object):

    def __init__(self, x_t, y_t, x_v, y_v):
        self.x_t = x_t
        self.y_t = y_t
        self.x_v = x_v
        self.y_v = y_v

    def model(self):
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "bagging_fraction": 0.95,
            "feature_fraction": 0.85,
            "reg_lambda": 2,
            "metric": "rmse"
        }
        d_half_1 = lgb.Dataset(self.x_t, label=self.y_t, categorical_feature=category_cols, free_raw_data=False)
        d_half_2 = lgb.Dataset(self.x_v, label=self.y_v, categorical_feature=category_cols, free_raw_data=False)
        watchlist_1 = [d_half_1, d_half_2]
        watchlist_2 = [d_half_2, d_half_1]
        print("Building model with first half and validating on second half:")
        model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1,
                                 verbose_eval=50, early_stopping_rounds=50)
        # predictions
        y_pred_valid1 = model_half_1.predict(self.x_v, num_iteration=model_half_1.best_iteration)
        print('oof score is', mean_squared_error(self.y_v, y_pred_valid1))
        models.append(model_half_1)
        print("Building model with second half and validating on first half:")
        model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2,
                                 verbose_eval=50, early_stopping_rounds=50)
        y_pred_valid2 = model_half_2.predict(self.x_t, num_iteration=model_half_2.best_iteration)
        print('oof score is', mean_squared_error(self.y_t, y_pred_valid2))
        models.append(model_half_2)
        ytrue = np.concatenate((self.y_t, self.y_v), axis=0)
        ypred = np.concatenate((y_pred_valid2, y_pred_valid1), axis=0)
        oof0 = mean_squared_error(ytrue, ypred)
        ooftotal = 0
        ooftotal += oof0 * len(ytrue)
        print('oof is', np.sqrt(oof0))


class Keras(object):

    def __init__(self, xt, yt, xv,
                 yv, dense1, dense2,
                 dense3, dense4, dropout1,
                 dropout2, dropout3, dropout4,
                 lr, batch_size, epochs,
                 patience, fold):

        self.x_t = xt
        self.y_t = yt
        self.x_v = xv
        self.yv = yv
        self.dense_dim_1 = dense1
        self.dense_dim_2 = dense2
        self.dense_dim_3 = dense3
        self.dense_dim_4 = dense4
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.fold = fold

    def body(self):
        # Inputs
        site_id = Input(shape=[1], name="site_id")
        building_id = Input(shape=[1], name="building_id")
        meter = Input(shape=[1], name="meter")
        primary_use = Input(shape=[1], name="primary_use")
        square_feet = Input(shape=[1], name="square_feet")
        year_built = Input(shape=[1], name="year_built")
        # floor_count = Input(shape=[1], name="floor_count")
        air_temperature = Input(shape=[1], name="air_temperature")
        cloud_coverage = Input(shape=[1], name="cloudcover")
        dew_temperature = Input(shape=[1], name="DewPointC")
        hour = Input(shape=[1], name="hour")
        precip = Input(shape=[1], name="precipMM")
        #wind_direction = Input(shape=[1], name="wind_direction")
        #wind_speed = Input(shape=[1], name="wind_speed")
        weekday = Input(shape=[1], name="weekday")
        beaufort_scale = Input(shape=[1], name="speed_beaufort")
        isholiday = Input(shape=[1], name='is_holiday')
        #pressure = Input(shape=[1], name='pressure')
        buildingmedian = Input(shape=[1], name='building_median')

        # Embeddings layers
        emb_site_id = Embedding(16, 2)(site_id)
        emb_building_id = Embedding(1449, 6)(building_id)
        emb_meter = Embedding(4, 2)(meter)
        emb_primary_use = Embedding(16, 2)(primary_use)
        emb_hour = Embedding(24, 3)(hour)
        emb_weekday = Embedding(7, 2)(weekday)
        emb_isholiday = Embedding(2, 2)(isholiday)
        # emb_wind_direction = Embedding(16,2)(wind_direction)

        concat_emb = concatenate([
            Flatten()(emb_site_id)
            , Flatten()(emb_building_id)
            , Flatten()(emb_meter)
            , Flatten()(emb_primary_use)
            , Flatten()(emb_hour)
            , Flatten()(emb_weekday)
            , Flatten()(emb_isholiday)
            # , Flatten() (emb_wind_direction)
        ])

        categ = Dropout(self.dropout1)(Dense(self.dense_dim_1, activation='relu') (concat_emb))
        categ = BatchNormalization()(categ)
        categ = Dropout(self.dropout2)(Dense(self.dense_dim_2, activation='relu') (categ))

        # main layer
        main_l = concatenate([
            categ
            , square_feet
            , year_built
            # , floor_count
            , air_temperature
            , cloud_coverage
            , dew_temperature
            , precip
            #, wind_direction
            #, wind_speed
            , beaufort_scale
            #, pressure
            , buildingmedian
        ])

        main_l = Dropout(self.dropout3)(Dense(self.dense_dim_3, activation='relu')(main_l))
        main_l = BatchNormalization()(main_l)
        main_l = Dropout(self.dropout4)(Dense(self.dense_dim_4, activation='relu')(main_l))

        # output
        output = Dense(1)(main_l)

        model = Model([
            site_id,
            building_id,
            meter,
            primary_use,
            square_feet,
            year_built,
            # floor_count,
            air_temperature,
            cloud_coverage,
            dew_temperature,
            hour,
            weekday,
            precip,
            #wind_direction,
            #wind_speed,
            beaufort_scale,
            isholiday,
            #pressure,
            buildingmedian], output)

        model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=MSE,
                      metrics=[root_mean_squared_error()])
        return model

    def callbacks(self):
        early_stopping = EarlyStopping(patience=self.patience,
                                       monitor='val_root_mean_squared_error',
                                       verbose=1)

        model_checkpoint = ModelCheckpoint("model_" + str(self.fold) + ".hdf5",
                                           save_best_only=True,
                                           verbose=1,
                                           monitor='val_root_mean_squared_error',
                                           mode='min')

        reducer = ReduceLROnPlateau(monitor='val_root_mean_squared_error',
                                    factor=0.5,
                                    patience=1,
                                    verbose=1,
                                    mode='auto',
                                    min_delta=0.0001,
                                    cooldown=0,
                                    min_lr=0)

        return early_stopping, model_checkpoint, reducer

    def train(self):
        model = self.body()
        hist = model.fit(self.x_t, self.y_t, batch_size=self.batch_size, epochs=self.epochs,
                         validation_data=(self.x_v, self.y_v), verbose=1,
                         callbacks=[self.callbacks])
        keras_model = models.load_model("model_" + str(self.fold) + ".hdf5",
                                        custom_objects={'root_mean_squared_error': root_mean_squared_error})
        oof = keras_model.predict(self.x_v)
        print('oof is', mean_squared_error(self.y_v, oof))
        return keras_model


def preprocess(option, df, build, weather):

    le = LabelEncoder
    cols = category_cols + feature_cols
    if option == 'train':
        df = df.merge(build, on='building_id', how='left')
        # df = df[df.site_id==siteid]
        df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
        df = memory_reducer(df)
        df['primary_use'] = le.transform(df['primary_use'])
        df = df[(df.month.isin(first))]  # &(df.site_id==siteid)]
        y = df.meter_reading  # .values
        df = df[cols]
    elif option == 'val':
        df = df.merge(build, on='building_id', how='left')
        # df = df[df.site_id==siteid]
        df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
        df = memory_reducer(df)
        df['primary_use'] = le.transform(df['primary_use'])
        df = df[(df.month.isin(second))]  # &(df.site_id==siteid)]
        y = df.meter_reading  # .values
        df = df[cols]
    return df, y


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=0)

#
# class XGBoost(object):
#
#
# class Catboost()

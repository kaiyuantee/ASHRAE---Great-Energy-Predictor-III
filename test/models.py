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

first = [2, 3, 4, 5, 6]
second = [8, 9, 10, 11, 12]
category_cols = ['building_id', 'site_id', 'primary_use', 'is_holiday', 'meter', 'speed_beaufort']
feature_cols = ['square_feet', 'year_built'] + ['hour', 'weekday', 'building_median'] + [
    'air_temperature', 'cloudcover',
    'DewPointC', 'precipMM', 'pressure',
    'wind_direction', 'wind_speed']


class LightGBM():
    #def __init__(self):

    #@staticmethod
    def model(self, x_t, y_t, x_v, y_v):
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
        d_half_1 = lgb.Dataset(x_t, label=y_t, categorical_feature=category_cols, free_raw_data=False)
        d_half_2 = lgb.Dataset(x_v, label=y_v, categorical_feature=category_cols, free_raw_data=False)
        watchlist_1 = [d_half_1, d_half_2]
        watchlist_2 = [d_half_2, d_half_1]
        print("Building model with first half and validating on second half:")
        model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1,
                                 verbose_eval=50, early_stopping_rounds=50)
        # predictions
        y_pred_valid1 = model_half_1.predict(x_v, num_iteration=model_half_1.best_iteration)
        print('oof score is', mean_squared_error(y_v, y_pred_valid1))
        models.append(model_half_1)
        print("Building model with second half and validating on first half:")
        model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2,
                                 verbose_eval=50, early_stopping_rounds=50)
        y_pred_valid2 = model_half_2.predict(x_t, num_iteration=model_half_2.best_iteration)
        print('oof score is', mean_squared_error(y_t, y_pred_valid2))
        models.append(model_half_2)
        ytrue = np.concatenate((y_t, y_v), axis=0)
        ypred = np.concatenate((y_pred_valid2, y_pred_valid1), axis=0)
        oof0 = mean_squared_error(ytrue, ypred)
        ooftotal = 0
        ooftotal += oof0 * len(ytrue)
        print('oof is', np.sqrt(oof0))


#class XGBoost():



#class CatBoost():




class Keras():
    def __init__(self, category_cols, feature_cols):
        self.cat_col = category_cols
        self.feat_col = feature_cols
        self.columns = self.cat_col + self.feat_col

    @staticmethod
    def body(self,
             dense_dim_1,
             dense_dim_2,
             dense_dim_3,
             dense_dim_4,
             dropout1,
             dropout2,
             dropout3,
             dropout4,
             lr):
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
        wind_direction = Input(shape=[1], name="wind_direction")
        wind_speed = Input(shape=[1], name="wind_speed")
        weekday = Input(shape=[1], name="weekday")
        beaufort_scale = Input(shape=[1], name="speed_beaufort")
        isholiday = Input(shape=[1], name='is_holiday')
        pressure = Input(shape=[1], name='pressure')
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

        categ = Dropout(dropout1)(Dense(dense_dim_1, activation='relu')(concat_emb))
        categ = BatchNormalization()(categ)
        categ = Dropout(dropout2)(Dense(dense_dim_2, activation='relu')(categ))

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
            , wind_direction
            , wind_speed
            , beaufort_scale
            , pressure
            , buildingmedian
        ])

        main_l = Dropout(dropout3)(Dense(dense_dim_3, activation='relu')(main_l))
        main_l = BatchNormalization()(main_l)
        main_l = Dropout(dropout4)(Dense(dense_dim_4, activation='relu')(main_l))

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
            wind_direction,
            wind_speed,
            beaufort_scale,
            isholiday,
            pressure,
            buildingmedian], output)

        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=MSE,
                      metrics=[root_mean_squared_error()])
        return model


    @staticmethod
    def callbacks(self, patience, fold):
        early_stopping = EarlyStopping(patience=patience,
                                       monitor='val_root_mean_squared_error',
                                       verbose=1)

        model_checkpoint = ModelCheckpoint("model_" + str(fold) + ".hdf5",
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

    @staticmethod
    def train(self, fold, x_t, y_t, x_v, y_v, batch_size, epochs):
        model = self.body()
        hist = model.fit(x_t, y_t, batch_size=batch_size, epochs=epochs,
                               validation_data=(x_v, y_v), verbose=1,
                               callbacks=[self.callbacks])
        keras_model = models.load_model("model_" + str(fold) + ".hdf5",
                                        custom_objects={'root_mean_squared_error': root_mean_squared_error})
        oof = keras_model.predict(x_v)
        print('oof is', mean_squared_error(y_v, oof))
        return keras_model


def preprocess(option,df,build,weather):

    # 'air_temperature_mean_lag72',
    # 'air_temperature_max_lag72',
    # 'air_temperature_min_lag72',
    # 'air_temperature_std_lag72',
    # 'cloudcover_mean_lag72',
    # 'DewPointC_mean_lag72',
    # 'precipMM_mean_lag72',
    # 'pressure_mean_lag72',
    # 'wind_direction_mean_lag72',
    # 'wind_speed_mean_lag72',
    # 'air_temperature_mean_lag3',
    # 'air_temperature_max_lag3',
    # 'air_temperature_min_lag3',
    # 'cloudcover_mean_lag3',
    # 'DewPointC_mean_lag3',
    # 'precipMM_mean_lag3',
    # 'pressure_mean_lag3',
    # 'wind_direction_mean_lag3',
    # 'wind_speed_mean_lag3']
    cols = category_cols + feature_cols
    if option == 'train':
        df = df.merge(build, on='building_id', how='left')
        # df = df[df.site_id==siteid]
        df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
        df = df[(df.month.isin(first))]  # &(df.site_id==siteid)]
        y = df.meter_reading  # .values
        df = df[cols]
    elif option == 'val':
        df = df.merge(build, on='building_id', how='left')
        # df = df[df.site_id==siteid]
        df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
        df = df[(df.month.isin(second))]  # &(df.site_id==siteid)]
        y = df.meter_reading  # .values
        df = df[cols]
    return df, y


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=0)

# if __name__ == '__main__':
#
#     if args.keras:



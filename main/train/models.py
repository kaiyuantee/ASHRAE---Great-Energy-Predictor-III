import numpy as np
import keras.backend as K
from keras import Input, Model, models
from keras.layers import Dense, Dropout, Embedding, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.losses import MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import lightgbm as lgb
import xgboost as xgb
import gc


class LightGBM(object):

    def __init__(self, x_t, y_t, x_v, y_v):

        self.x_t = x_t[cols]
        self.y_t = y_t
        self.x_v = x_v[cols]
        self.y_v = y_v
        self.train()

    def train(self):
        all_models = []
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
        all_models.append(model_half_1)
        print("Building model with second half and validating on first half:")
        model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2,
                                 verbose_eval=50, early_stopping_rounds=50)
        y_pred_valid2 = model_half_2.predict(self.x_t, num_iteration=model_half_2.best_iteration)
        print('oof score is', mean_squared_error(self.y_t, y_pred_valid2))
        all_models.append(model_half_2)
        ytrue = np.concatenate((self.y_t, self.y_v), axis=0)
        ypred = np.concatenate((y_pred_valid2, y_pred_valid1), axis=0)
        oof0 = mean_squared_error(ytrue, ypred)
        ooftotal = 0
        ooftotal += oof0 * len(ytrue)
        print('oof is', np.sqrt(oof0))
        gc.collect()


class XGBoost(object):

    def __init__(self, x_t, y_t, x_v, y_v):
        self.x_t = x_t[cols]
        self.y_t = y_t
        self.x_v = x_v[cols]
        self.y_v = y_v
        self.train()

    def train(self):
        print('\n...Training Now...')
        # TODO: make a list of arguments to pass?
        reg = xgb.XGBRegressor(n_estimators=5000,
                               eta=0.005,
                               subsample=1,
                               tree_method='gpu_hist',
                               max_depth=13,
                               objective='reg:squarederror',
                               reg_lambda=2
                               # num_boost_round=1000
                               )

        hist1 = reg.fit(self.x_t,
                        self.y_t,
                        eval_set=[(self.x_v, self.y_v)],
                        eval_metric='rmse',
                        verbose=20,
                        early_stopping_rounds=50)

        oof1 = hist1.predict(self.x_v)
        print('*' * 20)
        print('oof score is', mean_squared_error(self.y_v, oof1))
        print('*' * 20)
        gc.collect()


class Keras(object):

    def __init__(self, x_t, y_t, x_v, y_v, **kwargs):
        self.x_t = {col: x_t[col] for col in cols}
        self.x_v = {col: x_v[col] for col in cols}
        self.y_t = y_t
        self.y_v = y_v

        keys = ['dense_dim_1', 'dense_dim_2',
                'dense_dim_3', 'dense_dim_4',
                'dropout1', 'dropout2',
                'dropout3', 'dropout4',
                'lr', 'batch_size', 'epochs',
                'patience', 'fold']

        for key in keys:
            setattr(self, key, kwargs.get(key))

        self.train()

    def body(self):
        # Inputs
        site_id = Input(shape=[1], name="site_id")
        building_id = Input(shape=[1], name="building_id")
        meter = Input(shape=[1], name="meter")
        primary_use = Input(shape=[1], name="primary_use")
        square_feet = Input(shape=[1], name="square_feet")
        year_built = Input(shape=[1], name="year_built")
        air_temperature = Input(shape=[1], name="air_temperature")
        cloud_coverage = Input(shape=[1], name="cloud_coverage")
        dew_temperature = Input(shape=[1], name="dew_temperature")
        hour = Input(shape=[1], name="hour")
        precip = Input(shape=[1], name="precip_depth_1_hr")
        wind_direction = Input(shape=[1], name="wind_direction")
        wind_speed = Input(shape=[1], name="wind_speed")
        weekday = Input(shape=[1], name="weekday")
        beaufort_scale = Input(shape=[1], name="speed_beaufort")
        isholiday = Input(shape=[1], name='is_holiday')
        pressure = Input(shape=[1], name='sea_level_pressure')
        buildingmedian = Input(shape=[1], name='building_median')

        # Embeddings layers
        emb_site_id = Embedding(16, 2)(site_id)
        emb_building_id = Embedding(1449, 6)(building_id)
        emb_meter = Embedding(4, 2)(meter)
        emb_primary_use = Embedding(16, 2)(primary_use)
        emb_hour = Embedding(24, 3)(hour)
        emb_weekday = Embedding(7, 2)(weekday)
        emb_isholiday = Embedding(2, 2)(isholiday)
        emb_wind_direction = Embedding(16, 2)(wind_direction)

        concat_emb = concatenate([
            Flatten()(emb_site_id)
            , Flatten()(emb_building_id)
            , Flatten()(emb_meter)
            , Flatten()(emb_primary_use)
            , Flatten()(emb_hour)
            , Flatten()(emb_weekday)
            , Flatten()(emb_isholiday)
            , Flatten()(emb_wind_direction)
        ])

        categ = Dropout(self.dropout1)(Dense(self.dense_dim_1, activation='relu')(concat_emb))
        categ = BatchNormalization()(categ)
        categ = Dropout(self.dropout2)(Dense(self.dense_dim_2, activation='relu')(categ))

        # main layer
        main_l = concatenate([
            categ
            , square_feet
            , year_built
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

        model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=MSE,
                      metrics=[root_mean_squared_error])
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
        cb1, cb2, cb3 = self.callbacks()
        hist = model.fit(self.x_t, self.y_t, batch_size=self.batch_size, epochs=self.epochs,
                         validation_data=(self.x_v, self.y_v), verbose=1,
                         callbacks=[cb1, cb2, cb3])
        keras_model = models.load_model("model_" + str(self.fold) + ".hdf5",
                                        custom_objects={'root_mean_squared_error': root_mean_squared_error})
        oof = keras_model.predict(self.x_v)
        print('oof is', mean_squared_error(self.y_v, oof))
        gc.collect()
        return keras_model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=0)



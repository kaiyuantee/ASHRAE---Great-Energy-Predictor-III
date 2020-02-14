import numpy as np
import pandas as pd
import os
import keras.backend as K
from keras import Input, Model, models
from keras.layers import Dense, Dropout, Embedding, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.losses import mean_squared_error as mse_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import gc
import pickle
from tqdm import tqdm
from .utils import OUTPUT_ROOT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

category_cols = ["building_id",
                 "primary_use",
                 "meter",
                 "weekday",
                 "hour",
                 'is_holiday']


def columns(df):
    all_features = [col for col in df.columns if col not in ["timestamp", "site_id", "meter_reading", "row_id"]]
    return df[all_features]


def kerascolumns(df):
    all_features = [col for col in df.columns if col not in ["timestamp", "meter_reading", "row_id"]]
    return df[all_features]


class LightGBM(object):

    def __init__(self, x, fold, objective='regression',
                 boosting='gbdt', metric='rmse',
                 num_leaves=31, lr=0.05, bagg_freq=5,
                 bagg_frac=0.95, feature_frac=0.85,
                 reg_lambda=2, seed=555, num_boost_round=1000,
                 verbose_eval=50, early_stopping=50):

        self.x = x
        self.fold = fold
        self.seed = seed
        self.objective = objective
        self.boosting = boosting
        self.num_leaves = num_leaves
        self.bagg_freq = bagg_freq
        self.lr = lr
        self.bagg_frac = bagg_frac
        self.feature_frac = feature_frac
        self.reg_lambda = reg_lambda
        self.metric = metric
        self.cat_cols = category_cols
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.early_stopping = early_stopping
        self.train()

    def transform(self, i):

        x = self.x[self.x.site_id == i].reset_index(drop=True)
        y = x.meter_reading
        x = columns(x)

        return x, y

    def train(self):

        all_models = {}
        cv_scores = {"site_id": [], "cv_score": []}

        for i in tqdm(range(16)):

            x, y = self.transform(i)
            scores = 0
            all_models[i] = []
            y_pred_train_site = np.zeros(x.shape[0])
            kf = KFold(n_splits=self.fold, random_state=self.seed)
            for fold, (train_index, valid_index) in enumerate(kf.split(x, y)):
                x_t, x_v = x.iloc[train_index], x.iloc[valid_index]
                y_t, y_v = y.iloc[train_index], y.iloc[valid_index]
                params = {
                    "objective": self.objective,
                    "boosting": self.boosting,
                    "num_leaves": self.num_leaves,
                    "learning_rate": self.lr,
                    'bagging_freq': self.bagg_freq,
                    "bagging_fraction": self.bagg_frac,
                    "feature_fraction": self.feature_frac,
                    "reg_lambda": self.reg_lambda,
                    "metric": self.metric
                }
                d_half_1 = lgb.Dataset(x_t, label=y_t, categorical_feature=self.cat_cols, free_raw_data=False)
                d_half_2 = lgb.Dataset(x_v, label=y_v, categorical_feature=self.cat_cols, free_raw_data=False)
                watchlist = [d_half_1, d_half_2]
                print("Building model in fold 1")
                model_half_1 = lgb.train(params, train_set=d_half_1,
                                         num_boost_round=self.num_boost_round,
                                         valid_sets=watchlist, verbose_eval=self.verbose_eval,
                                         early_stopping_rounds=self.early_stopping)
                # predictions
                y_pred_valid1 = model_half_1.predict(x_v, num_iteration=model_half_1.best_iteration)
                y_pred_train_site[valid_index] = y_pred_valid1
                rmse1 = np.sqrt(mean_squared_error(y_v, y_pred_valid1))
                print('SiteID number :', i, 'Fold:', fold + 1, 'RMSE', rmse1)
                scores += rmse1 / 2
                all_models[i].append(model_half_1)
                gc.collect()

            oof0 = mean_squared_error(y, y_pred_train_site)
            cv_scores['site_id'].append(i)
            cv_scores['cv_score'].append(scores)
            print('Site_ID:', i, 'CV_RMSE:', np.sqrt(oof0))
            gc.collect()
        with open(OUTPUT_ROOT / 'lgbm_allmodels.p', 'wb') as output_file:
            pickle.dump(all_models, output_file)
        print(pd.DataFrame.from_dict(cv_scores))


# class XGBoost(object):
#
#     def __init__(self, x_t, y_t, x_v, y_v):
#         self.x_t = x_t[cols]
#         self.y_t = y_t
#         self.x_v = x_v[cols]
#         self.y_v = y_v
#         self.train()
#
#     def train(self):
#         print('\n...Training Now...')
#         # TODO: make a list of arguments to pass?
#         reg = xgb.XGBRegressor(n_estimators=5000,
#                                eta=0.005,
#                                subsample=1,
#                                tree_method='gpu_hist',
#                                max_depth=13,
#                                objective='reg:squarederror',
#                                reg_lambda=2
#                                # num_boost_round=1000
#                                )
#
#         hist1 = reg.fit(self.x_t,
#                         self.y_t,
#                         eval_set=[(self.x_v, self.y_v)],
#                         eval_metric='rmse',
#                         verbose=20,
#                         early_stopping_rounds=50)
#
#         oof1 = hist1.predict(self.x_v)
#         print('*' * 20)
#         print('oof score is', mean_squared_error(self.y_v, oof1))
#         print('*' * 20)
#         gc.collect()


class CatBoost(object):
    def __init__(self, x):
        self.x = x
        self.train()

    # def train(self):
    #     for i in tqdm(range(4)):
    #         x_t = self.x_t['meter' == i]
    #         y_t = x_t.meter_reading
    #         x_v = self.x_v['meter' == i]
    #         y_v = x_v.meter_reading
    #         model_filename = 'catboost'
    #         all_models = []
    #
    #         cat_params = {
    #             'n_estimators': 2000,
    #             'learning_rate': 0.1,
    #             'eval_metric': 'RMSE',
    #             'loss_function': 'RMSE',
    #             'metric_period': 10,
    #             'task_type': 'GPU',
    #             'early_stopping_rounds': 100,
    #             'depth': 8,
    #         }
    #
    #         estimator = CatBoostRegressor(**cat_params)
    #         estimator.fit(
    #             x_t, y_t,
    #             eval_set=[x_v, y_v],
    #             cat_features=category_cols,
    #             use_best_model=True,
    #             verbose=True)
    #
    #         estimator.save_model(model_filename + '.bin')
    #         all_models.append(model_filename + '.bin')
    #
    #         del estimator
    #         gc.collect()

    def transform(self, i):

        x = self.x[self.x.meter == i].reset_index(drop=True)
        y = x.meter_reading
        x = columns(x)

        return x, y

    def train(self):

        all_models = {}
        cv_scores = {"meter": [], "cv_score": []}

        for i in tqdm(range(4)):

            x, y = self.transform(i)
            scores = 0
            all_models[i] = []
            y_pred_train_site = np.zeros(x.shape[0])
            kf = KFold(n_splits=2, random_state=555)
            for fold, (train_index, valid_index) in enumerate(kf.split(x, y)):
                x_t, x_v = x.iloc[train_index], x.iloc[valid_index]
                y_t, y_v = y.iloc[train_index], y.iloc[valid_index]
                cat_params = {
                    'n_estimators': 2000,
                    'learning_rate': 0.1,
                    'eval_metric': 'RMSE',
                    'loss_function': 'RMSE',
                    'metric_period': 10,
                    'task_type': 'GPU',
                    'early_stopping_rounds': 100,
                    'depth': 8,
                }
                print('building some shit now')
                estimator = CatBoostRegressor(**cat_params)
                catmodel = estimator.fit(
                    x_t, y_t,
                    eval_set=[x_v, y_v],
                    cat_features=category_cols,
                    use_best_model=True,
                    verbose=True)
                # predictions

                y_pred_valid1 = catmodel.predict([x_v, y_v])
                y_pred_train_site[valid_index] = y_pred_valid1
                rmse1 = np.sqrt(mean_squared_error(y_v, y_pred_valid1))
                print('SiteID number :', i, 'Fold:', fold + 1, 'RMSE', rmse1)
                scores += rmse1 / 2
                all_models[i].append(catmodel)
                gc.collect()

            oof0 = mean_squared_error(y, y_pred_train_site)
            cv_scores['meter'].append(i)
            cv_scores['cv_score'].append(scores)
            print('Meter:', i, 'CV_RMSE:', np.sqrt(oof0))
            gc.collect()
        with open(OUTPUT_ROOT / 'catboost_allmodels.p', 'wb') as output_file:
            pickle.dump(all_models, output_file)
        print(pd.DataFrame.from_dict(cv_scores))


class Keras(object):

    def __init__(self, x, dense_dim_1=64, dense_dim_2=32,
                 dense_dim_3=32, dense_dim_4=16, dropout1=0.2,
                 dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.005,
                 batch_size=2048, epochs=10, patience=2, fold=2):
        self.x = x
        self.dense_dim_1 = dense_dim_1
        self.dense_dim_2 = dense_dim_2
        self.dense_dim_3 = dense_dim_3
        self.dense_dim_4 = dense_dim_4
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.fold = fold

        keys = ['dense_dim_1', 'dense_dim_2',
                'dense_dim_3', 'dense_dim_4',
                'dropout1', 'dropout2',
                'dropout3', 'dropout4',
                'lr', 'batch_size', 'epochs',
                'patience', 'fold']

        # for key in keys:
        #     setattr(self, key, kwargs.get(key))

        self.train()

    def body(self):
        # Inputs
        site_id = Input(shape=[1], name="site_id")
        building_id = Input(shape=[1], name="building_id")
        meter = Input(shape=[1], name="meter")
        primary_use = Input(shape=[1], name="primary_use")
        square_feet = Input(shape=[1], name="square_feet")
        air_temperature = Input(shape=[1], name="air_temperature")
        cloud_coverage = Input(shape=[1], name="cloud_coverage")
        dew_temperature = Input(shape=[1], name="dew_temperature")
        hour = Input(shape=[1], name="hour")
        weekday = Input(shape=[1], name="weekday")
        isholiday = Input(shape=[1], name='is_holiday')
        buildingmedian = Input(shape=[1], name='building_median')
        #         air_temperature_mean_lag18 = Input(shape=[1], name='air_temperature_mean_lag18')
        #         air_temperature_max_lag18 = Input(shape=[1], name='air_temperature_max_lag18')
        #         air_temperature_min_lag18 = Input(shape=[1], name='air_temperature_min_lag18')
        #         air_temperature_median_lag18 = Input(shape=[1], name='air_temperature_median_lag18')
        #         air_temperature_std_lag18 = Input(shape=[1], name='air_temperature_std_lag18')
        #         air_temperature_skew_lag18 = Input(shape=[1], name='air_temperature_skew_lag18')
        #         dew_temperature_mean_lag18 = Input(shape=[1], name='dew_temperature_mean_lag18')
        #         dew_temperature_max_lag18 = Input(shape=[1], name='dew_temperature_max_lag18')
        #         dew_temperature_min_lag18 = Input(shape=[1], name='dew_temperature_min_lag18')
        #         dew_temperature_median_lag18 = Input(shape=[1], name='dew_temperature_median_lag18')
        #         dew_temperature_std_lag18 = Input(shape=[1], name='dew_temperature_std_lag18')
        #         dew_temperature_skew_lag18 = Input(shape=[1], name='dew_temperature_skew_lag18')
        #         cloud_coverage_mean_lag18 = Input(shape=[1], name='cloud_coverage_mean_lag18')
        #         cloud_coverage_max_lag18 = Input(shape=[1], name='cloud_coverage_max_lag18')
        #         cloud_coverage_min_lag18 = Input(shape=[1], name='cloud_coverage_min_lag18')
        #         cloud_coverage_median_lag18 = Input(shape=[1], name='cloud_coverage_median_lag18')
        #         cloud_coverage_std_lag18 = Input(shape=[1], name='cloud_coverage_std_lag18')
        #         cloud_coverage_skew_lag18 = Input(shape=[1], name='cloud_coverage_skew_lag18')
        #         mean_building_meter_x = Input(shape=[1], name='mean_building_meter_x')
        #         mean_building_meter_y = Input(shape=[1], name='mean_building_meter_y')
        #         median_building_meter_x = Input(shape=[1], name='median_building_meter_x')
        #         median_building_meter_y = Input(shape=[1], name='median_building_meter_y')

        # Embeddings layers
        emb_site_id = Embedding(16, 2)(site_id)
        emb_building_id = Embedding(1449, 6)(building_id)
        emb_meter = Embedding(4, 2)(meter)
        emb_primary_use = Embedding(16, 2)(primary_use)
        emb_hour = Embedding(24, 3)(hour)
        emb_weekday = Embedding(7, 2)(weekday)
        emb_isholiday = Embedding(2, 2)(isholiday)

        concat_emb = concatenate([
            Flatten()(emb_site_id)
            , Flatten()(emb_building_id)
            , Flatten()(emb_meter)
            , Flatten()(emb_primary_use)
            , Flatten()(emb_hour)
            , Flatten()(emb_weekday)
            , Flatten()(emb_isholiday)
        ])

        categ = Dropout(self.dropout1)(Dense(self.dense_dim_1, activation='relu')(concat_emb))
        categ = BatchNormalization()(categ)
        categ = Dropout(self.dropout2)(Dense(self.dense_dim_2, activation='relu')(categ))

        # main layer
        main_l = concatenate([
            categ
            , square_feet
            , air_temperature
            , cloud_coverage
            , dew_temperature
            , buildingmedian
            #             , air_temperature_mean_lag18
            #             , air_temperature_max_lag18
            #             , air_temperature_min_lag18
            #             , air_temperature_median_lag18
            #             , air_temperature_std_lag18
            #             , air_temperature_skew_lag18
            #             , dew_temperature_mean_lag18
            #             , dew_temperature_max_lag18
            #             , dew_temperature_min_lag18
            #             , dew_temperature_median_lag18
            #             , dew_temperature_std_lag18
            #             , dew_temperature_skew_lag18
            #             , cloud_coverage_mean_lag18
            #             , cloud_coverage_max_lag18
            #             , cloud_coverage_min_lag18
            #             , cloud_coverage_median_lag18
            #             , cloud_coverage_std_lag18
            #             , cloud_coverage_skew_lag18
            #             , mean_building_meter_x
            #             , mean_building_meter_y
            #             , median_building_meter_x
            #             , median_building_meter_y
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
            air_temperature,
            cloud_coverage,
            dew_temperature,
            hour,
            weekday,
            isholiday,
            buildingmedian,
            #             air_temperature_mean_lag18,
            #             air_temperature_max_lag18,
            #             air_temperature_min_lag18,
            #             air_temperature_median_lag18,
            #             air_temperature_std_lag18,
            #             air_temperature_skew_lag18,
            #             dew_temperature_mean_lag18,
            #             dew_temperature_max_lag18,
            #             dew_temperature_min_lag18,
            #             dew_temperature_median_lag18,
            #             dew_temperature_std_lag18,
            #             dew_temperature_skew_lag18,
            #             cloud_coverage_mean_lag18,
            #             cloud_coverage_max_lag18,
            #             cloud_coverage_min_lag18,
            #             cloud_coverage_median_lag18,
            #             cloud_coverage_std_lag18,
            #             cloud_coverage_skew_lag18,
            #             mean_building_meter_x,
            #             mean_building_meter_y,
            #             median_building_meter_x,
            #             median_building_meter_y
        ], output)

        model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=mse_loss,
                      metrics=[root_mean_squared_error])
        return model

    def callbacks(self, fold):
        early_stopping = EarlyStopping(patience=self.patience,
                                       monitor='val_root_mean_squared_error',
                                       verbose=1)

        model_checkpoint = ModelCheckpoint(filepath=OUTPUT_ROOT / f'model_{fold}.hdf5',
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
        x_t = self.x
        y_t = x_t.meter_reading
        x_t = kerascolumns(x_t)
        kf = KFold(n_splits=self.fold)
        all_models = []
        ypred_all = np.zeros(x_t.shape[0])
        scores = 0
        for fold, (train_idx, valid_idx) in enumerate(kf.split(x_t, y_t)):
            early_stopping = EarlyStopping(patience=self.patience,
                                           monitor='val_root_mean_squared_error',
                                           verbose=1)

            model_checkpoint = ModelCheckpoint(f'model_{fold}.hdf5',
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
            x_train, x_valid = x_t.iloc[train_idx], x_t.iloc[valid_idx]
            y_train, y_valid = y_t.iloc[train_idx], y_t.iloc[valid_idx]
            x_train = {col: np.array(x_train[col]) for col in x_train.columns}
            x_valid = {col: np.array(x_valid[col]) for col in x_valid.columns}

            hist = model.fit(x_train, y_train,
                             batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_data=(x_valid, y_valid),
                             verbose=1,
                             callbacks=[early_stopping, model_checkpoint, reducer])
            keras_model = models.load_model(OUTPUT_ROOT / f'model_{fold}.hdf5',
                                            custom_objects={'root_mean_squared_error': root_mean_squared_error})
            y_pred = keras_model.predict(x_valid)
            ypred_all[valid_idx] = y_pred
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            print('Fold:', fold + 1, 'RMSE', rmse)
            scores += rmse / 2
            all_models.append(keras_model)
            gc.collect()

        oof0 = mean_squared_error(y_t, ypred_all)
        print('The scores are', scores)
        print('CV_RMSE:', np.sqrt(oof0))
        gc.collect()
        with open(OUTPUT_ROOT / 'keras_allmodels.p', 'wb') as output_file:
            pickle.dump(all_models, output_file)


def prediction(datadf, model, folds):
    if model == 'lightgbm':
        df_test_sites = []

        for i in tqdm(range(16)):

            print("Preparing test data for site_id", i)
            df = datadf[datadf.site_id == i]
            row_ids_site = df.row_id
            df = columns(df)
            y_pred_test_site = np.zeros(df.shape[0])
            with open(OUTPUT_ROOT / 'lgbm_allmodels.p', 'rb') as input_file:
                lgbm_allmodels = pickle.load(input_file)
            print("Predicting for site_id", i)

            for fold in range(folds):
                model_lgb = lgbm_allmodels[i][fold]
                y_pred_test_site += model_lgb.predict(df, num_iteration=model_lgb.best_iteration) / folds
                gc.collect()

            df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
            df_test_sites.append(df_test_site)

            print("Prediction for site_id", i, "completed")
            gc.collect()

        submission = pd.concat(df_test_sites)
        submission.meter_reading = np.clip(np.expm1(submission.meter_reading), 0, a_max=None)
        return submission.to_csv(OUTPUT_ROOT / "lgbm_prediction.csv", index=False)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=0)


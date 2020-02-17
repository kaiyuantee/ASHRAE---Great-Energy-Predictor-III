import numpy as np
import pandas as pd
import os
import keras.backend as K
from keras import Input, Model, models
from keras.layers import Dense, Dropout, Embedding, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.losses import MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc
import pickle
from tqdm import tqdm
from ..directories import OUTPUT_ROOT, LGBM_ROOT, KERAS_ROOT, CATBOOST_ROOT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

category_cols = ["building_id",
                 "primary_use",
                 "meter",
                 "weekday",
                 "hour",
                 'is_holiday']

catboostcols = ["building_id",
                "primary_use",
                "weekday",
                "hour",
                'is_holiday']


def lgbmcolumns(df):

    all_features = [col for col in df.columns if col not in ["timestamp", "site_id", "meter_reading", "row_id"]]
    return df[all_features]


def catcolumns(df):

    all_features = [col for col in df.columns if col not in ["timestamp", "meter", "meter_reading", "row_id"]]
    return df[all_features]


def kerascolumns(df):
    all_features = [col for col in df.columns if col not in ["timestamp", "meter_reading", "row_id"]]
    return df[all_features]


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


class LightGBM(object):

    def __init__(self, x, fold, lr=0.01, objective='regression',
                 boosting='gbdt', metric='rmse', num_leaves=31,
                 bagg_freq=5, bagg_frac=0.95, feature_frac=0.85,
                 reg_lambda=2, num_boost_round=1000,
                 verbose_eval=50, early_stopping=50):

        self.x = x
        self.objective = objective
        self.boosting = boosting
        self.num_leaves = num_leaves
        self.bagg_freq = bagg_freq
        self.lr = float(lr)
        self.bagg_frac = bagg_frac
        self.feature_frac = feature_frac
        self.reg_lambda = reg_lambda
        self.metric = metric
        self.cat_cols = category_cols
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.early_stopping = early_stopping
        self.fold = int(fold)

    def transform(self, i):

        x = self.x[self.x.site_id == i].reset_index(drop=True)
        y = x.meter_reading
        x = lgbmcolumns(x)

        return x, y

    def train(self):

        print('LightGBM Model')
        all_models = {}
        cv_scores = {"site_id": [], "cv_score": []}

        for i in tqdm(range(16)):
            print('Site_ID:', i)
            x, y = self.transform(i)
            scores = 0
            all_models[i] = []
            y_pred_train_site = np.zeros(x.shape[0])
            kf = KFold(n_splits=self.fold, shuffle=False)

            for fold, (train_index, valid_index) in enumerate(kf.split(x, y)):
                print('Fold:', fold)
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
                modelz = lgb.train(params, train_set=d_half_1,
                                   num_boost_round=self.num_boost_round,
                                   valid_sets=watchlist, verbose_eval=self.verbose_eval,
                                   early_stopping_rounds=self.early_stopping)
                # predictions
                y_pred = modelz.predict(x_v, num_iteration=modelz.best_iteration)
                y_pred_train_site[valid_index] = y_pred
                rmse = np.sqrt(mean_squared_error(y_v, y_pred))
                print('SiteID number :', i, 'Fold:', fold + 1, 'RMSE', rmse)
                scores += rmse / 2
                all_models[i].append(modelz)
                gc.collect()

            oof0 = mean_squared_error(y, y_pred_train_site)
            cv_scores['site_id'].append(i)
            cv_scores['cv_score'].append(scores)
            print('Site_ID:', i, 'CV_RMSE:', np.sqrt(oof0))
            gc.collect()
        with open(LGBM_ROOT / 'lgbm_allmodels.p', 'wb') as output_file:
            pickle.dump(all_models, output_file)
        print(pd.DataFrame.from_dict(cv_scores))

    def predict(self):

        df_test_sites = []

        for i in tqdm(range(16)):

            df = self.x[self.x.site_id == i]
            row_ids_site = df.row_id
            df = lgbmcolumns(df)
            y_pred_test_site = np.zeros(df.shape[0])
            with open(LGBM_ROOT / 'lgbm_allmodels.p', 'rb') as input_file:
                lgbm_allmodels = pickle.load(input_file)
            print("Predicting for site_id", i)

            for fold in range(self.fold):
                model_lgb = lgbm_allmodels[i][fold]
                y_pred_test_site += model_lgb.predict(df, num_iteration=model_lgb.best_iteration) / self.fold
                gc.collect()

            df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
            df_test_sites.append(df_test_site)

            print("Prediction for site_id", i, "completed")
            gc.collect()

        submission = pd.concat(df_test_sites)
        submission.meter_reading = np.clip(np.expm1(submission.meter_reading), 0, a_max=None)
        submission.to_csv(OUTPUT_ROOT / "lgbm_prediction.csv", index=False)


class CatBoost(object):
    def __init__(self, x, fold, lr=0.01, n_estimators=2000,
                 metric_period=10, early_stopping=100, depth=8):

        self.x = x
        self.fold = int(fold)
        self.n_estimators = n_estimators
        self.lr = float(lr)
        self.metric_period = metric_period
        self.early_stopping = early_stopping
        self.depth = depth
        self.cat_cols = catboostcols

    def transform(self, i):

        x = self.x[self.x.meter == i].reset_index(drop=True)
        y = x.meter_reading
        x = catcolumns(x)

        return x, y

    def train(self):

        print('CatBoost Model')
        all_models = {}
        cv_scores = {"meter": [], "cv_score": []}

        for i in tqdm(range(4)):
            print('\nMeter:', i)
            x, y = self.transform(i)
            scores = 0
            all_models[i] = []
            y_pred_train_site = np.zeros(x.shape[0])
            kf = KFold(n_splits=self.fold, shuffle=False)
            for fold, (train_index, valid_index) in enumerate(kf.split(x, y)):
                print('\nFold:', fold)
                x_t, x_v = x.iloc[train_index], x.iloc[valid_index]
                y_t, y_v = y.iloc[train_index], y.iloc[valid_index]
                cat_params = {
                    'n_estimators': self.n_estimators,
                    'learning_rate': self.lr,
                    'eval_metric': 'RMSE',
                    'loss_function': 'RMSE',
                    'metric_period': self.metric_period,
                    'task_type': 'GPU',
                    'devices': '0:1',
                    'early_stopping_rounds': self.early_stopping,
                    'depth': self.depth,
                }
                estimator = CatBoostRegressor(**cat_params)
                catmodel = estimator.fit(
                    x_t, y_t,
                    eval_set=(x_v, y_v),
                    cat_features=self.cat_cols,
                    use_best_model=True,
                    verbose=True)
                # predictions
                y_pred_valid1 = catmodel.predict(x_v)
                y_pred_train_site[valid_index] = y_pred_valid1
                rmse1 = np.sqrt(mean_squared_error(y_v, y_pred_valid1))
                print('Meter :', i, 'Fold:', fold + 1, 'RMSE', rmse1)
                scores += rmse1 / 2
                all_models[i].append(catmodel)
                gc.collect()

            oof0 = mean_squared_error(y, y_pred_train_site)
            cv_scores['meter'].append(i)
            cv_scores['cv_score'].append(scores)
            print('Meter:', i, 'CV_RMSE:', np.sqrt(oof0))
            gc.collect()
        with open(CATBOOST_ROOT / 'catboost_allmodels.p', 'wb') as output_file:
            pickle.dump(all_models, output_file)
        print(pd.DataFrame.from_dict(cv_scores))

    def predict(self):

        df_test_sites = []

        for i in tqdm(range(4)):

            df = self.x[self.x.meter == i]
            row_ids_site = df.row_id
            df = catcolumns(df)
            y_pred_test_site = np.zeros(df.shape[0])
            with open(CATBOOST_ROOT / 'catboost_allmodels.p', 'rb') as input_file:
                cat_allmodels = pickle.load(input_file)
            print("\nPredicting for meter", i)

            for fold in range(self.fold):
                model_cat = cat_allmodels[i][fold]
                y_pred_test_site += model_cat.predict(df) / self.fold
                gc.collect()

            df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
            df_test_sites.append(df_test_site)

            print("\nPrediction for meter:", i, "completed")
            gc.collect()
            submission = pd.concat(df_test_sites)
            submission.meter_reading = np.clip(np.expm1(submission.meter_reading), 0, a_max=None)
            submission.to_csv(OUTPUT_ROOT / "catboost_prediction.csv", index=False)


class Keras(object):

    def __init__(self, x, fold, batch_size, step_size=1000,
                 lr=0.001, dense_dim_1=64, dense_dim_2=32,
                 dense_dim_3=32, dense_dim_4=16, dropout1=0.2,
                 dropout2=0.1, dropout3=0.1, dropout4=0.1,
                 epochs=10, patience=3):
        self.x = x
        self.dense_dim_1 = dense_dim_1
        self.dense_dim_2 = dense_dim_2
        self.dense_dim_3 = dense_dim_3
        self.dense_dim_4 = dense_dim_4
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = patience
        self.fold = int(fold)
        self.step_size = int(step_size)

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
        air_temperature_mean_lag18 = Input(shape=[1], name='air_temperature_mean_lag18')
        air_temperature_max_lag18 = Input(shape=[1], name='air_temperature_max_lag18')
        air_temperature_min_lag18 = Input(shape=[1], name='air_temperature_min_lag18')
        air_temperature_median_lag18 = Input(shape=[1], name='air_temperature_median_lag18')
        dew_temperature_mean_lag18 = Input(shape=[1], name='dew_temperature_mean_lag18')
        dew_temperature_max_lag18 = Input(shape=[1], name='dew_temperature_max_lag18')
        dew_temperature_min_lag18 = Input(shape=[1], name='dew_temperature_min_lag18')
        dew_temperature_median_lag18 = Input(shape=[1], name='dew_temperature_median_lag18')
        cloud_coverage_mean_lag18 = Input(shape=[1], name='cloud_coverage_mean_lag18')
        cloud_coverage_max_lag18 = Input(shape=[1], name='cloud_coverage_max_lag18')
        cloud_coverage_min_lag18 = Input(shape=[1], name='cloud_coverage_min_lag18')
        cloud_coverage_median_lag18 = Input(shape=[1], name='cloud_coverage_median_lag18')
        mean_building_meter_x = Input(shape=[1], name='mean_building_meter_x')
        mean_building_meter_y = Input(shape=[1], name='mean_building_meter_y')
        median_building_meter_x = Input(shape=[1], name='median_building_meter_x')
        median_building_meter_y = Input(shape=[1], name='median_building_meter_y')

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
            , air_temperature_mean_lag18
            , air_temperature_max_lag18
            , air_temperature_min_lag18
            , air_temperature_median_lag18
            , dew_temperature_mean_lag18
            , dew_temperature_max_lag18
            , dew_temperature_min_lag18
            , dew_temperature_median_lag18
            , cloud_coverage_mean_lag18
            , cloud_coverage_max_lag18
            , cloud_coverage_min_lag18
            , cloud_coverage_median_lag18
            , mean_building_meter_x
            , mean_building_meter_y
            , median_building_meter_x
            , median_building_meter_y
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
            air_temperature_mean_lag18,
            air_temperature_max_lag18,
            air_temperature_min_lag18,
            air_temperature_median_lag18,
            dew_temperature_mean_lag18,
            dew_temperature_max_lag18,
            dew_temperature_min_lag18,
            dew_temperature_median_lag18,
            cloud_coverage_mean_lag18,
            cloud_coverage_max_lag18,
            cloud_coverage_min_lag18,
            cloud_coverage_median_lag18,
            mean_building_meter_x,
            mean_building_meter_y,
            median_building_meter_x,
            median_building_meter_y
        ], output)

        model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      loss=MSE,
                      metrics=[root_mean_squared_error])
        return model

    def callbacks(self, fold):
        early_stopping = EarlyStopping(patience=self.patience,
                                       monitor='val_root_mean_squared_error',
                                       verbose=1)

        model_checkpoint = ModelCheckpoint(os.path.join(KERAS_ROOT, f"model_{fold}.hdf5"),
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
        print('Keras Embedding Model')
        model = self.body()
        kf = KFold(n_splits=self.fold)
        cv_scores = {'Fold': [], 'cv_score': []}
        x_t = self.x
        y_t = x_t.meter_reading
        x_t = kerascolumns(x_t)
        ypred_all = np.zeros(x_t.shape[0])
        scores = 0
        for fold, (train_idx, valid_idx) in enumerate(kf.split(x_t, y_t)):
            print('Fold:', fold)
            cb1, cb2, cb3 = self.callbacks(fold)
            x_train, x_valid = x_t.iloc[train_idx], x_t.iloc[valid_idx]
            y_train, y_valid = y_t.iloc[train_idx], y_t.iloc[valid_idx]
            x_train = {col: np.array(x_train[col]) for col in x_train.columns}
            x_valid = {col: np.array(x_valid[col]) for col in x_valid.columns}
            model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_data=(x_valid, y_valid),
                      verbose=1,
                      callbacks=[cb1, cb2, cb3])
            keras_model = models.load_model(KERAS_ROOT / f'model_{fold}.hdf5',
                                            custom_objects={'root_mean_squared_error': root_mean_squared_error})
            y_pred = np.squeeze(keras_model.predict(x_valid))
            ypred_all[valid_idx] = y_pred
            rmse1 = np.sqrt(mean_squared_error(y_valid, y_pred))
            print('Fold:', fold + 1, 'RMSE', rmse1)
            scores += rmse1 / 2
            gc.collect()

        oof0 = mean_squared_error(y_t, ypred_all)
        cv_scores['Fold'].append(fold)
        cv_scores['cv_score'].append(scores)
        print('CV_RMSE:', np.sqrt(oof0))
        gc.collect()
        print(pd.DataFrame.from_dict(cv_scores))

    def predict(self):

        i = 0
        result = np.zeros((self.x.shape[0]), dtype=np.float32)
        row_ids = self.x.row_id
        keras_allmodels = []
        for fold in range(self.fold):
            keras_allmodels.append(models.load_model(KERAS_ROOT / f'model_{fold}.hdf5',
                                                     custom_objects={'root_mean_squared_error': root_mean_squared_error}))
        print('Predicting for Keras Embedding Model now')
        for j in tqdm(range(int(np.ceil(self.x.shape[0] / self.step_size)))):
            batched_df = self.x.iloc[i: i + self.step_size]
            for_prediction = kerascolumns(batched_df)
            result[i:min(i + self.step_size, self.x.shape[0])] =\
                np.expm1(sum([model.predict(for_prediction, batch_size=self.batch_size)[:, 0] for model in keras_allmodels]) / self.fold)
            i += self.step_size

        submission = pd.DataFrame({"row_id": row_ids, "meter_reading": result})
        submission.meter_reading = np.clip(np.expm1(submission.meter_reading), 0, a_max=None)
        submission.to_csv(OUTPUT_ROOT / 'keras_prediction.csv', index=False)

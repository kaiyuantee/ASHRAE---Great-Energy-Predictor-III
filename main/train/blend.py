import pandas as pd
import numpy as np
import os
import optuna
from .utils import OUTPUT_ROOT, DATA_ROOT
from .models import mean_squared_error


class GeneralizedMeanBlender():
    """Combines multiple predictions using generalized mean"""

    def __init__(self, p_range=(-2, 2)):
        """"""
        self.p_range = p_range
        self.p = None
        self.weights = None

    def _objective(self, trial, X, y):

        # create hyperparameters
        p = trial.suggest_uniform(f"p", *self.p_range)
        weights = [
            trial.suggest_uniform(f"w{i}", 0, 1)
            for i in range(X.shape[1])
        ]

        # blend predictions
        blend_preds, total_weight = 0, 0
        if p <= 0:
            for j, w in enumerate(weights):
                blend_preds += w * np.log1p(X[:, j])
                total_weight += w
            blend_preds = np.expm1(blend_preds / total_weight)
        else:
            for j, w in enumerate(weights):
                blend_preds += w * X[:, j] ** p
                total_weight += w
            blend_preds = (blend_preds / total_weight) ** (1 / p)

        # calculate mean squared error
        return np.sqrt(mean_squared_error(y, blend_preds))

    def fit(self, X, y, n_trials=10):
        # optimize objective
        obj = partial(self._objective, X=X, y=y)
        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials)
        # extract best weights
        if self.p is None:
            self.p = [v for k, v in study.best_params.items() if "p" in k][0]
        self.weights = np.array([v for k, v in study.best_params.items() if "w" in k])
        self.weights /= self.weights.sum()

    def transform(self, X):
        assert self.weights is not None and self.p is not None, \
            "Must call fit method before transform"
        if self.p == 0:
            return np.expm1(np.dot(np.log1p(X), self.weights))
        else:
            return np.dot(X ** self.p, self.weights) ** (1 / self.p)

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)


def main():
    testdf = pd.read_csv(DATA_ROOT/'test.csv')
    build = pd.read_csv(DATA_ROOT/'building_metadata.csv')
    leak = pd.read_feather(DATA_ROOT/ 'leak.feather')

    leak.fillna(0, inplace=True)
    leak = leak[(leak.timestamp.dt.year > 2016) & (leak.timestamp.dt.year < 2019)]
    leak.loc[leak.meter_reading < 0, 'meter_reading'] = 0 # remove negative values
    leak = leak[leak.building_id != 245]

    every_models = ['lightgbm.csv', 'catboost.csv', 'keras.csv']
    for i, model in enumerate(every_models):
        x = pd.read_csv(f'../input/{model}.csv', index_col=0).meter_reading
        x[x < 0] = 0
        testdf[f'pred{i}'] = x
        del x

    leakdf = pd.merge(leak, testdf[['building_id', 'meter', 'timestamp',
                                    *[f"pred{i}" for i in range(len(every_models))], 'row_id']], "left")
    leakdf = pd.merge(leak, build[['building_id', 'site_id']], 'left')

    # log1p then mean
    log1p_then_mean = np.mean(np.log1p(leak[[f"pred{i}" for i in range(len(every_models))]].values), axis=1)
    leak_score = np.sqrt(mean_squared_error(log1p_then_mean, np.log1p(leak.meter_reading)))
    print('log1p then mean score =', leak_score)

    # mean then log1p
    mean_then_log1p = np.log1p(np.mean(leak[[f"pred{i}" for i in range(len(every_models))]].values, axis=1))
    leak_score = np.sqrt(mean_squared_error(mean_then_log1p, np.log1p(leak.meter_reading)))
    print('mean then log1p score=', leak_score)

    X = np.log1p(leak[[f"pred{i}" for i in range(len(every_models))]].values)
    y = np.log1p(leak["meter_reading"].values)

    gmb = GeneralizedMeanBlender()
    gmb.fit(X, y, n_trials=20)
    # after optuna
    print(np.sqrt(mean_squared_error(gmb.transform(X), np.log1p(leak.meter_reading))))

    # make test predictions
    sample_submission = pd.read_csv(DATA_ROOT/"sample_submission.csv")
    x_test = testdf[[f"pred{i}" for i in range(len(every_models))]].values
    sample_submission['meter_reading'] = np.expm1(gmb.transform(np.log1p(x_test)))
    sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

    # save submission
    sample_submission.to_csv('finalpredictions.csv', index=False, float_format='%.4f')


if __name__ == '__main__':

    main()

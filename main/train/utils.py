import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_categorical_dtype
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
warnings.filterwarnings('ignore')
DATA_ROOT = Path(__file__).parent.parent.parent / 'Datasets'


def memory_reducer(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
               to reduce memory usage.
           """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def train_data():

    building_df = pd.read_csv(DATA_ROOT/'building_metadata.csv',
                              dtype={'building_id':np.int16},
                              index_col=None)
    train = pd.read_csv(DATA_ROOT/'train.csv',
                        dtype={'building_id':np.int16},
                        parse_dates=['timestamp'])

    train['meter_reading'] = np.log1p(train['meter_reading'])
    df = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
    df_group = df.groupby('building_id')['meter_reading']
    building_mean = df_group.mean()  # .astype(np.float16)
    building_median = df_group.median()  # .astype(np.float16)
    building_min = df_group.min()  # .astype(np.float16)
    building_max = df_group.max()  # .astype(np.float16)
    building_std = df_group.std()  # .astype(np.float16)
    df['building_mean'] = df['building_id'].map(building_mean)
    df['building_median'] = df['building_id'].map(building_median)
    df['building_min'] = df['building_id'].map(building_min)
    df['building_max'] = df['building_id'].map(building_max)
    df['building_std'] = df['building_id'].map(building_std)
    df = memory_reducer(df)
    building = memory_reducer(building_df)
    return df, building_df


def test_data():
    building = pd.read_csv('/content/drive/My Drive/building_metadata.csv',
                           dtype={'building_id': np.int16,
                                  'site_id': np.int8},
                           index_col=0)

    df = pd.read_csv('train.csv',
                     dtype={'building_id': np.int16,
                            'meter': np.int8},
                     parse_dates=['timestamp'])

    testdf = pd.read_csv('main.csv',
                         dtype={'building_id': np.int16,
                                'meter': np.int8},
                         parse_dates=['timestamp'])

    df['meter_reading_log1p'] = np.log1p(df['meter_reading'])
    df = df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
    df_group = df.groupby('building_id')['meter_reading_log1p']
    building_mean = df_group.mean()  # .astype(np.float16)
    building_median = df_group.median()  # .astype(np.float16)
    building_min = df_group.min()  # .astype(np.float16)
    building_max = df_group.max()  # .astype(np.float16)
    building_std = df_group.std()  # .astype(np.float16)
    testdf['building_mean'] = testdf['building_id'].map(building_mean)
    testdf['building_median'] = testdf['building_id'].map(building_median)
    testdf['building_min'] = testdf['building_id'].map(building_min)
    testdf['building_max'] = testdf['building_id'].map(building_max)
    testdf['building_std'] = testdf['building_id'].map(building_std)
    testdf = memory_reducer(testdf)
    building = memory_reducer(building)
    return testdf, building


class Weather(object):

    def __init__(self, x, y, df):
        self.x = x
        self.y = y
        self.df = df

    def holidays(self):
        uk = [1, 5]
        ire = [12]
        canada = [7, 11]
        us = [0, 2, 3, 4, 6, 8, 9, 10, 13, 14, 15]
        ukholi = ["2016-01-01", "2016-03-25", "2016-03-28", "2016-05-02", "2016-05-30",
                  "2016-08-29", "2016-12-26", "2016-12-27", "2017-01-01", "2017-01-02",
                  "2017-04-14", "2017-04-17", "2017-05-01", "2017-05-29", "2017-08-28",
                  "2017-12-25", "2017-12-26", "2018-01-01", "2018-03-30", "2018-04-02",
                  "2018-05-07", "2018-05-28", "2018-08-27", "2018-12-25", "2018-12-26"]
        ireholi = ["2016-01-01", "2016-03-17", "2016-03-28", "2016-05-02", "2016-06-06",
                   "2016-08-01", "2016-10-31", "2016-12-25", "2016-12-26", "2016-12-27",
                   "2017-01-01", "2017-01-02", "2017-03-17", "2017-04-17", "2017-05-01",
                   "2017-06-05", "2017-08-07", "2017-10-30", "2017-12-25", "2017-12-26",
                   "2018-01-01", "2018-03-19", "2018-04-02", "2018-05-07", "2018-06-04",
                   "2018-08-06", "2018-10-29", "2018-12-25", "2018-12-26"]
        canaholi = ["2016-01-01", "2016-02-15", "2016-03-25", "2016-05-23", "2016-07-01",
                    "2016-09-05", "2016-10-10", "2016-12-25", "2016-12-26", "2016-12-27",
                    "2017-01-01", "2017-02-20", "2017-04-14", "2017-05-22", "2017-07-01",
                    "2017-07-03", "2017-09-04", "2017-10-09", "2017-12-25", "2017-12-26",
                    "2018-01-01", "2018-02-19", "2018-03-30", "2018-04-02", "2018-05-21",
                    "2018-07-02", "2018-09-03", "2018-10-08", "2018-12-25", "2018-12-26"]
        for i in uk:
            self.df.loc[(self.df.site_id == i) & (self.df.timestamp.isin(ukholi)), 'is_holiday'] = self.df[
                (self.df.site_id == i) & (self.df.timestamp.isin(ukholi))].timestamp.isin(ukholi).astype(np.uint8)
        for j in ire:
            self.df.loc[(self.df.site_id == j) & (self.df.timestamp.isin(ireholi)), 'is_holiday'] = self.df[
                (self.df.site_id == j) & (self.df.timestamp.isin(ireholi))].timestamp.isin(ireholi).astype(np.uint8)
        for k in canada:
            self.df.loc[(self.df.site_id == k) & (self.df.timestamp.isin(canaholi)), 'is_holiday'] = self.df[
                (self.df.site_id == k) & (self.df.timestamp.isin(canaholi))].timestamp.isin(canaholi).astype(np.uint8)
        for l in us:
            dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
            us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
            self.df.loc[(self.df.site_id == l) & (self.df.timestamp.isin(us_holidays)), 'is_holiday'] = self.df[
                (self.df.site_id == l) & (self.df.timestamp.isin(us_holidays))].timestamp.isin(us_holidays).astype(np.uint8)
        self.df.loc[(self.df['weekday'] == 5) | (self.df['weekday'] == 6), 'is_holiday'] = 1
        self.df['is_holiday'] = self.df.is_holiday.fillna(0)

        return self.df

    def beaufort(self):

        beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8),
                    (6, 10.8, 13.9),
                    (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33),
                    (12, 33, 200)]
        self.df['wind_speed'] = self.df.wind_speed / 3.6
        # self.df['gust_speed'] = self.df.gust_speed / 3.6
        # self.df = self.df.round({'wind_speed':1,'gust_speed':1})
        self.df = self.df.round({'wind_speed': 1})
        for item in beaufort:
            self.df.loc[(self.df['wind_speed'] >= item[1]) & (self.df['wind_speed'] < item[2]), 'speed_beaufort'] = item[0]
            # self.df.loc[(self.df['gust_speed']>=item[1]) & (self.df['gust_speed']<item[2]), 'gust_beaufort'] = item[0]
        return self.df

    def add_lag_feature(self, window=3):
        group_df = self.df.groupby('site_id')
        cols = ['air_temperature', 'cloudcover', 'DewPointC',
                'precipMM', 'pressure', 'wind_direction', 'wind_speed']
        rolled = group_df[cols].rolling(window=window, min_periods=0)
        lag_mean = rolled.mean().reset_index()  # .astype(np.float16)
        lag_max = rolled.max().reset_index()  # .astype(np.float16)
        lag_min = rolled.min().reset_index()  # .astype(np.float16)
        lag_std = rolled.std().reset_index()  # .astype(np.float16)
        for col in cols:
            self.df[f'{col}_mean_lag{window}'] = lag_mean[col]
            self.df[f'{col}_max_lag{window}'] = lag_max[col]
            self.df[f'{col}_min_lag{window}'] = lag_min[col]
            self.df[f'{col}_std_lag{window}'] = lag_std[col]
        return self.df

    def timefeat(self):
        self.df['hour'] = np.int8(self.df['timestamp'].dt.hour)
        self.df['day'] = np.int8(self.df['timestamp'].dt.day)
        self.df['weekday'] = np.int8(self.df['timestamp'].dt.weekday)
        self.df['month'] = np.int8(self.df['timestamp'].dt.month)
        self.df['year'] = np.int8(self.df['timestamp'].dt.year - 2000)
        return self.df

    @staticmethod
    def degtocompass(num):
        val = int((num / 22.5) + .5)
        arr = [i for i in range(0, 16)]
        return arr[(val % 16)]

    def process(self):

        self.df['year'] = self.df.timestamp.dt.year - 2016
        self.df['month'] = self.df.year * 12 + self.df.timestamp.dt.month
        self.df = self.df.loc[(self.df.month > self.x) & (self.df.month <= self.y)]
        self.df.drop(['year', 'month'], axis=1, inplace=True)
        # TODO: tamporary put at here first until a better solution came up
        self.df['wind_direction'].fillna(self.df.wind_direction.mean(), inplace=True)
        # weather = add_lag_feature(add_lag_feature(weather,window=3), window=72)
        self.df = self.timefeat()
        self.df = self.holidays()
        self.df = self.beaufort()
        self.df['wind_direction'] = self.df['wind_direction'].apply(self.degtocompass)
        self.df = memory_reducer(self.df)
        return self.df


def create_dataset():

    df, build = train_data()
    weatherdf = pd.read_csv(DATA_ROOT/'weather_train.csv',
                            dtype={'site_id': np.int8},
                            parse_dates=['timestamp'],
                            index_col=None)
    weather_df = Weather(0, 12, weatherdf).process()

    return df, build, weather_df

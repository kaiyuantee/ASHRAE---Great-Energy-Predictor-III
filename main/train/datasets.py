import pandas as pd
import numpy as np
import datetime
import gc
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
DATA_ROOT = Path(__file__).parent.parent.parent / 'Datasets'


class Dataframe(object):

    def __init__(self, traindf, option):

        self.df = traindf
        self.option = option

    def process(self):

        if self.option == 'train':

            self.df['meter_reading'] = np.log1p(self.df['meter_reading'])
            newdf = self.df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
            df_group = newdf.groupby('building_id')['meter_reading']
            building_median = df_group.median()  # .astype(np.float16)
            newdf['building_median'] = newdf['building_id'].map(building_median)
            gc.collect()
            return newdf

        else:
            traindf = pd.read_csv(DATA_ROOT / 'train.csv',
                                  dtype={'building_id': np.int16},
                                  parse_dates=['timestamp'])

            traindf['meter_reading'] = np.log1p(traindf['meter_reading'])
            newdf = traindf.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
            df_group = newdf.groupby('building_id')['meter_reading']
            building_median = df_group.median()  # .astype(np.float16)
            self.df['building_median'] = newdf['building_id'].map(building_median)
            del traindf, newdf
            gc.collect()
            return self.df


class Weather(object):

    def __init__(self, df):

        self.df = df

    def fill_nan_values(self):

        # Find Missing Dates
        print('Getting missing dates now')
        time_format = "%Y-%m-%d %H:%M:%S"
        start_date = datetime.datetime.strptime(self.df['timestamp'].min(), time_format)
        end_date = datetime.datetime.strptime(self.df['timestamp'].max(), time_format)
        total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
        hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

        print('Preparing a temporary dataframe for filling')
        for site_id in range(16):
            site_hours = np.array(self.df[self.df['site_id'] == site_id]['timestamp'])
            new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
            new_rows['site_id'] = site_id
            self.df = pd.concat([self.df, new_rows])

            self.df = self.df.reset_index(drop=True)

        # Add new Features
        print('Creating new features now')
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df["day"] = self.df["timestamp"].dt.day
        self.df["week"] = self.df["timestamp"].dt.week
        self.df["month"] = self.df["timestamp"].dt.month
        self.df["hour"] = self.df['timestamp'].dt.hour
        self.df["weekday"] = self.df["timestamp"].dt.weekday
        self.df["weekofday"] = self.df["timestamp"].dt.weekofday

        # Reset Index for Fast Update
        self.df = self.df.set_index(['site_id', 'day', 'month'])
        print('Filling nan values now please hold')
        # Air temperature
        air_temperature_filler = pd.DataFrame(self.df.groupby(['site_id', 'day', 'month'])['air_temperature'].mean(),
                                              columns=["air_temperature"])
        self.df.update(air_temperature_filler, overwrite=False)

        # cloud_voerage
        # Step 1
        cloud_coverage_filler = self.df.groupby(['site_id', 'day', 'month'])['cloud_coverage'].mean()
        # Step 2
        cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'), columns=["cloud_coverage"])
        self.df.update(cloud_coverage_filler, overwrite=False)

        # dew temperature
        dew_temperature_filler = pd.DataFrame(self.df.groupby(['site_id', 'day', 'month'])['dew_temperature'].mean(),
                                              columns=["dew_temperature"])
        self.df.update(dew_temperature_filler, overwrite=False)

        # sea level pressure
        # Step 1
        sea_level_filler = self.df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].mean()
        # Step 2
        sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])
        self.df.update(sea_level_filler, overwrite=False)

        # wind direction
        wind_direction_filler = pd.DataFrame(self.df.groupby(['site_id', 'day', 'month'])['wind_direction'].mean(),
                                             columns=['wind_direction'])
        self.df.update(wind_direction_filler, overwrite=False)

        # wind speed
        wind_speed_filler = pd.DataFrame(self.df.groupby(['site_id', 'day', 'month'])['wind_speed'].mean(),
                                         columns=['wind_speed'])
        self.df.update(wind_speed_filler, overwrite=False)

        # precipitation depth 1 hour
        # Step 1
        precip_depth_filler = self.df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()
        # Step 2
        precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])
        self.df.update(precip_depth_filler, overwrite=False)

        # reset index and drop useless cols
        self.df.reset_index(inplace=True)
        self.df.drop(['day', 'week'], axis=1, inplace=True)

        return self.df

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
                (self.df.site_id == l) & (self.df.timestamp.isin(us_holidays))].timestamp.isin(us_holidays).astype(
                np.uint8)
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
            self.df.loc[(self.df['wind_speed'] >= item[1]) & (self.df['wind_speed'] < item[2]), 'speed_beaufort'] = \
            item[0]
            # self.df.loc[(self.df['gust_speed']>=item[1]) & (self.df['gust_speed']<item[2]), 'gust_beaufort'] = item[0]

        return self.df

    def add_lag_feature(self, window=3):

        group_df = self.df.groupby('site_id')
        cols = ['air_temperature', 'dew_temperature', 'cloud_coverage']
        rolled = group_df[cols].rolling(window=window, min_periods=0)
        lag_mean = rolled.mean().reset_index().astype(np.float16)
        lag_median = rolled.median().reset_index().astype(np.float16)
        lag_max = rolled.max().reset_index().astype(np.float16)
        lag_min = rolled.min().reset_index().astype(np.float16)
        lag_std = rolled.std().reset_index().astype(np.float16)
        lag_skew = rolled.skew().reset_index().astype(np.float16)
        for col in cols:
            self.df[f'{col}_mean_lag{window}'] = lag_mean[col]
            self.df[f'{col}_median_lag{window}'] = lag_median[col]
            self.df[f'{col}_max_lag{window}'] = lag_max[col]
            self.df[f'{col}_min_lag{window}'] = lag_min[col]
            self.df[f'{col}_std_lag{window}'] = lag_std[col]
            self.df[f'{col}_skew_lag{window}'] = lag_skew[col]

        return self.df

    def timefeat(self):

        self.df['hour'] = np.int8(self.df['timestamp'].dt.hour)
        # self.df['day'] = np.int8(self.df['timestamp'].dt.day)
        # self.df['week'] = np.int8(self.df['timestamp'].dt.week)
        self.df['weekday'] = np.int8(self.df['timestamp'].dt.weekday)
        self.df['month'] = np.int8(self.df['timestamp'].dt.month)
        # self.df['year'] = np.int8(self.df['timestamp'].dt.year - 2000)

        return self.df

    def set_localtime(self):

        zone_dict = {0: 4, 1: 0, 2: 7, 3: 4,
                     4: 7, 5: 0, 6: 4, 7: 4, 8: 4,
                     9: 5, 10: 7, 11: 4, 12: 0,
                     13: 5, 14: 4, 15: 4}

        for sid, zone in zone_dict.items():
            sids = self.df.site_id == sid
            self.df.loc[sids, 'timestamp'] = self.df[sids].timestamp - pd.offsets.Hour(zone)

        return self.df

    @staticmethod
    def degtocompass(num):

        val = int((num / 22.5) + .5)
        arr = [i for i in range(0, 16)]

        return arr[(val % 16)]

    def process(self):

        self.df.drop(['wind_direction', 'wind_speed', 'sea_level_pressure',
                      'precip_depth_1_hr'], axis=1, inplace=True)
        self.df = self.timefeat()
        self.df = self.set_localtime()
        self.df = self.df.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
        self.df = self.add_lag_feature(window=18)
        self.df = self.holidays()

        return self.df






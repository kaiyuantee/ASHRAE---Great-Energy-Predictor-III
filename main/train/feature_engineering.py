import numpy as np
from sklearn.preprocessing import LabelEncoder
from .utils import memory_reducer

le = LabelEncoder()
first = [2, 3, 4, 5, 6]
second = [8, 9, 10, 11, 12]

category_cols = ['building_id', 'site_id', 'primary_use',
                 'is_holiday', 'meter', 'speed_beaufort']

feature_cols = ['square_feet', 'hour', 'weekday', 'building_median'] + \
               ['air_temperature', 'cloud_coverage', 'dew_temperature',
                'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                'wind_speed']

cols = category_cols + feature_cols


class Preprocess(object):
    def __init__(self, df, build, weather):
        self.df = df
        self.build = build
        self.weather = weather

    @staticmethod
    def columns(df):

        all_features = [col for col in df.columns if col not in ["timestamp", "site_id", "meter_reading", "log_meter_reading"]]
        return all_features

    def core(self):

        self.df = self.df.merge(self.build, on='building_id', how='left')
        self.df = self.df.merge(self.weather, on=['site_id', 'timestamp'], how='left')
        self.df = memory_reducer(self.df)
        self.df['square_feet'] = np.log1p(self.df['square_feet'])
        self.df['primary_use'] = le.fit_transform(self.df.primary_use)
        df_building_meter = self.df.groupby(["building_id", "meter"]).agg(mean_building_meter=("log_meter_reading", "mean"),
                                                                          median_building_meter=("log_meter_reading", "median")).reset_index()
        df_building_meter_hour = self.df.groupby(["building_id", "meter", "hour"]).agg(mean_building_meter=("log_meter_reading", "mean"),
                                                                                       median_building_meter=("log_meter_reading", "median")).reset_index()
        self.df = self.df.merge(df_building_meter, on=["building_id", "meter"])
        self.df = self.df.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])
        self.df = memory_reducer(self.df)
        xt = self.df[(self.df.month.isin(first))]  # &(df.site_id==siteid)]
        yt = xt.meter_reading
        xt = xt[self.columns(self.df)]
        xv = self.df[(self.df.month.isin(second))]  # &(df.site_id==siteid)]
        yv = xv.meter_reading
        xv = xv[self.columns(self.df)]

        return xt, yt, xv, yv
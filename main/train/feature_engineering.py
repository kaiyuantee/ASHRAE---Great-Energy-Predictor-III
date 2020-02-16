import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from .utils import memory_reducer
from .directories import DATA_ROOT

le = LabelEncoder()


class Preprocess(object):
    def __init__(self, df, build, weather, option):
        self.df = df
        self.build = build
        self.weather = weather
        self.option = option

    def to_int8(self):
        cat_cols = ['weekday', 'hour', 'is_holiday']
        for col in cat_cols:
            self.df[col] = self.df[col].astype(np.int8)
        return self.df

    @staticmethod
    def new_feat(df):

        df_building_meter = df.groupby(["building_id", "meter"]).agg(mean_building_meter=("meter_reading", "mean"),
                                                                     median_building_meter=(
                                                                         "meter_reading", "median")).reset_index()
        df_building_meter_hour = df.groupby(["building_id", "meter", "hour"]).agg(
            mean_building_meter=("meter_reading", "mean"),
            median_building_meter=("meter_reading", "median")).reset_index()
        return df_building_meter, df_building_meter_hour

    def core(self):

        if self.option == 'test':
            traindf = pd.read_csv(DATA_ROOT / 'train.csv',
                                  dtype={'building_id': np.int16},
                                  parse_dates=['timestamp'])
            traindf['meter_reading'] = np.log1p(traindf['meter_reading'])
            traindf = traindf.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
            traindf['hour'] = traindf['timestamp'].dt.hour
            newdf1, newdf2 = self.new_feat(traindf)
            del traindf
            gc.collect()
        self.build.drop(['year_built', 'floor_count'], axis=1, inplace=True)
        self.df = self.df.merge(self.build, on='building_id', how='left')
        self.df = self.df.merge(self.weather, on=['site_id', 'timestamp'], how='left')
        self.df = memory_reducer(self.df)
        if self.option == 'train':
            newdf1, newdf2 = self.new_feat(self.df)
        self.df['square_feet'] = np.log1p(self.df['square_feet'])
        self.df['primary_use'] = le.fit_transform(self.df.primary_use)
        self.df = self.df.merge(newdf1, on=["building_id", "meter"])
        self.df = self.df.merge(newdf2, on=["building_id", "meter", "hour"])
        self.df = memory_reducer(self.df)
        self.df = self.to_int8()
        gc.collect()

        return self.df

import numpy as np
from sklearn.preprocessing import LabelEncoder
from .utils import memory_reducer

le = LabelEncoder()


class Preprocess(object):
    def __init__(self, df, build, weather):
        self.df = df
        self.build = build
        self.weather = weather

    def new_feat(self):

        df_building_meter = self.df.groupby(["building_id", "meter"]).agg(mean_building_meter=("meter_reading", "mean"),
                                                                          median_building_meter=(
                                                                          "meter_reading", "median")).reset_index()
        df_building_meter_hour = self.df.groupby(["building_id", "meter", "hour"]).agg(mean_building_meter=("meter_reading", "mean"),
                                                                                       median_building_meter=("meter_reading", "median")).reset_index()
        return df_building_meter, df_building_meter_hour

    def core(self):

        self.df = self.df.merge(self.build, on='building_id', how='left')
        self.df = self.df.merge(self.weather, on=['site_id', 'timestamp'], how='left')
        self.df = memory_reducer(self.df)
        self.df['square_feet'] = np.log1p(self.df['square_feet'])
        self.df['primary_use'] = le.fit_transform(self.df.primary_use)
        newdf1, newdf2 = self.new_feat()
        self.df = self.df.merge(newdf1, on=["building_id", "meter"])
        self.df = self.df.merge(newdf2, on=["building_id", "meter", "hour"])
        self.df = memory_reducer(self.df)

        return self.df

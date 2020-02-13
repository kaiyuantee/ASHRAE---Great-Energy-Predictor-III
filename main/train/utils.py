import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_categorical_dtype
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from .datasets import Dataframe, Weather
warnings.filterwarnings('ignore')
DATA_ROOT = Path(__file__).parent.parent.parent / 'Datasets'
OUTPUT_ROOT = Path(__file__).parent.parent.parent / 'outputdir'


class Dataset(object):

    def __init__(self, option):

        self.option = option

    def create_dataset(self):

        if (self.option == 'train') is not (self.option == 'test'):

            df = pd.read_csv(DATA_ROOT / f'{self.option}.csv',
                             dtype={'building_id': np.int16},
                             parse_dates=['timestamp'])
            building_df = pd.read_csv(DATA_ROOT/'building_metadata.csv',
                                      dtype={'building_id': np.int16},
                                      index_col=None)
            weatherdf = pd.read_csv(DATA_ROOT / f'weather_{self.option}.csv',
                                    dtype={'site_id': np.int8},
                                    parse_dates=['timestamp'],
                                    index_col=None)
            df = Dataframe(df, self.option).process()
            weather_df = Weather(weatherdf).process()
            df = memory_reducer(df)
            building_df = memory_reducer(building_df)
            weather_df = memory_reducer(weather_df)

            return df, building_df, weather_df

        else:
            print('Please enter either train or test only')
            exit()


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




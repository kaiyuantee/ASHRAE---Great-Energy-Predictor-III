import os
from .utils import *
DATA_ROOT = Path(__file__).parent.parent.parent / 'Datasets'
OUTPUT_ROOT = Path(__file__).parent.parent.parent / 'outputdir'
KERAS_ROOT = OUTPUT_ROOT / 'keras'
CATBOOST_ROOT = OUTPUT_ROOT / 'catboost'


def main():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    if not os.path.exists(KERAS_ROOT):
        os.makedirs(KERAS_ROOT)

    if not os.path.exists(CATBOOST_ROOT):
        os.makedirs(CATBOOST_ROOT)


if __name__ == '__main__':
    main()
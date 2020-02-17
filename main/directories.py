import os
from pathlib import Path
DATA_ROOT = Path(__file__).parent.parent / 'Datasets'
OUTPUT_ROOT = Path(__file__).parent.parent / 'outputdir'
LGBM_ROOT = OUTPUT_ROOT / 'lightgbm'
CATBOOST_ROOT = OUTPUT_ROOT / 'catboost'
KERAS_ROOT = OUTPUT_ROOT / 'keras'


def main():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    if not os.path.exists(LGBM_ROOT):
        os.makedirs(LGBM_ROOT)

    if not os.path.exists(KERAS_ROOT):
        os.makedirs(KERAS_ROOT)

    if not os.path.exists(CATBOOST_ROOT):
        os.makedirs(CATBOOST_ROOT)


if __name__ == '__main__':
    main()
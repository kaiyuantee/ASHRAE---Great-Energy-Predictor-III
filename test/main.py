import pandas as pd
import numpy as np
import argparse
from .models import LightGBM
from .utils import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', default='lightgbm', help='model type')
    arg('--epochs', default='10', help='number of epochs')

    args = parser.parse_args()

    if args.model == 'lightgbm':
        LightGBM()


if __name__ == '__main__':

    main()


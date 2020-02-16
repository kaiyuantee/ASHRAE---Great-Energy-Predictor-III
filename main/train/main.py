import argparse
from .models import *
from .utils import Dataset
from .feature_engineering import Preprocess


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', help='model type')
    arg('--mode', default='train', help='train or test mode')
    arg('--lr', default=0.005, help='learning rate')
    arg('--batch_size', default=1024, help='batchsize')
    arg('--epochs', default=10, help='number of epochs')
    arg('--patience', default=2, help='number of patience')
    arg('--fold', default=2, help='number of folds')

    args = parser.parse_args()

    df1, df2, df3 = Dataset(args.mode).create_dataset()

    newdf = Preprocess(df1, df2, df3, args.mode).core()

    if (args.mode == 'train') is not (args.mode == 'test'):

        if args.model == 'lightgbm':

            if args.mode == 'train':

                LightGBM(newdf, args.fold, args.mode).train()

            elif args.mode == 'test':

                LightGBM(newdf, args.fold, args.mode).predict()

        elif args.model == 'keras':

            if args.mode == 'train':

                Keras(newdf, args.fold).train()

            elif args.mode == 'test':

                Keras(newdf, args.fold).predict()

            else:
                exit()

        elif args.model == 'catboost':

            if args.mode == 'train':

                CatBoost(newdf, args.fold, args.mode).train()

            elif args.mode == 'test':

                CatBoost(newdf, args.fold, args.mode).predict()

            else:
                exit()
    else:
        print('Choose only train or test mode')
        exit()


if __name__ == '__main__':

    main()


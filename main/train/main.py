import argparse
from .models import *
from ..utils import Dataset


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', default='lightgbm', help='model type')
    arg('--starting_month', default=0, help='the begining month for weatherdf')
    arg('--ending_month', default=12, help='the ending month for weatherdf')
    arg('--option', default='train', help='train or test mode')
    arg('--dense_dim_1', default=64, help='number of nodes')
    arg('--dense_dim_2', default=32, help='number of nodes')
    arg('--dense_dim_3', default=32, help='number of nodes')
    arg('--dense_dim_4', default=16, help='number of nodes')
    arg('--dropout1', default=0.2, help='dropout scale')
    arg('--dropout2', default=0.1, help='dropout scale')
    arg('--dropout3', default=0.1, help='dropout scale')
    arg('--dropout4', default=0.1, help='dropout scale')
    arg('--lr', default=0.005, help='learning rate')
    arg('--batch_size', default=1024, help='batchsize')
    arg('--epochs', default=10, help='number of epochs')
    arg('--patience', default=2, help='number of patience')
    arg('--fold', default=2, help='number of folds')

    args = parser.parse_args()

    df1, df2, df3 = Dataset(args.starting_month,
                            args.ending_month,
                            args.option).create_dataset()

    xt, yt = preprocess('train', df1, df2, df3)
    xv, yv = preprocess('val', df1, df2, df3)

    if args.model == 'lightgbm':
        LightGBM(xt, yt, xv, yv).model()

    elif args.model == 'keras':
        all_models = []
        configs = [xt, yt, xv, yv,
                   args.dense_dim_1,
                   args.dense_dim_2,
                   args.dense_dim_3,
                   args.dense_dim_4,
                   args.dropout1,
                   args.dropout2,
                   args.dropout3,
                   args.dropout4,
                   args.lr,
                   args.batch_size,
                   args.epochs,
                   args.patience,
                   args.fold]

        for i in range(args.fold):
            if i % 2 == 0:
                xt, yt, xv, yv = xv, yv, xt, yt
            else:
                xt, yt, xv, yv = xt, yt, xv, yv
            all_models.append(Keras(configs).train())


if __name__ == '__main__':

    main()


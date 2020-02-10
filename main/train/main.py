import argparse
from .models import *
from main.train.utils import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', default='lightgbm', help='model type')
    arg('--dense_dim_1', default=10, help='number of nodes')
    arg('--dense_dim_2', default=10, help='number of nodes')
    arg('--dense_dim_3', default=10, help='number of nodes')
    arg('--dense_dim_4', default=10, help='number of nodes')
    arg('--dropout1', default=0.25, help='dropout scale')
    arg('--dropout2', default=0.25, help='dropout scale')
    arg('--dropout3', default=0.25, help='dropout scale')
    arg('--dropout4', default=0.25, help='dropout scale')
    arg('--lr', default=0.01, help='learning rate')
    arg('--batch_size', default=128, help='batchsize')
    arg('--epochs', default=10, help='number of epochs')
    arg('--patience', default=2, help='number of patience')
    arg('--fold', default=2, help='number of folds')

    args = parser.parse_args()

    df1, df2, df3 = create_dataset()
    xt, yt = preprocess('train', df1, df2, df3)
    xv, yv = preprocess('val', df1, df2, df3)
    if args.model == 'lightgbm':
        LightGBM(xt, yt, xv, yv).model()
    elif args.model == 'keras':
        Keras(xt, yt, xv, yv,
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
              args.fold).train()


if __name__ == '__main__':

    main()


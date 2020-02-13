import argparse
from pathlib import Path
from .models import *
from .utils import Dataset
from .feature_engineering import Preprocess


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', help='model type')
    arg('--mode', default='train', help='train or test mode')
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
    arg('--output-dir', help='path where to save')
    arg('--train-only', default='train', help='train only', action='store_true')
    arg('--test-only', default='test', help='test only without submission', action='store_true')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    df1, df2, df3 = Dataset(args.mode).create_dataset()

    newdf = Preprocess(df1, df2, df3, args.mode).core()

#     if (args.mode == 'train') is not (args.mode == 'test'):
#         if args.mode == 'test':
#             prediction(newdf)

#         elif args.mode == 'train':

#             if args.model == 'lightgbm':

#                 LightGBM(newdf, args.fold)

#             elif args.model == 'keras':
#                 all_models = []
#                 configs = dict(dense_dim_1=args.dense_dim_1,
#                                dense_dim_2=args.dense_dim_2,
#                                dense_dim_3=args.dense_dim_3,
#                                dense_dim_4=args.dense_dim_4,
#                                dropout1=args.dropout1,
#                                dropout2=args.dropout2,
#                                dropout3=args.dropout3,
#                                dropout4=args.dropout4,
#                                lr=args.lr,
#                                batch_size=args.batch_size,
#                                epochs=args.epochs,
#                                patience=args.patience,
#                                fold=args.fold)

#                 for i in range(args.fold):
#                     if i % 2 == 0:
#                         xt, yt, xv, yv = xv, yv, xt, yt
#                     else:
#                         xt, yt, xv, yv = xt, yt, xv, yv
#                     all_models.append(Keras(xt, yt, xv, yv, **configs))

#             elif args.model == 'catboost':
#                 all_models = []
#                 for i in range(args.fold):
#                     if i % 2 == 0:
#                         xt, yt, xv, yv = xv, yv, xt, yt
#                     else:
#                         xt, yt, xv, yv = xt, yt, xv, yv
#                         all_models.append(CatBoost(xt, yt, xv, yv))
#         else:
#             print('Choose only train or test mode')
#             exit()


if __name__ == '__main__':

    main()


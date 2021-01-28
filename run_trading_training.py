import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from trading_paper.utils import make_model, parse_args, calculate_accuracy, anchor
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')


def run_training_lstm(X, model, window=5, n_epochs=300, num_layers=1, size_layer=128, bidirectional=False):
    pbar = tqdm(range(n_epochs), desc='train loop')
    state_dim = 2 if bidirectional else 1
    for i in pbar:
        state = np.zeros((state_dim, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, X.shape[0] - 1, window):
            index = min(k + window, X.shape[0] - 1)
            batch_x = np.expand_dims(
                X.iloc[k: index, :].values, axis=0
            )
            batch_y = X.iloc[k + 1: index + 1, :].values
            try:
                logits, state, loss = model.update(batch_x, batch_y, state)
            except:
                print("What")
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))
    return model


if __name__ == '__main__':
    args = parse_args()


    def date_parser(x):
        return pd.to_datetime(x, format=args.date_format)


    # Read Dataset
    df = pd.read_csv(args.dataset, parse_dates=[args.date_column], date_parser=date_parser)
    df = df[(df[args.date_column] > args.start_date)]
    print("Dataset size: " + str(df.shape[0]) + " rows")

    # Scale Data
    target_index = df.columns.get_loc(args.target_column)
    minmax = MinMaxScaler().fit(df.iloc[:, target_index].astype('float32').values.reshape(-1, 1))  # Close index
    df_log = minmax.transform(df.iloc[:, target_index].astype('float32').values.reshape(-1, 1))  # Close index
    df_log = pd.DataFrame(df_log)

    # Split train and test
    df_train = df_log.iloc[:-args.test_size]
    df_test = df_log.iloc[-args.test_size:]

    # build model
    model_params = {
        'size': df_log.shape[1],
        'output_size': df_log.shape[1],
        'learning_rate': args.lr,
        'num_layers': args.n_layers,
        'size_layer': args.size_layer,
        'dropout_rate': args.dropout_rate
    }
    if 'cnn' in args.model:
        model_params['kernel_size'] = args.kernel_size
        model_params['n_attn_heads'] = args.n_attn_heads

    for i in range(args.n_experiments):
        np.random.seed()
        print("Running Experiment " + str(i))
        model = make_model(args.model, **model_params)

        if 'lstm' in args.model:
            bidirectional = 'bidirectional' in args.model
            model = run_training_lstm(X=df_train, model=model, window=args.window, n_epochs=args.n_epochs,
                                      num_layers=args.n_layers, size_layer=args.size_layer, bidirectional=bidirectional)

        model.close()

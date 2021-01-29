import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from trading_paper.utils import make_model, parse_args, anchor, run_training_lstm, evaluate_lstm
import sys
import warnings
import matplotlib.pyplot as plt
import time
import os
import joblib

if not sys.warnoptions:
    warnings.simplefilter('ignore')


if __name__ == '__main__':
    args = parse_args()


    def date_parser(x):
        return pd.to_datetime(x, format=args.date_format)


    # Read Dataset
    df = pd.read_csv(args.dataset, parse_dates=[args.date_column], date_parser=date_parser)
    df = df[(df[args.date_column] >= args.start_date) & (df[args.date_column] <= args.end_date)]
    print("Dataset size: " + str(df.shape[0]) + " rows")

    # Scale Data
    target_index = df.columns.get_loc(args.target_column)
    minmax = MinMaxScaler().fit(df.iloc[:, target_index].astype('float32').values.reshape(-1, 1))  # Close index
    df_log = minmax.transform(df.iloc[:, target_index].astype('float32').values.reshape(-1, 1))  # Close index
    df_log = pd.DataFrame(df_log)
    surplus = (df_log.shape[0] - args.test_size) - ((df_log.shape[0] - args.test_size) // args.window) * args.window
    df_log = df_log.iloc[:-surplus]
    # Split train and test
    df_train = df_log.iloc[:-args.test_size]
    df_test = df_log.iloc[-args.test_size - args.window:].values

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
    elif 'lstm' == args.model:
        model_params['stochastic'] = args.stochastic
        model_params['init_logstd'] = args.init_logstd
    results = []
    base_dir = args.log_dir + '/' + args.model + '/'
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(minmax.inverse_transform(df_train.values), label='true_values', marker='o')
    axs[1].plot(minmax.inverse_transform(df_test[-args.test_size:]), label='true_values', marker='o')
    for i in range(args.n_experiments):
        np.random.seed()
        print("Running Experiment " + str(i))
        model = make_model(args.model, **model_params)
        tf_path = base_dir + str(time.time()) if not args.debug else None
        if not args.debug and not os.path.exists(tf_path):
            os.makedirs(tf_path)
        if 'lstm' in args.model:
            bidirectional = 'bidir' in args.model
            init_state = model.get_init_state()
            model, last_state, predictions = run_training_lstm(X=df_train, model=model, window=args.window,
                                                               n_epochs=args.n_epochs,
                                                               num_layers=args.n_layers, size_layer=args.size_layer,
                                                               bidirectional=bidirectional, tf_path=tf_path)

            num_samples = args.stochastic_samples if args.stochastic or True else 1
            true_predict, simulation_predict = evaluate_lstm(df_test=df_test, model=model, window=args.window,
                                                             init_state=init_state, stochastic=args.stochastic,
                                                             num_samples=args.stochastic_samples)
            true_predict = minmax.inverse_transform(true_predict)
            predictions = minmax.inverse_transform(predictions)
            for j in range(num_samples):
                simulation_predict[j] = minmax.inverse_transform(simulation_predict[j])
            if args.anchor:
                true_predict = anchor(true_predict, args.anchor_weight)
                predictions = anchor(predictions, args.anchor_weight)
                for j in range(num_samples):
                    simulation_predict[j] = anchor(simulation_predict[j], args.anchor_weight)
            results.append((true_predict, simulation_predict))
            axs[0].plot(predictions, label='predict_' + str(i + 1))
            axs[1].plot(true_predict, label='good_predict_' + str(i + 1), marker='x')
            for j in range(num_samples):
                axs[1].plot(simulation_predict[j], label='simulation_predict_' + str(i + 1) + '_' + str(j + 1))

        # save model and parameters
        if not args.debug:
            joblib.dump(minmax, tf_path + '/scaler.gz')
            joblib.dump(model_params, tf_path + '/model_params.gz')
            model.save(tf_path + '/model.gz')
        model.close()


    for ax in axs:
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel("Value")
    plt.suptitle("S&P")
    axs[0].set_title('Train Data')
    axs[1].set_title('Test Data')
    plt.show()

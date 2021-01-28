import numpy as np
from trading_paper.models import LSTM, BidirectionalLSTM, GRU, BidirectionalGRU, Attention, CNN
import argparse
import tensorflow as tf
from tqdm import tqdm
import copy

name_to_model = {
    'lstm': LSTM,
    'bidir_lstm': BidirectionalLSTM,
    'gru': GRU,
    'bidir_gru': BidirectionalGRU,
    'cnn': CNN,
}


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


def make_model(model_name, **model_args):
    tf.reset_default_graph()
    if model_name in name_to_model.keys():
        model = name_to_model[model_name](**model_args)
    else:
        raise ValueError(model_name + " model not implemented!")

    return model


def run_training_lstm(X, model, window=5, n_epochs=300, num_layers=1, size_layer=128, bidirectional=False,
                      tf_path=''):
    pbar = tqdm(range(n_epochs), desc='train loop')
    writer = None
    if tf_path != '' and tf_path is not None:
        writer = tf.summary.FileWriter(tf_path)
        counter = 0
    if bidirectional:
        init_value = (np.zeros((1, num_layers * 2 * size_layer)), np.zeros((1, num_layers * 2 * size_layer)))
    else:
        init_value = np.zeros((1, num_layers * 2 * size_layer))
    for i in pbar:
        state = copy.deepcopy(init_value)
        total_loss, total_acc = [], []
        for k in range(0, X.shape[0] - 1, window):
            index = min(k + window, X.shape[0] - 1)
            batch_x = np.expand_dims(
                X.iloc[k: index, :].values, axis=0
            )
            batch_y = X.iloc[k + 1: index + 1, :].values
            logits, state, loss = model.update(batch_x, batch_y, state)
            total_loss.append(loss)
            accuracy = calculate_accuracy(batch_y[:, 0], logits[:, 0])
            total_acc.append(accuracy)
            if writer:
                summaries = [tf.Summary.Value(tag="loss", simple_value=loss),
                             tf.Summary.Value(tag="accuracy", simple_value=accuracy)]
                summary = tf.Summary(value=summaries)
                writer.add_summary(summary, counter)
                counter += 1

        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))
    predictions = np.zeros(X.shape)
    predictions[0] = X.iloc[0]
    state = copy.deepcopy(init_value)

    for k in range(0, (X.shape[0] // window) * window, window):
        logits, state = model.forward(np.expand_dims(
                    X.iloc[k: k + window], axis=0
                ), state)
        predictions[k + 1: k + window + 1] = logits

    # upper_b = (X.shape[0] // window) * window
    # if upper_b != X.shape[0]:
    #     logits, state = model.forward(np.expand_dims(
    #         X.iloc[upper_b:], axis=0
    #     ), state)
    #     predictions[upper_b + 1: X.shape[0] + 1] = out_logits
    #     date_ori.append(date_ori[-1] + timedelta(days=1))
    return model, state, predictions


def evaluate_lstm(df_test, model, init_state, window=5, stochastic=False,  num_samples=5):
    # if not stochastic:
    #     num_samples = 1
    test_size = df_test.shape[0] - window
    true_predict = []
    simulation_predict = [[] for _ in range(num_samples)]
    current_window = df_test[:window]
    simulation_current_window = [copy.deepcopy(df_test[:window]) for _ in range(num_samples)]
    last_state = init_state
    last_simulation_state = [copy.deepcopy(init_state) for _ in range(num_samples)]
    for i in range(test_size):
        out_logits, last_state = model.forward(current_window, last_state)
        true_predict.append(out_logits[-1])
        current_window = np.append(current_window, [df_test[window + i]], axis=0)
        current_window = np.delete(current_window, (0), axis=0)
        for j in range(num_samples):
            out_logits_simulation, last_simulation_state[j] = model.forward(simulation_current_window[j],
                                                                            last_simulation_state[j])
            simulation_predict[j].append(out_logits_simulation[-1])
            simulation_current_window[j] = np.append(simulation_current_window[j], [out_logits_simulation[-1]], axis=0)
            simulation_current_window[j] = np.delete(simulation_current_window[j], (0), axis=0)

    return true_predict, simulation_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='trading_paper/dataset/S&P.csv', help='Training dataset')
    parser.add_argument('--target_column', type=str, default='Price', help='Target column in dataset')
    parser.add_argument('--date_column', type=str, default='referenceDate', help='Date column in dataset')
    parser.add_argument('--date_format', type=str, default='%d/%m/%Y', help='Format of date column in dataset')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--n_layers', type=int, default=1, help='Number hidden layers')
    parser.add_argument('--size_layer', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for CNN model')
    parser.add_argument('--n_attn_heads', type=int, default=16, help='Number of attention heads for CNN model')
    parser.add_argument('--lr', type=float, default=0.01, help='Optimizer learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.8, help='Dropout rate')
    parser.add_argument('--stochastic', action='store_true', help='Make a stochastic model')
    parser.add_argument('--init_logstd', type=float, default=-1, help='Initial model variance')
    parser.add_argument('--stochastic_samples', type=int, default=5, help='Number of samples in stochastic models')
    parser.add_argument('--anchor', action='store_true', help='Smooth the output')
    parser.add_argument('--anchor_weight', type=float, default=0.3, help='Smoothness parameter')
    parser.add_argument('--start_date', type=str, default='2018-07-10', help='Starting date of the train dataset')
    parser.add_argument('--test_size', type=int, default=30, help='Size of test dataset')
    parser.add_argument('--n_experiments', type=int, default=1, help='Number of models to train')
    parser.add_argument('--window', type=int, default=5, help='Number of days in input')
    parser.add_argument('--model', type=str, default="lstm", help='Model architecture',
                        choices=['lstm', 'bidir_lstm', 'gru', 'bidir_gru', 'attention', 'cnn'])
    parser.add_argument('--log_dir', type=str, default='trading_paper/logs/', help='Tensorboard directory')
    parser.add_argument('--debug', action='store_true', help="Debug mode (Don't save logs)")
    return parser.parse_args()


def tf_switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

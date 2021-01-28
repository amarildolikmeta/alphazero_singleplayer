import numpy as np
from trading_paper.models import LSTM, BidirectionalLSTM, GRU, BidirectionalGRU, Attention, CNN
import argparse
import tensorflow as tf


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
    if model_name == 'lstm':
        model = LSTM(**model_args)
    elif model_name == 'bidir_lstm':
        model = BidirectionalLSTM(**model_args)
    elif model_name == 'gru':
        model = GRU(**model_args)
    elif model_name == 'bidir_gru':
        model = BidirectionalGRU(**model_args)
    elif model_name == 'attention':
        model = Attention(**model_args)
    elif model_name == 'cnn':
        model = CNN(**model_args)
    else:
        raise ValueError(model_name + " model not implemented!")

    return model


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
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropuot rate')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--start_date', type=str, default='2018-07-10', help='Starting date of the train dataset')
    parser.add_argument('--test_size', type=int, default=30, help='Size of test dataset')
    parser.add_argument('--n_experiments', type=int, default=1, help='Number of models to train')
    parser.add_argument('--window', type=int, default=5, help='Number of days in input')
    parser.add_argument('--model', type=str, default="lstm", help='Model architecture',
                        choices=['lstm', 'bidir_lstm,' 'gru', 'bidir_gru', 'attention', 'cnn'])
    return parser.parse_args()

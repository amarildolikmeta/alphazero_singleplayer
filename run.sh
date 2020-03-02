#!/bin/bash
python alphazero.py --game=Taxi --stochastic --gamma=0.95 --max_ep_len=200 --n_hidden_layers=3 --n_hidden_units=32 --eval_freq=10 --n_epochs=100 --temp=0.5 --lr=0.1

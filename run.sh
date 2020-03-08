#!/bin/bash
python alphazero.py --game=TaxiEasy --stochastic --gamma=0.95 --max_ep_len=200 --n_hidden_layers=3 --n_hidden_units=32 --eval_freq=10 --n_epochs=100 --temp=0.25 --lr=0.1 --batch_size=2048
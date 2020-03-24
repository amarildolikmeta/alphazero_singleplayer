#!/bin/bash
python hyperopt_taxi.py --parallel --mcts_only --game=Taxi --grid=grid.txt --stochastic --gamma=0.95 --max_ep_len=200 --eval_episodes=10 --n_mcts=500 --n_ep=1 --eval_freq=1
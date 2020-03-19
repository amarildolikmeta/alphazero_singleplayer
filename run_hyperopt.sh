#!/bin/bash
python hyperopt_taxi.py --game=Taxi --grid=grid.txt --stochastic --gamma=0.95 --max_ep_len=200 --eval_episodes=10 --n_mcts=50 --n_ep=1 --eval_freq=1
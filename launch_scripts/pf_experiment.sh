#!/bin/bash
cd ..
python3 alphazero.py --budget=1000000 --particles=1 --parallel --game=RaceStrategy  --gamma=0.99 --max_ep_len=55 --eval_freq=1 --c=2.2 --n_ep=1 --eval_episodes=10 --n_mcts=50 --mcts_only --n_experiments=1 --unbiased --budget_scheduler --slope=2.9 --max_workers=10 --min_budget=30
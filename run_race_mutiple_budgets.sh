#!/bin/bash


python3 run_multiple_budgets_experiment.py --game=RaceStrategy   --gamma=0.99 --max_ep_len=20   --particles 1 --unbiased  --eval_freq=20 --n_epochs=50 --lr=0.1 --batch_size=2048 --temp=0 --c=3.2 --n_ep=20 --eval_episodes=40 --n_mcts=50 --mcts_only --n_experiments=1 --max_workers 3 --depth_based_bias --budget_scheduler --parallel --scale_reward &

python3 run_multiple_budgets_experiment.py --game=RaceStrategy  --gamma=0.99 --max_ep_len=20   --stochastic --n_hidden_layers=3 --n_hidden_units=16 --eval_freq=20 --n_epochs=50 --lr=0.1 --batch_size=2048 --temp=0 --c=4.0 --n_ep=1 --eval_episodes=40 --n_mcts=50 --mcts_only --n_experiments=1 --max_workers 3 --depth_based_bias --budget_scheduler --alpha 0.92 --parallel --scale_reward&

python3 run_multiple_budgets_experiment.py --game=RaceStrategy    --gamma=0.99 --max_ep_len=20  --particles 1 --biased  --eval_freq=20 --n_epochs=50 --lr=0.1 --batch_size=2048 --temp=0 --c=3.2 --n_ep=20 --eval_episodes=40 --n_mcts=50 --mcts_only --n_experiments=1 --max_workers 3 --depth_based_bias --budget_scheduler --parallel --alpha 0.92 --scale_reward&

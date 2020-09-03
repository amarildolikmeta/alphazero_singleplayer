#!/bin/bash


python3 run_multiple_budgets_experiment.py --game=Cartpole   --gamma=0.99 --max_ep_len=100 --fail_prob 0.05   --particles 1 --unbiased --temp=0 --c=3.2 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias  --parallel --scale_reward &

python3 run_multiple_budgets_experiment.py --game=Cartpole  --gamma=0.99 --max_ep_len=100  --fail_prob 0.05   --stochastic --temp=0 --c=4.0 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias  --alpha 0.92 --parallel --scale_reward&

python3 run_multiple_budgets_experiment.py --game=Cartpole    --gamma=0.99 --max_ep_len=100 --fail_prob 0.05  --particles 1 --biased  --temp=0 --c=3.2 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias  --parallel --alpha 0.85 --scale_reward&


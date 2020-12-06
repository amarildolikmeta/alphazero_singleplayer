#!/bin/bash


python3 run_multiple_budgets_experiment.py --game=GridworldDiscrete   --gamma=0.99 --max_ep_len=25  --fail_prob 0.1 --scale_reward --mcts_only --temp=0 --c=3.2 --depth_based_bias --unbiased  --particles 1   --eval_episodes=20  --n_experiments=1 --max_workers 3  --parallel  &

python3 run_multiple_budgets_experiment.py --game=GridworldDiscrete  --gamma=0.99 --max_ep_len=25  --fail_prob 0.1 --scale_reward --mcts_only --temp=0 --c=4.0 --depth_based_bias  --stochastic  --alpha 0.25 --eval_episodes=20   --n_experiments=1 --max_workers 3  --parallel &

python3 run_multiple_budgets_experiment.py --game=GridworldDiscrete    --gamma=0.99 --max_ep_len=25  --fail_prob 0.1 --scale_reward  --mcts_only --temp=0 --c=3.2 --depth_based_bias --biased  --particles 1 --alpha 0.25 --eval_episodes=20  --n_experiments=1 --max_workers 3   --parallel &

python3 run_multiple_budgets_experiment.py --game=GridworldDiscrete    --gamma=0.99 --max_ep_len=25 --fail_prob 0.1 --scale_reward  --mcts_only --temp=0 --c=3.2 --depth_based_bias --model_based --particles 1  --eval_episodes=20  --n_experiments=1 --max_workers 3  --parallel &

#!/bin/bash

python3 run_multiple_budgets_experiment.py --game=RiverSwim   --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.4 --scale_reward --mcts_only --temp=0 --c=3.2 --unbiased  --particles 1   --eval_episodes=40  --n_experiments=1 --max_workers 3  --parallel  &

python3 run_multiple_budgets_experiment.py --game=RiverSwim  --gamma=0.99 --max_ep_len=20 --chain_dim 7  --fail_prob 0.4 --scale_reward --mcts_only --temp=0 --c=3.2  --stochastic  --alpha 0.85 --eval_episodes=40   --n_experiments=1 --max_workers 3  --parallel &

python3 run_multiple_budgets_experiment.py --game=RiverSwim    --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.4 --scale_reward  --mcts_only --temp=0 --c=3.2  --biased  --particles 1 --alpha 0.85 --eval_episodes=40  --n_experiments=1 --max_workers 3   --parallel &

python3 run_multiple_budgets_experiment.py --game=RiverSwim    --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.4 --scale_reward  --mcts_only --temp=0 --c=3.2  --model_based --particles 1  --eval_episodes=40  --n_experiments=1 --max_workers 3  --parallel &

#!/bin/bash
cd ..

#python3 run_multiple_budgets_experiment.py --game=RiverSwim-continuous   --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.1   --particles 1 --unbiased  --temp=0 --c=3.2 --n_ep=1 --eval_episodes=100 --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias --budget_scheduler --parallel --scale_reward &

#python3 run_multiple_budgets_experiment.py --game=RiverSwim-continuous  --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.1   --stochastic --temp=0 --c=4.0 --n_ep=1 --eval_episodes=100  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias --budget_scheduler --alpha 0.92 --parallel --scale_reward&

#python3 run_multiple_budgets_experiment.py --game=RiverSwim-continuous    --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.1  --particles 1 --biased  --temp=0 --c=3.2 --n_ep=1 --eval_episodes=100  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias --budget_scheduler --parallel --alpha 0.85 --scale_reward&

python3 run_multiple_budgets_experiment.py --game=RiverSwim-continuous    --gamma=0.99 --max_ep_len=20 --chain_dim 7 --fail_prob 0.1  --particles 1 --biased --second_version --temp=0 --c=3.2 --n_ep=1 --eval_episodes=100  --mcts_only --n_experiments=1 --max_workers 15 --depth_based_bias --budget_scheduler --parallel --alpha 0.85 --scale_reward&

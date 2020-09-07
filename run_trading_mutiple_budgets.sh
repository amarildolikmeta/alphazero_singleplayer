#!/bin/bash


#python3 run_multiple_budgets_experiment.py --game=Trading-v0   --gamma=0.99 --max_ep_len=50 --particles 1   --unbiased  --temp=0 --c=1.2  --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias --budget_scheduler --parallel --scale_reward &

#python3 run_multiple_budgets_experiment.py --game=Trading-v0  --gamma=0.99 --max_ep_len=50   --stochastic   --temp=0 --c=1.2  --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias --budget_scheduler --alpha 0.88 --parallel --scale_reward&

#python3 run_multiple_budgets_experiment.py --game=Trading-v0    --gamma=0.99 --max_ep_len=50  --particles 1 --biased    --temp=0 --c=1.2  --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 3 --depth_based_bias --budget_scheduler --parallel --alpha 1.0 --scale_reward&

#python3 run_multiple_budgets_experiment.py --game=Trading-v0    --gamma=0.99 --max_ep_len=50  --particles 1 --biased --second_version  --temp=0 --c=1.2  --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 20 --depth_based_bias --budget_scheduler --parallel --alpha 0.88 --scale_reward&

python3 run_multiple_budgets_experiment.py --game=Trading-v0    --gamma=0.99 --max_ep_len=50  --particles 1 --biased --third_version  --temp=0 --c=1.2  --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 20 --depth_based_bias --budget_scheduler --parallel --alpha 0.88 --scale_reward&

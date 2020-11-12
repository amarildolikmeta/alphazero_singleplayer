#!/bin/bash
cd ..

python3 run_multiple_alpha_experiment.py --game=Trading-v0  --gamma=0.99 --max_ep_len=50   --stochastic  --temp=0 --c=1.2  --eval_episodes=40 --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias   --parallel --scale_reward --budget_scheduler&

python3 run_multiple_alpha_experiment.py --game=Trading-v0    --gamma=0.99 --max_ep_len=50 --particles 1  --biased   --temp=0 --c=1.2 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 5 --depth_based_bias  --parallel  --scale_reward --budget_scheduler& #


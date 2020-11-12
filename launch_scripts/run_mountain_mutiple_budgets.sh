#!/bin/bash
cd ..

#python3 run_multiple_budgets_experiment.py --game=MountainCar   --gamma=0.99 --max_ep_len=250 --fail_prob 0.05   --particles 1 --unbiased --temp=0 --c=3.2 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 10 --depth_based_bias  --parallel --scale_reward &

#python3 run_multiple_budgets_experiment.py --game=MountainCar  --gamma=0.99 --max_ep_len=250  --fail_prob 0.05   --stochastic --temp=0 --c=4.0 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 10 --depth_based_bias  --alpha 0.25 --parallel --scale_reward&

#python3 run_multiple_budgets_experiment.py --game=MountainCar    --gamma=0.99 --max_ep_len=250 --fail_prob 0.05  --particles 1 --biased  --temp=0 --c=3.2 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 10 --depth_based_bias  --parallel --alpha 0.25 --scale_reward&

python3 run_multiple_budgets_experiment.py --game=MountainCar    --gamma=0.99 --max_ep_len=250 --fail_prob 0.05  --particles 1 --biased --second_version --temp=0 --c=3.2 --n_ep=1 --eval_episodes=40  --mcts_only --n_experiments=1 --max_workers 10 --depth_based_bias  --parallel --alpha 0.4 --scale_reward&

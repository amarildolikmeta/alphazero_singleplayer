#!/bin/bash
cd ..
python3 run_hyper_param_opt.py --parallel --mcts_only --game=RaceStrategy  --budget=1000000 --particles=1 --gamma=0.99 --max_ep_len=55 --temp=0  --n_ep=1 --eval_episodes=6  --n_experiments=1 --max_workers=10 --budget_scheduler --slope=2.9 --min_budget=30 --unbiased --db --dbname="pf_reward" --opt_iters=10

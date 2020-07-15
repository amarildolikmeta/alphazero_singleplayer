#!/bin/bash
python3 run_hyper_param_opt.py --parallel --mcts_only --game=RiverSwim-continuous  --budget=300000 --particles 1 --unbiased --gamma=0.99 --max_ep_len=20 --chain_dim 7 --temp=0  --n_ep=20 --eval_episodes=4  --n_experiments=1 --max_workers 10 --depth_based_bias

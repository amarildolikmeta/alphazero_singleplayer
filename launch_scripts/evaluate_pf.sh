#!/bin/bash
cd ..
python3 evaluate_pf_method.py --game=MiniGrid-RiverSwim-continuous-v0 --budget=10000 --gamma=0.99 --max_ep_len=10 --parallel --n_workers 6  --temp=0.15 --c=1.2 --n_ep=1 --eval_episodes=20 --n_mcts=50 --mcts_only --n_experiments=1
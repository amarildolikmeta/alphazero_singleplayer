#!/bin/bash
cd ..
python3 alphazero.py --game=MiniGrid-RiverSwim-continuous-v0 --alpha=0.44 --gamma=0.95 --max_ep_len=10  --budget 10000 --temp=0.15 --c=1.2 --n_ep=1 --eval_episodes=10 --n_mcts=50 --mcts_only --n_experiments=1 --parallel --stochastic --alpha=0.44
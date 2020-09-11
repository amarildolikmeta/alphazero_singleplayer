budgets=(1000 5000 10000 )
for i in "${budgets[@]}"
do
  python3 run_hyper_param_opt.py --parallel --mcts_only --game=Trading-v0  --budget=$i --stochastic --gamma=0.99  --max_ep_len=50 --temp=0  --n_ep=1 --eval_episodes=10  --n_experiments=1 --max_workers=5 --opt_iters=20 --depth_based_bias --scale_reward &
  python3 run_hyper_param_opt.py  --parallel  --mcts_only --game=Trading-v0  --budget=$i --particles 1 --biased   --gamma=0.99 --max_ep_len=50  --temp=0  --n_ep=1 --eval_episodes=10 --n_experiments=1 --max_workers 5 --opt_iters=20 --depth_based_bias  --scale_reward &
  python3 run_hyper_param_opt.py  --parallel  --mcts_only --game=Trading-v0  --budget=$i --particles 1 --unbiased   --gamma=0.99 --max_ep_len=50  --temp=0  --n_ep=1 --eval_episodes=10 --n_experiments=1 --max_workers 5 --opt_iters=20 --depth_based_bias  --scale_reward &
done
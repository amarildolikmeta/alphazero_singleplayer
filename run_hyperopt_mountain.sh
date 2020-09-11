budgets=(1000 3000 5000 7000 10000)
for i in "${budgets[@]}"
do
  #python3 run_hyper_param_opt.py --parallel --mcts_only --game=MountainCar   --fail_prob 0.05 --budget=$i --stochastic --gamma=0.99  --max_ep_len=250 --temp=0  --n_ep=1 --eval_episodes=10  --n_experiments=1 --max_workers=5 --opt_iters=12 --depth_based_bias --scale_reward &
  #python3 run_hyper_param_opt.py  --parallel  --mcts_only --game=MountainCar  --fail_prob 0.05 --budget=$i --min_alpha_hp 0.25 --particles 1 --biased   --gamma=0.99 --max_ep_len=250  --temp=0  --n_ep=1 --eval_episodes=10 --n_experiments=1 --max_workers 5 --opt_iters=12 --depth_based_bias  --scale_reward &
  #python3 run_hyper_param_opt.py  --parallel  --mcts_only --game=MountainCar --fail_prob 0.05 --budget=$i --particles 1 --unbiased   --gamma=0.99 --max_ep_len=250  --temp=0  --n_ep=1 --eval_episodes=10 --n_experiments=1 --max_workers 5 --opt_iters=12 --depth_based_bias  --scale_reward &
  python3 run_hyper_param_opt.py  --parallel  --mcts_only --game=MountainCar  --fail_prob 0.05 --budget=$i --min_alpha_hp 0.25 --particles 1 --biased --second_version  --gamma=0.99 --max_ep_len=250  --temp=0  --n_ep=1 --eval_episodes=10 --n_experiments=1 --max_workers 10 --opt_iters=12 --depth_based_bias  --scale_reward &

done
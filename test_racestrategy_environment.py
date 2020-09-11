from envs.race_strategy_full import RaceModel
import numpy as np

mdp = RaceModel()
_ = mdp.reset()
ret = 0
while True:
    # print(s)
    a = np.random.choice([0, 1, 2, 3], 9, replace=True)
    s, r, done, _ = mdp.step(a)
    print("Reward:" + str(r) + " Lap Time: " + str(r * mdp.max_lap_time))
    mdp.set_signature(mdp.get_signature())
    ret += r
    if done:
        print("Return:", ret)
        # print("Race Time:", mdp.time)
        break

from envs.race_strategy_full import RaceModel
import numpy as np
import time

mdp = RaceModel(verbose=False)
_ = mdp.reset()
ret = 0
lap = 1

agent_queue = mdp.get_agents_standings()
start = time.time()
while True:
    agent = mdp.get_next_agent()
    #print("Lap {}, agent {}".format(lap, agent))
    print()
    a = np.random.choice([0, 1, 2, 3])
    s, r, done, _ = mdp.partial_step(a, agent)
    #print("Reward:" + str(r))
    mdp.set_signature(mdp.get_signature())
    ret += r
    if mdp.has_transitioned():
        lap += 1
        now = time.time()
        milliseconds = now-start
        print(milliseconds)
        start = time.time()
    if done:
        print("Return:", ret)
        # print("Race Time:", mdp.time)
        break

from envs.race_strategy_full import RaceModel
import numpy as np

mdp = RaceModel(verbose=False)
_ = mdp.reset()
ret = 0
lap = 1

agent_queue = mdp.get_agents_standings()

while True:
    agent = agent_queue.get()
    print("Lap {}, agent {}".format(lap, agent))
    print()
    if agent_queue.empty():
        lap += 1
        agent_queue = mdp.get_agents_standings()
    a = np.random.choice([0, 1, 2, 3])
    s, r, done, _ = mdp.partial_step(a, agent)
    print("Reward:" + str(r))
    mdp.set_signature(mdp.get_signature())
    ret += r
    if done:
        print("Return:", ret)
        # print("Race Time:", mdp.time)
        break

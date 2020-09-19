import json

from race_components.race_ol_uct import RaceOL_MCTS
from race_components.race_pf_uct import RacePFMCTS


def load_race_agents_config(path: str, gamma: float):
    with open(path) as json_file:
        data = json.load(json_file)
        json_file.close()

    mcts_makers = []
    mcts_params = []
    c_dpw = []

    for agent in data["agents"]:
        if agent['agent_class'] == "race_ol_uct":
            mcts_makers.append(RaceOL_MCTS)
        elif agent['agent_class'] == "race_pf_uct":
            mcts_makers.append(RacePFMCTS)
        else:
            raise ValueError("Agent {} does not exist for race strategy".format(agent["agent_class"]))

        params = agent["parameters"]
        params["gamma"] = gamma
        mcts_params.append(params)
        c_dpw.append(agent["c_dpw"])

    return mcts_makers, mcts_params, c_dpw

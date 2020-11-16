from collections import defaultdict


class OfflineScore(object):
    """Object containing the results for one evaluation iteration"""

    def __init__(self, episode_rewards: list, rewards_per_timestep: list, ep_lengths: list, action_counts: list):
        self.episode_rewards = episode_rewards
        self.rewards_per_timestep = rewards_per_timestep
        self.ep_lengths = ep_lengths
        self.action_counts = action_counts

    def get_dictionary(self, gamma=1.) -> dict:

        returns = []
        lens = []
        rews = []
        actions = defaultdict(list)

        # Compute the discounted return
        for r_list in self.rewards_per_timestep:
            discount = 1
            disc_rew = 0
            for r in r_list:
                disc_rew += discount * r
                discount *= gamma
            rews.append(disc_rew)

        # Fill the lists for building the dataframe
        for ret, length, action_cnt in zip(self.episode_rewards, self.ep_lengths, self.action_counts):
            returns.append(ret)
            lens.append(length)
            for action_index, count in enumerate(action_cnt):
                actions[action_index].append(count)

        # Store the result of the experiment
        data = {"total_reward": returns,
                "discounted_reward": rews,
                "length": lens}

        # Store the action counts
        for index in actions:
            data["action_{}_count".format(str(index))] = actions[index]

        return data
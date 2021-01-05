import numpy as np
from stable_baselines.common.base_class import BaseRLModel

from resources.plots import plot


class Evaluator():
    def __init__(self) -> None:
        # init statistics
        self.episodes_rewards = []
        self.episodes_actions = []

    def run(self, model: BaseRLModel,
            episodes: int):
        """
        Evaluate a model on its env for some time
        :param model: trained BaseRLModel
        :param episodes: n episodes
        :return:
        """
        env = model.get_env()
        for i in range(episodes):
            rewards = [0 for i in range(env.steps)]
            actions = [0 for i in range(env.steps)]
            # get the first observation out of the environment
            state = env.reset()
            series = env.timeseries
            # play through the env
            while not env.done:
                # _states are only useful when using LSTM policies
                action, _states = model.predict(state)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                state, reward, done, _ = env.step(action)
                # print(obs, action, reward, done)
                if type(action) is np.ndarray:
                    actions.append(int(action[0]))
                else:
                    actions.append(int(action))
                rewards.append(reward)
            # Append to all Statistics
            self.episodes_rewards.append(rewards)
            self.episodes_actions.append(actions)
            # plot the actions against its series
            plot(series, actions)

            print("Rewards in Episode: {} are: {}".format(i, np.sum(rewards)))
        print("Maximum Reward: ", np.max(self.episodes_rewards),
              "\nAverage Reward: ", np.mean(self.episodes_rewards),
              "\n TestEpisodes: ", episodes)

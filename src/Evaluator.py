from copy import deepcopy

import numpy as np
from pandas import DataFrame
from stable_baselines.common.base_class import BaseRLModel
import pprint

from resources.plots import plot


class Evaluator():
    def __init__(self) -> None:
        # init statistics
        self.episodes_rewards = []
        self.episodes_actions = []
        self.episodes_stats = []

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
            stats = env.stats
            # play through the env
            while not env.done:
                # _states are only useful when using LSTM policies
                action, _states = model.predict(state)
                state, reward, done, _ = env.step(action)
                # verify action
                if type(action) is np.ndarray:
                    actions.append(int(action[0]))
                else:
                    actions.append(int(action))
                rewards.append(reward)
            # Append to all Statistics
            self.episodes_rewards.append(sum(rewards))
            self.episodes_actions.append(actions)
            # plot the actions against its series
            plot(series, actions)

            stats.print_history()
            print("Rewards in Episode: {}\n are: {}".format(i, np.sum(rewards)))
        print("Maximum Reward: ", np.max(self.episodes_rewards),
              "\nAverage Reward: ", np.mean(self.episodes_rewards),
              "\n TestEpisodes: ", episodes)


class Stats():
    """
    The Class Stats should be linked to every Timeseries.
    For each Timeseries we can then link statistics over FN, FP, TN, TP.
    Furthermore we can support the Stats for each Training Iteration and look at Trends and fitting performance
    Maybe look at https://pypi.org/project/pycm/
    """

    def __init__(self, series: DataFrame) -> None:
        self.series = deepcopy(series)
        values = self.series['anomaly'].value_counts(dropna=False).keys().tolist()
        counts = self.series['anomaly'].value_counts(dropna=False).tolist()
        self.absolutes = dict(zip(values, counts))
        self.confusion = {
            "FN": 0,
            "FP": 0,
            "TN": 0,
            "TP": 0,
        }
        self.history = []

    def update(self, reward: int, action: int) -> None:
        if reward == -5 and action == 0:
            self.confusion["FN"] += 1
        if reward == -1 and action == 1:
            self.confusion["FP"] += 1
        if reward == 1 and action == 0:
            self.confusion["TN"] += 1
        if reward == 5 and action == 1:
            self.confusion["TP"] += 1

    def print_confusion_matrix(self):
        print("\nAbsolute Occurrences:")
        pprint.pprint(self.absolutes, width=1)
        print("\nConfusion Matrix:")
        pprint.pprint(self.confusion, width=1)

    def print_history(self):
        print("History:\n")
        print("\nAbsolute Occurrences:")
        pprint.pprint(self.absolutes, width=1)
        print("\nConfusion Matrix:")
        pprint.pprint(self.history, width=1)

    def reset(self):
        """
        Call on Training Episode
        :return:
        """
        self.history.append(self.confusion)
        self.confusion = {
            "FN": 0,
            "FP": 0,
            "TN": 0,
            "TP": 0,
        }



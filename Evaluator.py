import matplotlib.pyplot as plt
import numpy as np
from stable_baselines.common.base_class import BaseRLModel


class Evaluator():
    def __init__(self,
                 model: BaseRLModel,
                 episodes: int) -> None:
        # base setup
        self.model = model
        self.episodes = episodes
        self.env = model.get_env()

        # init statistics
        self.episodes_rewards = []
        self.episodes_actions = []

    def run(self):
        """
        Execute the evaluation of our model inside its environment.
        :return:
        """
        for i in range(self.episodes):
            rewards = [0 for i in range(self.env.steps)]
            actions = [0 for i in range(self.env.steps)]
            # get the first observation out of the environment
            state = self.env.reset()
            series = self.env.timeseries
            # play through the env
            while not self.env.done:
                # _states are only useful when using LSTM policies
                action, _states = self.model.predict(state)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                state, reward, done, _ = self.env.step(action)
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
            self.plot(series, actions)

            print("Rewards in Episode: {} are: {}".format(i, np.sum(rewards)))
        print("Maximum Reward: ", np.max(self.episodes_rewards),
              "\nAverage Reward: ", np.mean(self.episodes_rewards),
              "\n TestEpisodes: ", self.episodes)

    def plot(series, actions):
        series = series
        plt.figure(figsize=(10, 7))
        plt.plot(series.index, actions, label="Actions", linestyle="solid")
        plt.plot(series.index, series["anomaly"], label="True Label", linestyle="dotted")
        for col in series.columns:
            plt.plot(series.index, series[col], label=col, linestyle="dashed")
        plt.legend()
        plt.ylabel('Reward Sum')
        plt.show()

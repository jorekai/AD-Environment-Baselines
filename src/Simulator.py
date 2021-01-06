from stable_baselines import DQN
from stable_baselines.common import BaseRLModel

import config
from src.DynamicStateEnv import DynamicStateEnv
from src.Evaluator import Evaluator
from src.PolicyNetwork import CustomPolicy


class Simulator():
    def __init__(self, environment: DynamicStateEnv, evaluator: Evaluator) -> None:
        self.env = environment
        self.evaluator = evaluator
        self.model = self.create_model()

    def create_model(self):
        dqn = DQN(CustomPolicy,
                  self.env,
                  gamma=0.999,
                  learning_rate=0.001,
                  buffer_size=500000,
                  exploration_fraction=0.1,
                  exploration_final_eps=0.01,
                  exploration_initial_eps=0.5,
                  train_freq=4,
                  batch_size=256,
                  double_q=True,
                  learning_starts=5000,
                  target_network_update_freq=1500,
                  prioritized_replay=True,
                  prioritized_replay_alpha=0.6,
                  prioritized_replay_beta0=0.4,
                  prioritized_replay_beta_iters=None,
                  prioritized_replay_eps=1e-06,
                  param_noise=False,
                  n_cpu_tf_sess=None,
                  verbose=1,
                  tensorboard_log=config.LOG_PATH,
                  _init_setup_model=True,
                  policy_kwargs=None,
                  full_tensorboard_log=False,
                  seed=None)
        return dqn

    def run(self, training_steps: int,
            log_name: str) -> None:
        self.train(timesteps=training_steps,
                   log_name=log_name)
        self.eval(self.model,
                  1)

    def train(self, timesteps: int,
              log_name: str):
        self.model.learn(total_timesteps=timesteps,
                         tb_log_name=log_name)

    def eval(self, model: BaseRLModel,
             episodes: int):
        self.evaluator.run(model=model,
                           episodes=episodes)


if __name__ == '__main__':
    sim = Simulator(environment=DynamicStateEnv(verbose=True),
                    evaluator=Evaluator())
    sim.run(25000, "Test")

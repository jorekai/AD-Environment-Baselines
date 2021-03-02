from stable_baselines import DQN, TRPO
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C

import config
from src.DynamicStateEnv import DynamicStateEnv
from src.DynamicStatePredictionEnv import DynamicStatePredictionEnv
from src.Evaluator import Evaluator
from src.PolicyNetwork import CustomPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

class Simulator():
    def __init__(self, environment: DynamicStateEnv, evaluator: Evaluator) -> None:
        self.env = environment
        self.evaluator = evaluator
        self.model = self.create_model()

    def create_model(self):
        dqn = DQN(CustomPolicy,
                  self.env,
                  gamma=0.99,
                  learning_rate=0.0001,
                  buffer_size=500000,
                  exploration_fraction=0.85,
                  exploration_final_eps=0.05,
                  exploration_initial_eps=1,
                  train_freq=4,
                  batch_size=1024,
                  double_q=True,
                  learning_starts=50000,
                  target_network_update_freq=256,
                  prioritized_replay=True,
                  prioritized_replay_alpha=1,
                  prioritized_replay_beta0=0.8,
                  prioritized_replay_beta_iters=None,
                  prioritized_replay_eps=1e-06,
                  param_noise=False,
                  n_cpu_tf_sess=None,
                  verbose=1,
                  tensorboard_log=config.ROOT_DIR + config.LOG_PATH,
                  _init_setup_model=True,
                  policy_kwargs=None,
                  full_tensorboard_log=False,
                  seed=None)
        return dqn

    def create_a2c(self):
        return A2C(MlpPolicy,
                   self.env,
                   gamma=0.99,
                   n_steps=5,
                   vf_coef=0.25,
                   ent_coef=0.01,
                   max_grad_norm=0.5,
                   learning_rate=0.0007,
                   alpha=0.99,
                   momentum=0.0,
                   epsilon=1e-05,
                   lr_schedule='constant',
                   verbose=0,
                   tensorboard_log=None,
                   _init_setup_model=True,
                   policy_kwargs=None,
                   full_tensorboard_log=False,
                   seed=None,
                   n_cpu_tf_sess=None)

    def create_ppo1(self):
        return PPO1(MlpPolicy, self.env,
                    gamma=0.99,
                    timesteps_per_actorbatch=1500,
                    clip_param=0.2,
                    entcoeff=0.01,
                    optim_epochs=4,
                    optim_stepsize=0.001,
                    optim_batchsize=256,
                    lam=0.95,
                    adam_epsilon=1e-05,
                    schedule='linear',
                    verbose=0,
                    tensorboard_log=None,
                    _init_setup_model=True,
                    policy_kwargs=None,
                    full_tensorboard_log=False,
                    seed=None,
                    n_cpu_tf_sess=1)

    def create_trpo(self):
        return TRPO(MlpPolicy,
                    self.env,
                    gamma=0.99,
                    timesteps_per_batch=1024,
                    max_kl=0.01,
                    cg_iters=10,
                    lam=0.98,
                    entcoeff=0.0,
                    cg_damping=0.01,
                    vf_stepsize=0.0003,
                    vf_iters=3,
                    verbose=0,
                    tensorboard_log=config.ROOT_DIR + config.LOG_PATH,
                    _init_setup_model=True,
                    policy_kwargs=None,
                    full_tensorboard_log=False,
                    seed=None,
                    n_cpu_tf_sess=None)

    def run(self, training_steps: int,
            log_name: str) -> None:
        self.train(timesteps=training_steps,
                   log_name=log_name)
        self.eval(self.model,
                  5)

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
                    evaluator=Evaluator("/LargeHorizon_Betas_5L/"))
    sim.run(10000, "DebugRun")
    history = sim.env.test_stats
    print(history.history)

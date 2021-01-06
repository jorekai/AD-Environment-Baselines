import datetime
import os

# Environment
LABELS = {"NORMAL": 0, "ANOMALY": 1}
REWARDS = {"TP": 5, "TN": 1, "FP": -1, "FN": -5}
ACTION_SPACE = [LABELS["NORMAL"], LABELS["ANOMALY"]]

# Storage
STORAGE_PATH = "/storage/"
FIGURE_PATH = "/storage/figures/"
LOG_PATH = "/tensorboard/"
MODEL_PATH = "/storage/models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data
TRAIN_SET_NAME = "full_dataset_train"
TRAIN_DIR = ROOT_DIR + "/datasets/train/"
TEST_SET_NAME = "full_dataset_test"
TEST_DIR = ROOT_DIR + "/datasets/test/"
FIX_TRAIN_FILE = False
DATASETS_DIR = ROOT_DIR + "/datasets/"

# Memory
MEMORY_NAME = "binary_state_env"
RELOAD_MEMORY = False
MEMORY_SIZE = 500000
MEMORY_INIT_SIZE = 5000

# Debug
VERBOSE = False

# Learning
EPISODES = 100  # 1 Episode is 1 complete Timeseries
EPOCHS = 10  # 1 Epoch is 1 experience replay of size batch
ALPHA = 0.001  # learning rate estimator
GAMMA = 0.999  # discount factor
EPSILON_INIT = 0.01
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 7500000
EPSILON_DECAY = 0.9
BATCH_SIZE = 256  # something between 128 and 512
TIMESTEPS = 0
UPDATE_STEPS = 10
STEPS = 25

# Figures
REWARDS_TRAINING = "rewards_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"

# NN
HIDDEN_NEURONS = 64


import random
from typing import Tuple, List

import gym
import numpy as np

import config
from resources import utils, plots


class DynamicStateEnv(gym.Env):

    def __init__(self,
                 train_directory=config.TRAIN_DIR,
                 test_directory=config.TEST_DIR,
                 verbose=config.VERBOSE):
        super(DynamicStateEnv, self).__init__()
        if verbose:
            self.init_time = utils.start_timer()

        # init dataframes
        self.__init_dataframes(train_directory, test_directory)

        # gym interface and state attributes
        self.action_space = gym.spaces.Discrete(len(config.ACTION_SPACE))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(config.STEPS, self.train_dataframes[0].columns.size + 1),
                                                dtype=np.float32)
        self.timeseries = []
        self.cursor = -1
        self.cursor_init = 0
        self.states = []
        self.done = False
        self.steps = config.STEPS
        self.cursor_init = config.STEPS

        # init training and testing attributes
        self.__init_train_test_attributes()


        # DEBUG INFO
        self.verbose = verbose
        if verbose:
            self.train_stats_files = utils.init_stats(self.train_dataframes)
            self.train_stats = None
            self.test_stats_files = utils.init_stats(self.test_dataframes)
            self.test_stats = None
            self.__info()

    def __state(self, previous_state: List = []) -> np.ndarray:
        """
        The State property of our environment is the concatenation of our previous states,action pairs and the current
        state,action pair.
        The returned state will consist of a binary state, the binary state can be seen as labelling and not labelling.
        Therefore the observation only returns the path which we chose by our policy
        :param previous_state: array of previous state
        :return: binary_state
        """
        # if we are at the init state we gather the previous samples all together
        if self.cursor == self.steps:
            state = []
            # gather the first n_steps of features and store them as a list
            for i in range(self.cursor):
                statecols0 = [self.timeseries[column][i] for column in self.timeseries]
                # add a init action 0
                statecols0.append(0)
                # append to our binary state
                state.append(statecols0)

            # do the equivalent for the other action in the action space
            state.pop(0)
            statecols1 = [self.timeseries[column][i] for column in self.timeseries]
            statecols1.append(1)
            state.append(statecols1)
            combined = np.array(state, dtype='float32')
            return combined

        # if we are beyond our initial state choose the previous states and concatenate with our current state
        if self.cursor > self.steps:
            # get the binary state representation
            cols0 = [self.timeseries[column][self.cursor] for column in self.timeseries]
            cols0.append(0)
            cols1 = [self.timeseries[column][self.cursor] for column in self.timeseries]
            cols1.append(1)
            # drop the oldest feature of n_steps and append our current features
            state0 = np.concatenate((previous_state[1:self.steps],
                                     [cols0]))
            state1 = np.concatenate((previous_state[1:self.steps],
                                     [cols1]))
            # concatenate our binary tree structure and return it
            combined = np.array([state0, state1], dtype='float32')
            return combined

    def __reward(self) -> List[int]:
        """
        Returns a reward List because we want to choose our states by a binary choice of actions
        The Reward will contain two integers specified in the config
        :return: List(int, int)
        """
        # if we are at the init position of our horizon then set our binary rewards according to the env
        if self.cursor >= self.steps:

            # if the label at the horizon is 0 e.g. a normal value
            if self.timeseries['anomaly'][self.cursor] == 0:
                return [config.REWARDS["TN"], config.REWARDS["FP"]]

            # if the label at the horizon is 0 e.g. an anomalous value
            if self.timeseries['anomaly'][self.cursor] == 1:
                return [config.REWARDS["FN"], config.REWARDS["TP"]]
        # return a binary [0, 0] if we are not at the init step yet
        else:
            return [0, 0]

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state and return the initial state
        :return: state
        """
        # if we are training on one file just get the first file in our list, else set file index to 0
        if not self.fix_train_file:
            self.file_index = (self.file_index + 1) % self.train_range
            self.file_index_test = (self.file_index_test + 1) % self.test_range
        else:
            self.file_index = 0
            self.file_index_test = 0
        # if we are training or testing we need to set different DataFrames
        if not self.test:
            self.timeseries = self.train_dataframes[self.file_index]
            self.train_stats = self.train_stats_files[self.file_index]
            if self.verbose:
                print("Current File: ", utils.get_filename_by_index(file_list=self.train_files, idx=self.file_index))
        else:
            self.timeseries = self.test_dataframes[self.file_index_test]
            self.test_stats = self.test_stats_files[self.file_index_test]
            if self.verbose:
                print("Current File: ", utils.get_filename_by_index(file_list=self.test_files, idx=self.file_index_test))
        # initialize cursor and goal flage
        self.cursor = self.cursor_init
        self.done = False

        # set init state and return init state
        self.states = self.__state()

        return self.states

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        The Step inside our environment. We propagate through the environment by increasing a cursor running over
        the pandas DataFrame.
        :param action: valid action (0,1)
        :return: Tuple(observation, reward, done, debug)
        """
        # if action is valid and cursor is initialized correctly to step horizon
        assert (action in self.action_space)
        assert (self.cursor >= self.steps)

        # get the reward of our action as a list of rewards
        reward = self.__reward()

        # update the cursor because we are making a step in the environment
        self.update_cursor()

        # on verbose log training stats, training/testing difference
        if self.verbose:
            if not self.test:
                self.train_stats.update(reward[action], action)
                if self.done:
                    self.train_stats.reset()
            else:
                self.test_stats.update(reward[action], action)
                if self.done:
                    self.test_stats.reset()

        # if we are done get the final state concatenation, else return the correct state
        if self.done:
            state = np.array([self.states, self.states])
        else:
            state = self.__state(self.states)

        # if our state is of a binary shape we want to extract the state by our action
        if len(np.shape(state)) > len(np.shape(self.states)):
            self.states = state[action]
        else:
            self.states = state

        # return the Tuple of observation, reward, done, debug
        return state[action], float(reward[action]), self.done, {}

    def update_cursor(self) -> None:
        """
        Increment the cursor, used to get to the next state
        :return: None
        """
        self.cursor += 1
        if self.cursor >= self.timeseries['anomaly'].size:
            self.done = True

    def is_done(self) -> bool:
        """
        Are we done with the current timeseries?
        :return: boolean
        """
        return True if self.cursor >= len(self.timeseries) - 1 else False

    def __is_anomaly(self):
        """
        Is the current position a anomaly?
        :return: boolean
        """
        return True if self.timeseries["anomaly"][self.cursor] == 1 else False

    def __str__(self) -> str:
        """
        Get the current Filename if needed
        :return: str
        """
        return utils.get_filename_by_index(self.train_files, self.file_index)

    def __info(self) -> None:
        """
        Get Information about the Current Environment
        :return: None
        """
        data_msg = f"Loaded {len(self.train_dataframes)} training frames. Overall training on {self.train_size} Samples\n" \
                   f"Loaded {len(self.test_dataframes)} testing frames. Overall testing on {self.test_size} Samples"
        init_msg = f"Initialised BaseEnvironment in {round(utils.get_duration(self.init_time), 3)} seconds."
        print(data_msg + "\n" + init_msg)

    def show_series(self, fix: bool = True) -> None:
        """
        Can visualize all Series inside the dataset or just one random series
        :param fix: if False plots all Series inside directory
        :return: void
        """
        if fix:
            idx = random.randint(0, len(self.train_files) - 1)
            frame = self.train_dataframes[idx]
            name = utils.get_filename_by_index(self.train_files, idx)
            plots.plot_series(frame, name)
        else:
            for idx, file in enumerate(self.train_dataframes):
                frame = self.train_dataframes[idx]
                name = utils.get_filename_by_index(self.train_files, idx)
                plots.plot_series(frame, name)

    def __init_dataframes(self, train_dir: str, test_dir: str) -> None:
        """
        DatFrames are loaded via the directory path, training and testing environment differ in their states
        :param train_dir: path
        :param test_dir: path
        :return: None
        """
        self.train_directory = train_dir
        self.train_files = utils.get_file_list_from_directory(self.train_directory)
        self.test_directory = test_dir
        self.test_files = utils.get_file_list_from_directory(self.test_directory)

        self.train_dataframes, self.test_dataframes = utils.init_dataframes(train_files=self.train_files,
                                                                            test_files=self.test_files)

    def __init_train_test_attributes(self):
        """
        Attributes like training samples overall, length of the frames and the current index are set here
        :return: None
        """
        # TRAINING
        self.fix_train_file = config.FIX_TRAIN_FILE
        self.file_index = random.randint(0, len(self.train_files) - 1)
        self.train_size = utils.get_sample_size_overall(self.train_dataframes)
        self.train_range = len(self.train_dataframes)

        # TESTING
        self.file_index_test = random.randint(0, len(self.test_files) - 1)
        self.test_range = len(self.test_dataframes)
        self.test_size = utils.get_sample_size_overall(self.test_dataframes)
        self.test = False

    def render(self, mode='human'):
        """
        Overrides Stable Baselines rendering method
        :param mode:
        :return:
        """
        pass

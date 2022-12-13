from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


class MAB(ABC):

    def __init__(self):
        self.contextual_rewards = None

    @abstractmethod
    def choose_arm(self, front_ex_arm, back_ex_arm):
        best_value = -np.inf
        best_arm = -1
        for arm in range(len(self.narms)):
            if self.contextual_rewards[arm, front_ex_arm, back_ex_arm] > best_value:
                best_arm = arm
                best_value = self.contextual_rewards[arm, front_ex_arm, back_ex_arm]

        return best_arm

    @abstractmethod
    def contextual_update(self, arm, front_arm, back_arm, reward):
        self.contextual_rewards[arm, front_arm, back_arm] = reward


class EpsGreedy(MAB):

    def __init__(self, narms, epsilon):
        # Set number of arms
        self.narms = narms
        # Exploration probability
        self.epsilon = epsilon
        # Q0 values
        self.Q0 = np.ones(self.narms) * np.inf
        # Total step count
        self.step_n = 0
        # Step count for each arm
        self.step_arm = np.zeros(self.narms)
        # Mean reward for each arm
        self.AM_reward = np.zeros(self.narms)

        # itself plus two neighbours, i.e. the one in front and the one in the back
        self.contextual_rewards = np.zeros((narms, narms, narms))
        super().__init__()

    # Play one round and return the action (chosen arm)
    def choose_arm(self, tround):
        # Generate random number
        p = np.random.rand()

        if p < self.epsilon:
            action = np.random.choice(self.narms)
        else:
            # Q0 values are initially set to np.inf. Hence, choose an arm with maximum Q0 value (
            # for all of them is np.inf, and therefore will play all of the arms at least one time)

            if len(np.where(self.Q0 == 0)[0]) < self.narms:
                # choose an arm with maximum Q0 value
                action = np.random.choice(np.where(self.Q0 == max(self.Q0))[0])
                # after the arm is chosen, set the corresponding Q0 value to zero
                self.Q0[action] = 0
            else:
                # Now, after that we ensure that there is no np.inf in Q0 values and all of them are set to zero
                # we return to play based on average mean rewards
                action = super(EpsGreedy, self).choose_arm(self.AM_reward)
        return action

    def update(self, arm, reward):
        super(EpsGreedy, self).update(arm, reward)

    def contextual_update(self, arm, front_arm, back_arm, reward):
        super(EpsGreedy, self).contextual_update(arm, front_arm, back_arm, reward)


class UCB(MAB):

    def __init__(self, narms, rho):
        # Set number of arms
        self.narms = narms
        # Rho
        self.rho = rho
        # Q0 values
        self.Q0 = np.ones(self.narms) * np.inf
        # Total step count
        self.step_n = 0
        # Step count for each arm
        self.step_arm = np.zeros(self.narms)
        # Mean reward for each arm
        self.AM_reward = np.zeros(self.narms)
        super().__init__()

    # Play one round and return the action (chosen arm)
    def choose_arm(self, tround):
        # Q0 values are initially set to np.inf. Hence, choose an arm with maximum Q0 value (
        # for all of them is np.inf, and therefore will play all of the arms at least one time)

        if len(np.where(self.Q0 == 0)[0]) < self.narms:
            # choose an arm with maximum Q0 value
            action = np.random.choice(np.where(self.Q0 == max(self.Q0))[0])
            # after the arm is chosen, set the corresponding Q0 value to zero
            self.Q0[action] = 0
        else:
            # Now, after that we ensure that there is no np.inf in Q0 values and all of them are set to zero
            # we return to play based on average mean rewards

            # construct UCB values which performs the sqrt part
            ucb_values = np.zeros(self.narms)
            for arm in range(self.narms):
                if self.step_arm[arm] > 0:
                    ucb_values[arm] = np.sqrt(self.rho * (np.log(self.step_n)) / self.step_arm[arm])
            action = super(UCB, self).choose_arm(self.AM_reward + ucb_values)
        return action

    def update(self, arm, reward, context=None):
        super(UCB, self).update(arm, reward)

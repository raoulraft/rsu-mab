from abc import ABC, abstractmethod
import numpy as np


class MAB(ABC):

    @abstractmethod
    def choose_arm(self, arm_q_values):
        # average mean reward array
        # at round t is passed to this function
        self.arm_q_values = arm_q_values
        # choose an arm which yields maximum value of average mean reward, tie breaking randomly
        chosen_arm = np.random.choice(np.where(self.arm_q_values == max(self.arm_q_values))[0])
        return chosen_arm

    @abstractmethod
    def update(self, arm, reward):
        # get the chosen arm
        self.arm = arm
        # update the overall step of the model
        self.step_n += 1
        # update the step of individual arms
        self.step_arm[self.arm] += 1
        # update average mean reward of each arm
        self.AM_reward[self.arm] = ((self.step_arm[self.arm] - 1) / float(self.step_arm[self.arm])
                                    * self.AM_reward[self.arm] + (1 / float(self.step_arm[self.arm])) * reward)



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
        super().__init__()

    # Play one round and return the action (chosen arm)
    def choose_arm(self, tround):
        # Generate random number
        p = np.random.rand()

        if self.epsilon == 0 and self.step_n == 0:
            action = np.random.choice(self.narms)
        elif p < self.epsilon:
            action = np.random.choice(self.narms)
        else:
            # Q0 values are initially set to np.inf. Hence, choose an arm with maximum Q0 value (
            # for all of them is np.inf, and therefore will play all of the arms at least one time)

            if len(np.where(self.Q0 == 0)[0]) < 10:
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

        if len(np.where(self.Q0 == 0)[0]) < 10:
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
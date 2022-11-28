import functools
import numpy as np
import wandb
import math
import statistics

from gym.spaces import MultiDiscrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from rsu import RSU
from utils import get_lmda, get_overload, get_depletion, get_latency


def env(n_rsu, n_cpus_max, lmda_zones):
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(n_rsu, n_cpus_max, lmda_zones)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(n_rsu, n_cpus_max, lamda_zones):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = VehicularParallelEnv(n_rsu, n_cpus_max, lamda_zones)
    env = parallel_to_aec(env)
    return env


class VehicularParallelEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, n_rsu, n_cpus_max, lmda_zones, reward_mode=0, threshold_queue=20,
                 threshold_battery=0, battery_weight=0.5, queue_weight=0.5, battery_recharge_rate=1,
                 battery_depletion_rate=1, epsilon_battery=0.0000001, epsilon_queue=0.00001, proc_rate=1,
                 queue_max_size=20, battery_max_size=100):

        self.reward_mode = reward_mode  # 0: competitive, 1: mean, 2: increase performance of the worst
        self.threshold_queue = threshold_queue  # PARAMETRO DA FAR VARIARE, per ora lo lascio così
        self.threshold_battery = threshold_battery  # should always be zero
        self.battery_weight = battery_weight  # PARAMETRO DA FAR VARIARE, per ora lo lascio così
        self.queue_weight = queue_weight  # PARAMETRO DA FAR VARIARE, per ora lo lascio così

        self.battery_recharge_rate = battery_recharge_rate  # Fabio
        self.battery_depletion_rate = battery_depletion_rate  # PARAMETRO DA FAR VARIARE, per ora ora lo lascio così
        self.epsilon_battery = epsilon_battery  # PARAMETRO DA FAR VARIARE, provane alcuni e vedi se alcuni convergono e altri no
        self.epsilon_queue = epsilon_queue  # PARAMETRO DA FAR VARIARE, provane alcuni e vedi se alcuni convergono e altri no

        self.proc_rate = proc_rate  # OK!
        self.n_cpus_max = n_cpus_max
        self.queue_max_size = queue_max_size  # Fabio
        self.battery_max_size = battery_max_size  # OK!
        self.lmda_zones = lmda_zones
        self.episode = -1  # starts at -1, then immediately is increased to 0 by calling reset()
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_rsu = n_rsu
        self.rsu = [RSU(starting_ce=n_cpus_max, starting_off_prob=0,
                        threshold_queue=threshold_queue, threshold_battery=threshold_battery, proc_rate=proc_rate,
                        battery_recharge_rate=battery_recharge_rate, battery_depletion_rate=battery_depletion_rate,
                        n_cpus_max=n_cpus_max, queue_max_size=queue_max_size,
                        battery_max_size=battery_max_size) for _ in range(self.n_rsu)]

        self.feature_size = self.n_rsu  # lambda of the zone of each rsu
        self.action_per_agent = [11, self.n_cpus_max]
        self.possible_agents = ["rsu" + str(r) for r in range(self.n_rsu)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._action_spaces = {agent: MultiDiscrete(self.action_per_agent) for agent in self.possible_agents}
        self._observation_spaces = {agent: Box(low=0, high=1, shape=(self.feature_size,), dtype=np.float32)
                                    for agent in self.possible_agents}

        self.steps = 0
        self.max_steps = 1
        self.cumulative_reward = 0
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Box(low=0, high=1, shape=(self.feature_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return MultiDiscrete(self.action_per_agent)

    def render(self, mode="human"):
        pass

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        '''

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.cumulative_reward = 0
        self.episode += 1
        obs = {agent: self._get_obs(agent_idx) for agent_idx, agent in enumerate(self.agents)}

        return obs

    def step(self, actions):
        # action = cpu, off for each rsu
        self.steps += 1
        for n in range(self.n_rsu):
            # print("rsu: ", n)
            # print("actions: ", actions[n])
            self._execute_actions(actions[n], n)

        done = self.steps >= self.max_steps

        rewards = {agent: self._get_reward(agent_idx) for agent_idx, agent in enumerate(self.agents)}
        reward_list = [self._get_reward(agent_idx) for agent_idx, _ in enumerate(self.agents)]
        dones = {agent: done for agent in self.agents}
        observations = {self._get_obs(agent_idx) for agent_idx, _ in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        if self.reward_mode == 0:
            rewards = [self._get_reward(agent_idx) for agent_idx, _ in enumerate(self.agents)]
            _rewards = rewards
            worst_reward = min(_rewards)
            index_worst_rsu = rewards.index(worst_reward)
        elif self.reward_mode == 1:
            rewards = [self._get_reward(agent_idx) for agent_idx, _ in enumerate(self.agents)]
            _rewards = rewards
            worst_reward = min(_rewards)
            index_worst_rsu = rewards.index(worst_reward)

            mean_reward = statistics.mean(rewards)
            rewards = [mean_reward for _ in self.agents]
        else:
            rewards = [self._get_reward(agent_idx) for agent_idx, _ in enumerate(self.agents)]
            _rewards = rewards
            worst_reward = min(rewards)
            index_worst_rsu = rewards.index(worst_reward)
            rewards = [worst_reward for _ in self.agents]

        sum_rewards = sum(_rewards)
        self.cumulative_reward += sum_rewards

        if done:
            for i in range(self.n_rsu):
                wandb.log({f"rsu {i} CEs": self.rsu[i].ce, "episode": self.episode}, commit=False)
                wandb.log({f"rsu {i} OFF PROB": self.rsu[i].off_prob, "episode": self.episode}, commit=False)
                wandb.log({f"rsu {i} received reward": rewards[i], "episode": self.episode}, commit=False)
            wandb.log({"episode reward": self.cumulative_reward, "episode": self.episode}, commit=False)
            wandb.log({"worst performing rsu": index_worst_rsu, "episode": self.episode}, commit=True)
            pass

        return observations, rewards, dones, infos

    def _execute_actions(self, actions, agent_index):
        self.rsu[agent_index].set_ce(actions[0])
        self.rsu[agent_index].set_off_prob(actions[1])

    def _get_obs(self, agent_index):
        obs = self.lmda_zones[agent_index]
        # obs = np.array(obs)
        return obs

    # reward is equal for all rsu. I still keep agent_index for possible extensions with different rewards
    def _get_reward(self, agent_index):
        # reward is equal for all rsu
        rsu = self.rsu[agent_index]
        offloading_prob = [agent.get_off_prob() for agent in self.rsu]
        rsu_id = agent_index
        lmda_zones = self.lmda_zones
        computing_elements = [self.rsu[idx].get_ce() for idx in range(self.n_rsu)]
        proc_rate = rsu.proc_rate  # [self.rsu[idx].proc_rate for idx in range(self.n_rsu)]  # same for all RSUs
        battery_recharge_rate = self.rsu[rsu_id].battery_recharge_rate  # same for all RSUs
        battery_depletion_rate = 1
        CE = self.rsu[rsu_id].get_ce()
        battery_max_size = self.rsu[rsu_id].battery_max_size
        threshold_battery = self.rsu[rsu_id].threshold_battery
        queue_max_size = self.rsu[rsu_id].queue_max_size
        threshold_latency = self.rsu[rsu_id].threshold_queue

        epsilon_battery = self.epsilon_battery
        epsilon_queue = self.epsilon_queue
        battery_weight = self.battery_weight
        queue_weight = self.queue_weight

        in_rate = get_lmda(offloading_probabilities=offloading_prob, rsu_id=rsu_id,
                           lmda_zones=lmda_zones, computing_elements=computing_elements, mu=proc_rate)

        # if threshold battery is not zero, then it doesn't work. (get_depletions checks that threshold<CE to use
        # battery equations rather than queue equations)
        prob_dep = get_depletion(battery_recharge_rate, battery_depletion_rate, CE, battery_max_size, threshold_battery)
        assert 0 <= prob_dep <= 1

        prob_latency = get_latency(in_rate, proc_rate, CE, queue_max_size, threshold_latency)
        assert 0 <= prob_latency <= 1, f"prob {prob_latency} is > 1"

        wandb.log({f"prob depletion rsu {agent_index}": prob_dep, "episode": self.episode}, commit=False)
        wandb.log({f"prob overload rsu {agent_index}": prob_latency, "episode": self.episode}, commit=False)

        # TODO: reward_dep and reward_overload should both be between 0 and 1, but are instead between -1 and 1
        if prob_dep >= epsilon_battery:
            reward_dep = (math.log(prob_dep) - math.log(epsilon_battery)) / (math.log(epsilon_battery))
        else:
            reward_dep = 1

        if prob_latency >= epsilon_queue:
            reward_overload = (math.log(prob_latency) - math.log(epsilon_queue)) / (math.log(epsilon_queue))
        else:
            reward_overload = 1

        """

        reward_dep = - prob_dep
        reward_overload = -prob_latency

        """

        """

        reward_dep = -math.log(prob_dep)
        reward_overload = -math.log(prob_latency)

        """

        reward = (battery_weight * reward_dep) + (queue_weight * reward_overload)

        wandb.log({f"episode reward depletion rsu {agent_index}": reward_dep, "episode": self.episode}, commit=False)
        wandb.log({f"episode reward latency rsu {agent_index}": reward_overload, "episode": self.episode},
                  commit=False)
        wandb.log({f"episode total reward rsu {agent_index}": reward, "episode": self.episode}, commit=False)

        # TODO: maybe give both reward_latency and reward_depletion only to cpu-agent, while give only
        #       reward_latency to offloading-agents

        return reward
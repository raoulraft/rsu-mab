from mab_env import VehicularParallelEnv as MultiEnv
import math
import wandb
import random
import numpy as np
from mab import MAB, UCB, EpsGreedy
import itertools
project_name = "edge-vehicular-network-rsu-mab"

rsu_scenario = [4]
cpu_scenario = [2]
lmda_zones = [[0.4, 0.3, 0.2, 0.5], ]  # [[1.4, 0.8, 0.9, 2.1], ]

cpu_actions = 2
offloading_actions = 11

a = [0, 1]
b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
c = list(itertools.product(a, b))


for reward_function in [1, 2]:
    n_zones_index = 0
    for n_rsu in rsu_scenario:
        for cpu in cpu_scenario:
            for use_eps in [True, False]:
                # CPU agents, one for each RSU. Each agent can pull 3 levers (1 CPU, 2 CPU, 3 CPU)

                # cpu_agents = [EpsGreedy(cpu, epsilon_greedy) for _ in range(n_rsu)] if not is_ucb else [UCB(cpu, 2, "cpu") for _ in range(n_rsu)]
                # offloading_agents = [EpsGreedy(11, epsilon_greedy) for _ in range(n_rsu)] if not is_ucb else [UCB(11, 2, "offload") for _ in range(n_rsu)]

                t = 0
                env = MultiEnv(n_rsu, cpu, lmda_zones[n_zones_index], reward_mode=reward_function, use_epsilon=use_eps)

                wandb.init(
                    name="baseline",
                    project=project_name,
                    tags=["n {}".format(n_rsu),
                          "cpu {}".format(cpu),
                          "reward mode {}".format(reward_function),
                          "use eps {}".format(use_eps),
                          "baseline"
                          ],
                    entity="xraulz",
                    reinit=True,
                )
                for i in range(n_rsu):
                    wandb.log({f"lmda_zone{i}": lmda_zones[n_zones_index][i]}, commit=False)

                actions = list(itertools.product(c, c, c, c))
                best_found_rewards = -np.inf
                while t < 50000:
                    _ = env.reset()
                    done = False
                    while not done:
                        t += 1

                        action = actions.pop(random.randrange(len(actions)))
                        _, rewards, done, _ = env.step(action)
                        sum_reward = sum(rewards)
                        if sum_reward >= best_found_rewards:
                            best_found_rewards = sum_reward

                        wandb.log({"best found reward": best_found_rewards, "episode": t}, commit=False)

                wandb.finish()
        n_zones_index += 1


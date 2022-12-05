from mab_env import VehicularParallelEnv as MultiEnv
import math
import wandb
import random
from mab import MAB, UCB, EpsGreedy
project_name = "edge-vehicular-network-rsu-mab"

rsu_scenario = [4]
cpu_scenario = [4]
lmda_zones = [[0.4, 1.3, 0.2, 0.5], ]  # [[1.4, 0.8, 0.9, 2.1], ]
epsilon_greedy = 0.2

for reward_function in [2]:
    n_zones_index = 0
    for n_rsu in rsu_scenario:

        for cpu in cpu_scenario:
            # CPU agents, one for each RSU. Each agent can pull 3 levers (1 CPU, 2 CPU, 3 CPU)

            # TODO: try UCB
            cpu_agents = [EpsGreedy(cpu, epsilon_greedy) for _ in range(n_rsu)]
            offloading_agents = [EpsGreedy(11, epsilon_greedy) for _ in range(n_rsu)]

            alpha = 0.001  # maybe use different alphas for cpu and offloading?

            t = 0
            env = MultiEnv(n_rsu, cpu, lmda_zones[n_zones_index], reward_mode=reward_function)

            wandb.init(
                project=project_name,
                tags=["n {}".format(n_rsu),
                      "cpu {}".format(cpu),
                      "reward mode {}".format(reward_function),
                      ],
                entity="xraulz",
                reinit=True,
            )
            for i in range(n_rsu):
                wandb.log({f"lmda_zone{i}": lmda_zones[n_zones_index][i]}, commit=False)
            for episode in range(100000):
                _ = env.reset()
                done = False
                while not done:
                    t += 1

                    action_vec = []
                    for i in range(n_rsu):
                        best_cpu_arm = cpu_agents[i].choose_arm(t)
                        best_offloading_arm = offloading_agents[i].choose_arm(t)

                        action_vec.append([best_cpu_arm, best_offloading_arm])

                    # print("action vec:", action_vec)
                    _, rewards, done, _ = env.step(action_vec)

                    for i in range(n_rsu):

                        reward = rewards[i]
                        best_cpu_arm = action_vec[i][0]
                        best_offloading_arm = action_vec[i][1]
                        cpu_agents[i].update(best_cpu_arm, reward)
                        offloading_agents[i].update(best_offloading_arm, reward)

                        # wandb.log({f"n_t_cpu[{i}][{best_cpu_arm}]":n_t_cpu[i][max_cpu_index]}, commit=False)
                        # wandb.log({f"n_t_offloading[{i}][{max_off_index}]": n_t_offloading[i][max_off_index]}, commit=False)

            wandb.finish()
        n_zones_index += 1
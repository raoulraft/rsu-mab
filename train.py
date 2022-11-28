from mab_env import VehicularParallelEnv as MultiEnv
import math
import wandb
import random
project_name = "edge-vehicular-network-rsu-mab"

rsu_scenario = [2]
cpu_scenario = [2]
lmda_zones = [[0.4, 3.8, 0.9, 4.1], ]  # [[1.4, 0.8, 0.9, 2.1], ]
for reward_function in [2]:
    n_zones_index = 0
    for n_rsu in rsu_scenario:

        for cpu in cpu_scenario:
            # CPU agents, one for each RSU. Each agent can pull 3 levers (1 CPU, 2 CPU, 3 CPU)
            n_t_actions_cpu = [1 for _ in range(cpu)]
            q_actions_cpu = [0 for _ in range(cpu)]
            n_t_cpu = [n_t_actions_cpu for _ in range(n_rsu)]
            q_a_cpu = [q_actions_cpu for _ in range(n_rsu)]
            # print("counter cpu-action for each rsu:", n_t_cpu)
            # print("q-values cpu-action for each rsu:", q_a_cpu)
            n_t_actions_offloading = [1 for _ in range(11)]
            q_actions_offloading = [0 for _ in range(11)]
            n_t_offloading = [n_t_actions_offloading for _ in range(n_rsu)]
            q_a_offloading = [q_actions_offloading for _ in range(n_rsu)]
            # print("counter offloading-action for each rsu:", n_t_offloading)
            # print("q-values offloading-action for each rsu:", q_a_offloading)
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
            for episode in range(10000):
                _ = env.reset()
                done = False
                while not done:
                    t += 1
                    a_cpu = [0 for _ in range(cpu)]
                    a_cpu = [a_cpu for _ in range(n_rsu)]
                    a_off = [0 for _ in range(11)]
                    a_off = [a_off for _ in range(n_rsu)]
                    # print("action-value of each cpu-action:", a_cpu)
                    # print("action-value of each offloading-action:", a_off)
                    action_vec = []
                    for i in range(n_rsu):
                        for actions in range(len(a_cpu[i])):
                            # TODO: try to change q_a_cpu from using alpha to be a mean
                            a_cpu[i][actions] = q_a_cpu[i][actions] + math.sqrt(2 * math.log(t) / n_t_cpu[i][actions])
                            wandb.log({f"a_cpu[{i}][{actions}]": a_cpu[i][actions]}, commit=False)
                            wandb.log({f"q_a_cpu[{i}][{actions}]": q_a_cpu[i][actions]}, commit=False)
                            uncertainty = math.sqrt(2 * math.log(t) / n_t_cpu[i][actions])
                            wandb.log({f"uncertainty-cpu[{i}][{actions}]": uncertainty}, commit=False)

                            # print("action-value of each cpu-action for rsu ", i, ":", a_cpu[i])
                            # print("action-value of each cpu-action for rsu ", i, " for a single actions with index:", actions, ":", a_cpu[i][actions])
                        # max_cpu_index = a_cpu[i].index(max(a_cpu[i]))
                        max_cpu_index = random.choice([k for k in range(len(a_cpu[i])) if a_cpu[i][k] == max(a_cpu[i])])
                        for actions in range(len(a_off[i])):
                            a_off[i][actions] = q_a_offloading[i][actions] + math.sqrt(2 * math.log(t) / n_t_offloading[i][actions])
                            wandb.log({f"a_off[{i}][{actions}]": a_off[i][actions]}, commit=False)
                            wandb.log({f"q_a_offloading[{i}][{actions}]": q_a_offloading[i][actions]}, commit=False)
                            uncertainty = math.sqrt(2 * math.log(t) / n_t_offloading[i][actions])
                            wandb.log({f"uncertainty-off[{i}][{actions}]": uncertainty}, commit=False)
                            # print("action-value of each off-action for rsu ", i, ":", a_off[i])
                            # print("action-value of each off-action for rsu ", i, " for a single actions with index:", actions, ":", a_off[i][actions])
                        # max_off_index = a_off[i].index(max(a_off[i]))
                        max_off_index = random.choice([k for k in range(len(a_off[i])) if a_off[i][k] == max(a_off[i])])

                        action_vec.append([max_cpu_index, max_off_index])
                        # print("action vec building:", action_vec)

                    print("action vec:", action_vec)
                    _, rewards, done, _ = env.step(action_vec)

                    # reward = -1

                    for i in range(n_rsu):
                        reward = rewards[i]

                        max_cpu_index = action_vec[i][0]
                        max_off_index = action_vec[i][1]
                        q_a_cpu[i][max_cpu_index] = (1 - alpha) * q_a_cpu[i][max_cpu_index] + (alpha * (reward - q_a_cpu[i][max_cpu_index]))
                        q_a_offloading[i][max_off_index] = (1 - alpha) * q_a_offloading[i][max_off_index] + (alpha * (reward - q_a_offloading[i][max_off_index]))
                        n_t_cpu[i][max_cpu_index] += 1
                        n_t_offloading[i][max_off_index] += 1
                        # print(f"n_t_cpu[{i}]:", n_t_cpu[i])
                        # print(f"n_t_offloading[{i}]:", n_t_offloading[i])
                        wandb.log({f"n_t_cpu[{i}][{max_cpu_index}]":n_t_cpu[i][max_cpu_index]}, commit=False)
                        wandb.log({f"n_t_offloading[{i}][{max_off_index}]": n_t_offloading[i][max_off_index]}, commit=False)

            wandb.finish()
        n_zones_index += 1
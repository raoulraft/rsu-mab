from mab_env import VehicularParallelEnv as MultiEnv
import wandb
from mab import EpsGreedy
project_name = "edge-vehicular-network-rsu-mab"

rsu_scenario = [4]
cpu_scenario = [4]
lmda_zones = [[0.4, 3.8, 0.9, 4.1], ]  # [[1.4, 0.8, 0.9, 2.1], ]
epsilon_greedy = 0.05

for _ in range(3):
    for reward_function in [2]:
        n_zones_index = 0
        for n_rsu in rsu_scenario:

            for cpu in cpu_scenario:

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

                        _, rewards, done, _ = env.step(action_vec)

                        for i in range(n_rsu):
                            reward = rewards[i]
                            best_cpu_arm = action_vec[i][0]
                            best_offloading_arm = action_vec[i][1]
                            cpu_agents[i].update(best_cpu_arm, reward)
                            wandb.log({f"cpu-agents[{i}][{best_cpu_arm}].AM_reward": cpu_agents[i].AM_reward[best_cpu_arm]},commit=False)
                            wandb.log(
                                {f"offloading-agents[{i}][{best_offloading_arm}].AM_reward": offloading_agents[i].AM_reward[best_offloading_arm]},
                                commit=False)
                            offloading_agents[i].update(best_offloading_arm, reward)

                        for i in range(4):
                            wandb.log({f"reward_pdf[{i}]": reward_pdf[i]}, commit=False)

                wandb.finish()
            n_zones_index += 1
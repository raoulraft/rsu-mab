from mab_env_offloading import VehicularParallelEnv as MultiEnv
import wandb
from mab import MAB, UCB, EpsGreedy
project_name = "edge-vehicular-network-rsu-mab-offloading-only"

rsu_scenario = [4]
cpu_scenario = [4]
lmda_zones = [[0.4, 1.3, 0.2, 0.5], ]  # [[1.4, 0.8, 0.9, 2.1], ]
epsilon_greedy = 0.2
is_ucb = True

for reward_function in [0, 1, 2]:
    for use_epsilon in [True, False]:
        for is_ucb in [True, False]:
            n_zones_index = 0
            for n_rsu in rsu_scenario:
                for cpu in cpu_scenario:

                    offloading_agents = [EpsGreedy(11, epsilon_greedy) for _ in range(n_rsu)] if not is_ucb else [UCB(11, 2) for _ in range(n_rsu)]

                    t = 0
                    env = MultiEnv(n_rsu, cpu, lmda_zones[n_zones_index], reward_mode=reward_function, use_epsilon=use_epsilon)

                    wandb.init(
                        project=project_name,
                        name=f"reward {reward_function} ucb {is_ucb} epsilon {use_epsilon} n {n_rsu}",
                        tags=["n {}".format(n_rsu),
                              "cpu {}".format(cpu),
                              "reward mode {}".format(reward_function),
                              "is_ucb {}".format(is_ucb),
                              "use_epsilon {}".format(use_epsilon),
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
                                best_cpu_arm = 0
                                best_offloading_arm = offloading_agents[i].choose_arm(t)

                                action_vec.append([best_cpu_arm, best_offloading_arm])

                            _, rewards, done, _ = env.step(action_vec)

                            for i in range(n_rsu):

                                reward = rewards[i]
                                best_offloading_arm = action_vec[i][1]
                                offloading_agents[i].update(best_offloading_arm, reward)

                    wandb.finish()
                n_zones_index += 1
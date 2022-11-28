class RSU:
    def __init__(self, starting_ce, starting_off_prob, threshold_queue, threshold_battery,
                 battery_recharge_rate, battery_depletion_rate, proc_rate, n_cpus_max, queue_max_size,
                 battery_max_size):
        self.ce = starting_ce
        self.off_prob = starting_off_prob
        self.threshold_queue = threshold_queue
        self.threshold_battery = threshold_battery

        self.battery_recharge_rate = battery_recharge_rate
        self.battery_depletion_rate = battery_depletion_rate

        self.proc_rate = proc_rate
        self.n_cpus_max = n_cpus_max
        self.queue_max_size = queue_max_size
        self.battery_max_size = battery_max_size

    def get_ce(self):
        return self.ce

    def set_ce(self, ce):
        self.ce = ce + 1  # action [0, 1] -> [1, 2] CEs

    def set_off_prob(self, off_prob):
        self.off_prob = off_prob / 10

    def get_off_prob(self):
        return self.off_prob
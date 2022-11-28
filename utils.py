import numpy as np


def p0(in_rate, proc_rate, CE, K):
    p01 = 0
    p02 = 0
    for k in range(CE + 1):
        p01 = p01 + ((np.power(in_rate, k, dtype=np.longdouble)) / (
                    np.power(proc_rate, k, dtype=np.longdouble) * np.math.factorial(k)))

    for k in range(CE + 1, K + 1):
        p02 = p02 + ((np.power(in_rate, k - CE, dtype=np.longdouble)) / (
                    np.power(proc_rate, k - CE, dtype=np.longdouble) * np.power(CE, k - CE, dtype=np.longdouble)))

    c02 = ((np.power(in_rate, CE, dtype=np.longdouble)) / (
                np.power(proc_rate, CE, dtype=np.longdouble) * np.math.factorial(CE)))
    return np.power(p01 + (c02 * p02), -1, dtype=np.longdouble)


def pk(in_rate, proc_rate, CE, K, k):
    if (k <= CE):
        return (np.power((in_rate / proc_rate), k, dtype=np.longdouble) / (np.math.factorial(k))) * p0(in_rate,
                                                                                                       proc_rate, CE, K)
    else:
        rho = in_rate / (CE * proc_rate)
        return (np.power((in_rate / proc_rate), k, dtype=np.longdouble) / (
                    np.math.factorial(CE) * np.power(rho, (CE - k), dtype=np.longdouble))) * p0(in_rate, proc_rate, CE,
                                                                                                K)


# returns a probability > 1 if rho is > 1. Therefore, we clip it to 1
def get_overload(lamda, mu, CE, K, k_th):
    sum_p = 0
    for k in range(k_th, K + 1):
        sum_p += pk(lamda, mu, CE, K, k)

    sum_p = 1 if sum_p >= 1 else sum_p
    return sum_p


# probability of battery depletion for a single RSU
def get_depletion(lamda, mu, CE, K, k_th):
    # print("lamda:", lamda, "mu:", mu, "CE:", CE, "K:", K, "k_th:", k_th)
    sum_p = 0
    for k in range(k_th + 1):
        sum_p = sum_p + pk(lamda, mu, CE, K, k)
    return sum_p


# probability to experience a latency bigger than tau_th slots
def get_latency(lamda, mu, CE, K, tau_th):
    prob = 0
    for k in range(K + 1):
        if (np.ceil(k / CE) + 1) > tau_th:
            prob += pk(lamda, mu, CE, K, k)
    prob = 1 if prob >= 1 else prob
    return prob


def get_lmda(offloading_probabilities, rsu_id, lmda_zones, computing_elements, mu):
    rsu_offloading_prob = offloading_probabilities[rsu_id]

    # it requires that proc_rate (mu) is the same for all RSUs. Might fix it in the future
    load_rsu = [lmda_zone / (computing_element * mu) for lmda_zone, computing_element in
                zip(lmda_zones, computing_elements)]
    weight_rsu = load_rsu[rsu_id] / sum(load_rsu)
    lmda_off = [lmda_zone * prob_off for lmda_zone, prob_off in zip(lmda_zones, offloading_probabilities)]
    lmda_others = [lmda_offloading for i, lmda_offloading in enumerate(lmda_off) if i != rsu_id]
    sum_other_offloadings = sum(lmda_others)
    lmda = lmda_zones[rsu_id] * (1 - rsu_offloading_prob) + (1 - weight_rsu) * sum_other_offloadings
    return lmda
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import trange
import os
from pathlib import Path
import glob
from PIL import Image

with open("parameters.json") as json_parameters:
    parameters = json.load(json_parameters)

packet_rate = parameters['packet_rate']  # lambda (paquets par seconde)
n_classes = parameters['n_classes']  # nombre de classes de jobs
reward = parameters['rewards']  # rewards par classe
p = [d[0] / d[1] for d in parameters['distribution']]  # probabilité par classe qu'un job soit de cette classe
assert sum(p) == 1, "La somme des probabilité doit être égale à 1"
capacity = parameters['capacity']

# actions
ADMIT = 1
REJECT = 0
actions = {ADMIT, REJECT}  # Set of actions


def fractional_knapsack_problem():
    """
    Determine the optimal stationary threshold policy for the underlying MDP by solving the fractional knapsack problem.
    Wi = Pi: poids du packet i = probabilité du paquet i
    Ci = PiRi: valeur du packet i = proba * reward du paquet i
    -> Ci / Wi = PiRi / Pi = Ri
    L> On sort par reward
    :return:
    """
    # tri des numéro de classes par reward
    class_order = [i for _, i in sorted(zip(reward, range(n_classes)), reverse=True)]
    required_capacity = 0  # cumul des capacités requises par classe admises
    policy = [REJECT for _ in range(n_classes)]
    for class_num in class_order:
        if p[class_num] + required_capacity < capacity:
            required_capacity += p[class_num]
            policy[class_num] = ADMIT
        else:
            policy[class_num] = (capacity - required_capacity) / p[class_num]
            break
    estimated_reward = sum([_p * _policy * r for _p, _policy, r in zip(p, policy, reward)]) * packet_rate

    return {"policy_per_class": policy,
            "estimated_reward": estimated_reward}


def robinson_monro(epsilon="dynamic",
                   polyak_average_window=1,
                   nb_episodes=1000,
                   episode_duration=1,  # en seconde
                   discount=1 / 2,  # discount
                   c=capacity,  # 0.34 # capacity  # c >= 0
                   alpha0=2.5):
    epsillon_type = epsilon
    alpha = alpha0
    assert alpha < n_classes, "Le seuil doit être inférieur au nombre de classe"

    nb_episode_packet = episode_duration * packet_rate

    policies = []  # track of policy
    alphas = [alpha]
    tbar = trange(1, nb_episodes + 1)
    tbar.set_description(f"plot__epsillon_{epsillon_type}-polyakWindow_{polyak_average_window}"
                         f"-nEpisode_{nb_episodes}_durationEp_{episode_duration}-c_{c}"
                         f"-discount_{discount}-alpha0_{alpha0}")
    for num_episode in tbar:
        # policy definition
        policy = np.arange(1, n_classes + 1)
        policy = np.where(policy <= int(alpha), 1, 0)
        policy = policy.tolist()
        if int(alpha) < n_classes:
            policy[int(alpha)] = alpha - int(alpha)
        policies.append(policy)

        # new sample
        new_packets = np.array(np.random.choice(np.arange(0, n_classes), size=int(nb_episode_packet), p=p))

        nb_admitted = 0
        for class_num in range(n_classes):
            nb_admitted += np.where(new_packets == class_num, 1, 0).sum() * policy[class_num]
        Y = nb_admitted / nb_episode_packet

        # iterate
        if epsillon_type == "dynamic":
            epsilon = c / (num_episode ** discount)
        alpha_mean = np.mean(alphas[-polyak_average_window:])
        alpha = max(0, min(alpha_mean + epsilon * (c - Y), n_classes))
        alphas.append(alpha)
        # print("alpha", alpha)

    figname = f"_epsillon_{epsillon_type}-polyakWindow_{polyak_average_window}" \
              f"-nEpisode_{nb_episodes}_durationEp_{episode_duration}-c_{c}" \
              f"-discount_{discount}-alpha0_{alpha0}"
    plot_policies(policies, figname=figname)

    estimated_reward = sum([_p * _policy * r for _p, _policy, r in zip(p, policy, reward)]) * packet_rate
    return {"policy_per_class": policy,
            "estimated_reward": estimated_reward}


def plot_policies(policies, figname):
    figpath = Path("plot") / f"plot_{figname}" / "figs"
    policies = np.array(policies).T

    os.makedirs(figpath, exist_ok=True)
    tbar = trange(1, 101)
    tbar.set_description("Exporting plots")
    for window in tbar:
        for i in range(n_classes):
            plt.plot(policies[i, :int((window / 100) * policies.shape[1])], label=f"class {i}")
        plt.legend()
        plt.savefig(figpath / f"wind_{str(window).rjust(3, '0')}%.png")
        plt.title(figname)
        plt.close()
    make_gif(figpath, figname)


def make_gif(figpath, figname):
    images = glob.glob(str(figpath / "*.png"))
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    os.makedirs(figpath.parent.parent.parent / "gifs", exist_ok=True)
    frame_one.save(figpath.parent.parent.parent / "gifs" / f"gif_{figname}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)


def makeGrid(pars_dict):
    keys = pars_dict.keys()
    combinations = itertools.product(*pars_dict.values())
    ds = [dict(zip(keys, cc)) for cc in combinations]
    return ds


def fct(epsilon="dynamic",
        polyak_average_window=1,
        nb_episodes=1000,
        episode_duration=1,  # en seconde
        discount=1 / 2,  # discount
        c=capacity,  # 0.34 # capacity  # c >= 0
        alpha0=2.5):
    print(epsilon, polyak_average_window, nb_episodes, episode_duration, discount, c, alpha0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # pprint(fractional_knapsack_problem())
    grid_search_params = {"epsilon": ["dynamic", 0.5, 1],
                          "polyak_average_window": [1, 10, 100],
                          "nb_episodes": [1000],
                          "episode_duration": [1],
                          "discount": [1 / 2],
                          "c": [capacity],
                          "alpha0": [2.5, 0]}

    params = makeGrid(grid_search_params)
    for param in params:
        pprint(robinson_monro(**param))

    # pprint(robinson_monro(polyak_average_window=1, epsilon=0.5, nb_episodes=100))
    # make_gif(Path("/Users/nathanael.l/SynchroDir/Etudes/Cours et TP/M2 Informatique CMI (2021-2022)/Reenforcement Learning/TP1_optimal_traffic_admission_control/plot/plot_nEpisode_1000-c_0.34-discount_0.5-alpha0_1.2991978188022562"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

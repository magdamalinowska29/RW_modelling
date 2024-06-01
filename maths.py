
import random

from scipy.stats import truncnorm


def choose(p):

    """p: probability distribution, list of float"""

    rand_num=random.random()
    #print(rand_num)

    sum=0

    choice_ind=0

    for option_ind in range(len(p)):
        sum += p[option_ind]
        if rand_num<=sum:
            choice_ind=option_ind

            break



    return choice_ind

def generate_reward(mu, a):

    """a function to generate reward outcomes based on participant's choices

    parameters:

        mu: list tof floats, length 3, reward probability of each bandit

        a: int, inde of the chosen option

    returns:

        0 if there was no reward
        1 if there was a reward"""

    rand_num=random.random()

    rew_prob = mu[int(a)]  # get the probability of reward for the chosen option

    #print(rand_num)

    if rand_num <= rew_prob:
        return 1
    else:
        return 0


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

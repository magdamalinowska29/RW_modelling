

import numpy as np
import random

from maths import choose

Q = [0.5, 0.5,0.5];  # initial value of each option

beta=10
T=10
mu=[0.5, 0.5, 0.5]
alpha=0.5

def choose(p):

    """p: probability distribution, list of float"""

    rand_num=random.random()
    print(rand_num)

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

    rew_prob=mu[int(a)] # get the probability of reward for the chosen option

    print(rand_num)

    if rand_num<=rew_prob:
        return 1
    else:
        return 0




"""loop over the trials"""

a=np.zeros(len(range(T))) #to hold answers
r=np.zeros(len(range(T))) # to hold reward outcomes

for t in range(T):

    # calculate the probability of each choice

    Q_b =np.zeros(len(Q))

    for j in range(len(Q)):
        Q_b[j] =Q[j] *beta

    print(Q_b)

    p = np.exp(Q_b) / sum(np.exp(Q_b))

    print(p)

    #make choice according to choice probabilities

    a[t]=choose(p)
    print(a)

    #generate the reward based on the chocie

    r[t]=generate_reward(mu, a[t])

    print(r)

    #update values


    delta= r[t]-Q[int(a[t])]
    print('delta')
    print(delta)

    Q[int(a[t])]=Q[int(a[t])] + alpha*delta


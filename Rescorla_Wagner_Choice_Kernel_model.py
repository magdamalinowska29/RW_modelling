import numpy as np
from maths import choose, generate_reward, get_truncated_normal
from scipy.optimize import minimize
import math
import random


from math import inf


class Rescorla_Wagner_CK_model():

    def simulate(self, T, mu, alpha, beta, alpha_c, beta_c):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: float, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                    alpha_c: float between 0 and 1, learning rate for choice kernel

                    beta_c: float, temperature for the choice kelner

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        CK = [0, 0, 0];  #initial choice kernel for 3 options

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        for t in range(T):

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):

                Q_b[j]=Q[j]*beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


                # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])


            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #update kernel

            CK_new=np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k]=(1-alpha_c)*CK[k]

            CK=CK_new

            CK[int(a[t])] = CK[int(a[t])] + alpha_c * 1



        return a,r




    def simulate_accuracy(self, T, mu, alpha, beta, alpha_c, beta_c):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: float, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                    alpha_c: float between 0 and 1, learning rate for choice kernel

                    beta_c: float, temperature for the choice kelner

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        CK = [0, 0, 0];  #initial choice kernel for 3 options

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        best=np.zeros(len(range(T)))
        mid=np.zeros(len(range(T)))
        worst=np.zeros(len(range(T)))

        for t in range(T):

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):

                Q_b[j]=Q[j]*beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


                # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])

            # classify as best, mid or worst
            if mu[int(a[t])] == max(mu):
                best[t] = 1
            elif mu[int(a[t])] == min(mu):
                worst[t] = 1
            else:
                mid[t] = 1

            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #update kernel

            CK_new=np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k]=(1-alpha_c)*CK[k]

            CK=CK_new

            CK[int(a[t])] = CK[int(a[t])] + alpha_c * 1


        return a, r, best, mid, worst


    def simulate_change_accuracy(self, T, mu, alpha, beta, alpha_c, beta_c, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: float, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                    alpha_c: float between 0 and 1, learning rate for choice kernel

                    beta_c: float, temperature for the choice kelner

                    changepoints: list of int, trials at which the stimulus was changed

                     new_pos: list of lists, the positions at which the stimulus is being chnaged

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        CK = [0, 0, 0];  #initial choice kernel for 3 options

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        best=np.zeros(len(range(T)))
        mid=np.zeros(len(range(T)))
        worst=np.zeros(len(range(T)))

        curr_change = 0

        for t in range(T):

            # if its a changepoint, reset the vales of changed options

            pos_to_change = []
            prob_to_change = []
            if t in changepoints:
                pos_to_change = new_pos[curr_change]
                prob_to_change = new_probs[curr_change]
                curr_change += 1

            for pos in pos_to_change:

                prob = 0
                if len(pos_to_change) == 1:
                    prob = prob_to_change[0]
                else:
                    prob = prob_to_change[pos - 1]

                mu[pos - 1] = prob
                Q[pos - 1] = 0.5

                CK[pos-1]=0
                # print(prob)

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):

                Q_b[j]=Q[j]*beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


                # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])

            # classify as best, mid or worst
            if mu[int(a[t])] == max(mu):
                best[t] = 1
            elif mu[int(a[t])] == min(mu):
                worst[t] = 1
            else:
                mid[t] = 1

            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #update kernel

            CK_new=np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k]=(1-alpha_c)*CK[k]

            CK=CK_new

            CK[int(a[t])] = CK[int(a[t])] + alpha_c * 1


        return a, r, best, mid, worst



    def likelihood(self, a, r, alpha, beta, alpha_c, beta_c):

        """a function that calculates the likelihood of choosing the actual chosen option

        Parameters:

            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

            alpha: float between 0 and 1, learning rate

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5];  # initial value of each option

        CK = [0, 0, 0];  # initial choice kernel for 3 options

        T=len(a)

        choice_prob=np.zeros(len(a))

        """loop over trials"""

        for t in range(T):

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            # update kernel

            CK_new = np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k] = (1 - alpha_c) * CK[k]

            CK = CK_new

            CK[int(a[t])] = CK[int(a[t])] + alpha_c * 1

        #compute negative log - likelihood


        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_pav(self, a, a_comp, r, alpha, beta, alpha_c, beta_c):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian learning

        Parameters:

            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

            alpha: float between 0 and 1, learning rate

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5] # initial value of each option

        CK = [0, 0, 0] # initial choice kernel for 3 options

        T=len(a)

        choice_prob=np.zeros(len(a))

        """loop over trials"""

        for t in range(T):

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a_comp[t])]


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + alpha * delta

            # update kernel

            CK_new = np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k] = (1 - alpha_c) * CK[k]

            CK = CK_new

            CK[int(a_comp[t])] = CK[int(a_comp[t])] + alpha_c * 1

        #compute negative log - likelihood


        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change(self, a, r, alpha, beta, alpha_c, beta_c, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option

        Parameters:

            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

            alpha: float between 0 and 1, learning rate

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

            changepoints: list of int, trials at which the stimulus was changed

            new_pos: list of lists, the positions at which the stimulus is being chnaged

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5];  # initial value of each option

        CK = [0, 0, 0];  # initial choice kernel for 3 options

        T=len(a)

        choice_prob=np.zeros(len(a))

        """loop over trials"""

        for t in range(T):

            """accounting for changepoints"""
            curr_change = 0

            # if its a changepoint, reset the vales of changed options

            pos_to_change = []
            if t in changepoints:
                pos_to_change = new_pos[curr_change]
                curr_change += 1

            for pos in pos_to_change:
                pos_new = pos - 1
                Q[pos_new] = 0.5

                CK[pos_new]=0

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            # update kernel

            CK_new = np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k] = (1 - alpha_c) * CK[k]

            CK = CK_new

            CK[int(a[t])] = CK[int(a[t])] + alpha_c * 1

        #compute negative log - likelihood


        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change_pav(self, a, a_comp, r, alpha, beta, alpha_c, beta_c, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian learning

        Parameters:

            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

            alpha: float between 0 and 1, learning rate

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

            changepoints: list of int, trials at which the stimulus was changed

            new_pos: list of lists, the positions at which the stimulus is being chnaged

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5]  # initial value of each option

        CK = [0, 0, 0]  # initial choice kernel for 3 options

        T=len(a)

        choice_prob=np.zeros(len(a))

        """loop over trials"""

        for t in range(T):

            """accounting for changepoints"""
            curr_change = 0

            # if its a changepoint, reset the vales of changed options

            pos_to_change = []
            if t in changepoints:
                pos_to_change = new_pos[curr_change]
                curr_change += 1

            for pos in pos_to_change:
                pos_new = pos - 1
                Q[pos_new] = 0.5

                CK[pos_new]=0

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta + CK[j]* beta_c #the same as in rascola wagener, but also taking CK in consideration

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a_comp[t])]


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + alpha * delta

            # update kernel

            CK_new = np.zeros(len(CK))

            for k in range(len(CK)):
                CK_new[k] = (1 - alpha_c) * CK[k]

            CK = CK_new

            CK[int(a_comp[t])] = CK[int(a_comp[t])] + alpha_c * 1

        #compute negative log - likelihood


        NegLL = -sum(np.log(choice_prob))

        return NegLL


    def fit(self, a, r):

        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood(a, r, x[0], x[1], x[2], x[3])

        #init_guess_1=random.random()

        #init_guess_2=np.random.exponential(1)

        #init_guess_3 = random.random()

        #init_guess_4 = np.random.exponential(1)

        init_guess_1 = random.uniform(0, 1)
        init_guess_2 = np.random.exponential(1)+1
        init_guess_3 = random.uniform(0, 1)
        init_guess_4 = np.random.exponential(1)+1

        X0 = [init_guess_1, init_guess_2, init_guess_3, init_guess_4];


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(1,math.inf),(0,1),(1,math.inf)])
        #res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL=res.fun

        Xfit=res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_pav(self, a, a_comp, r):

        """a function to fit the parameters to the data in pavlovian learning

        Parameters:
            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_pav(a, a_comp, r, x[0], x[1], x[2], x[3])

        #init_guess_1=random.random()

        #init_guess_2=np.random.exponential(1)

        #init_guess_3 = random.random()

        #init_guess_4 = np.random.exponential(1)

        init_guess_1 = random.uniform(0, 1)
        init_guess_2 = np.random.exponential(1)+1
        init_guess_3 = random.uniform(0, 1)
        init_guess_4 = np.random.exponential(1)+1

        X0 = [init_guess_1, init_guess_2, init_guess_3, init_guess_4];


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(1,math.inf),(0,1),(1,math.inf)])
        #res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL=res.fun

        Xfit=res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC


    def fit_change(self, a, r, changepoints, new_pos):
        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc = lambda x: self.likelihood_change(a, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=x[1], alpha_c=x[2], beta_c=x[3])

        '''
        init_guess_1 = random.random()
        init_guess_2 = np.random.exponential(1)+1
        init_guess_3 = random.random()
        init_guess_4 = np.random.exponential(1)+1'''


        #init_guess_1 = random.uniform(0.2, 1)
        #init_guess_2 = random.uniform(2, 12)  # draw values from th distribution based on the subject fits
        #init_guess_3 = random.uniform(0.2, 1)
        #init_guess_4 = random.uniform(2, 12)

        alpha_X = get_truncated_normal(mean=0.5664515917, sd=0.3169843578, low=0, upp=1)
        init_guess_1 = alpha_X.rvs()

        beta_X = get_truncated_normal(mean=5.174611978, sd=2.975011374, low=1, upp=100)
        init_guess_2 = beta_X.rvs()

        alpha_X_c = get_truncated_normal(mean=0.3128277583, sd=0.1547298159, low=0, upp=1)
        init_guess_3 = alpha_X_c.rvs()

        beta_X_c = get_truncated_normal(mean=3.279238993, sd=2.056496848, low=1, upp=100)
        init_guess_4 = beta_X_c.rvs()

        X0 = [init_guess_1, init_guess_2, init_guess_3, init_guess_4];


        res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B', bounds=[(0, 1), (1,100), (0, 1), (1,100)])
        # res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL = res.fun

        Xfit = res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_pav(self, a, a_comp, r, changepoints, new_pos):
        """a function to fit the parameters to the data in pavlovian learning

        Parameters:
            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc = lambda x: self.likelihood_change_pav(a, a_comp, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=x[1], alpha_c=x[2], beta_c=x[3])

        '''
        init_guess_1 = random.random()
        init_guess_2 = np.random.exponential(1)+1
        init_guess_3 = random.random()
        init_guess_4 = np.random.exponential(1)+1'''


        init_guess_1 = random.uniform(0.2, 1)
        init_guess_2 = random.uniform(2, 12)  # draw values from th distribution based on the subject fits
        init_guess_3 = random.uniform(0.2, 1)
        init_guess_4 = random.uniform(2, 12)
        '''
        alpha_X = get_truncated_normal(mean=0.4922177625, sd=0.2531502716, low=0, upp=1)
        init_guess_1 = alpha_X.rvs()

        beta_X = get_truncated_normal(mean=4.520888457, sd=3.369674226, low=1, upp=100)
        init_guess_2 = beta_X.rvs()

        alpha_X_c = get_truncated_normal(mean=0.3044229358, sd=0.176750516, low=0, upp=1)
        init_guess_3 = alpha_X_c.rvs()

        beta_X_c = get_truncated_normal(mean=3.273113367, sd=2.063707511, low=1, upp=100)
        init_guess_4 = beta_X_c.rvs()'''

        X0 = [init_guess_1, init_guess_2, init_guess_3, init_guess_4];


        res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B', bounds=[(0.2, 1), (2,12), (0.2, 1), (2,12)])
        # res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL = res.fun

        Xfit = res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC


#a,r = simulate(100,[0.2,0.5,0.8],0.5,10,0.2,2)

#NegLL=likelihood(a,r,0.5,10,0.2,2)

#Xfit,LL, BIC=fit(a,r)

#print(NegLL)

#print(Xfit)
#print(LL)




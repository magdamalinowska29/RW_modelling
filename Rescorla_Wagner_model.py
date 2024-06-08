

import numpy as np
from maths import choose, generate_reward, get_truncated_normal
from scipy.optimize import minimize
import math
import random
import matplotlib.pyplot as plt


from math import inf


class Rescorla_Wagner_model():





    def simulate(self, T, mu, alpha, beta):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: int, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        for t in range(T):

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j]=Q[j]*beta



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


                # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])


            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

        return a,r



    def simulate_change(self, T, mu, alpha, beta, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: int, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                     changepoints: list of int, trials at which the stimulus was changed

                     new_pos: list of lists, the positions at which the stimulus is being chnaged

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        curr_change=0
        for t in range(T):

            #if its a changepoint, reset the vales of changed options

            pos_to_change=[]
            prob_to_change = []
            if t in changepoints:
                pos_to_change=new_pos[curr_change]
                prob_to_change=new_probs[curr_change]
                curr_change+=1

            for pos in pos_to_change:

                prob=0
                if len(pos_to_change)==1:
                    prob=prob_to_change[0]
                else:
                    prob=prob_to_change[pos-1]

                mu[pos-1]=prob
                Q[pos-1]=0.5
                #print(prob)

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j]=Q[j]*beta



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


                # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])


            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

        return a,r

    def simulate_change_accuracy(self, T, mu, alpha, beta, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model #TODO: try with seperate learning rates and seperate beta for the new trial

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: int, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                     changepoints: list of int, trials at which the stimulus was changed

                     new_pos: list of lists, the positions at which the stimulus is being chnaged

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        best=np.zeros(len(range(T)))
        mid=np.zeros(len(range(T)))
        worst=np.zeros(len(range(T)))

        curr_change=0
        for t in range(T):

            #if its a changepoint, reset the vales of changed options

            pos_to_change=[]
            prob_to_change = []
            if t in changepoints:
                pos_to_change=new_pos[curr_change]
                prob_to_change=new_probs[curr_change]
                curr_change+=1

            for pos in pos_to_change:

                prob=0
                if len(pos_to_change)==1:
                    prob=prob_to_change[0]
                else:
                    prob=prob_to_change[pos-1]

                mu[pos-1]=prob
                Q[pos-1]=0.5
                #print(prob)

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j]=Q[j]*beta



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


            # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])


            #classify as best, mid or worst
            if mu[int(a[t])]==max(mu):
                best[t]=1
            elif mu[int(a[t])]==min(mu):
                worst[t]=1
            else:
                mid[t]=1

            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

        return a,r, best, mid, worst

    def simulate_change_test(self, T, mu, alpha, beta, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha: float between 0 and 1, learning rate

                    beta: int, reverse temperature, controls the level of stochasticity of the choice. 0 means completelt random
                     responding, high numbers menas deterministicallt choosing the option with highest reward values

                     changepoints: list of int, trials at which the stimulus was changed

                     new_pos: list of lists, the positions at which the stimulus is being chnaged

                returns:
                    a: list of ints, choices
                    r: list of boolean, reward outcomes"""

        Q = [0.5, 0.5,0.5]; #initial value of each option

        """loop over the trials"""

        a = np.zeros(len(range(T)))  # to hold answers
        r = np.zeros(len(range(T)))  # to hold reward outcomes

        curr_change=0

        new_pos=[]
        for t in range(T):

            #if its a changepoint, reset the vales of changed options


            if t in changepoints:
                rand=random.randint(0,2)
                pos_to_change=rand
                new_pos.append(pos_to_change)
                probs_poss=random.random()
                prob_to_change=probs_poss
                curr_change+=1



                mu[pos_to_change]=prob_to_change
                Q[pos_to_change]=0.5
            #print(prob)

            #calculate the probability of each choice

            Q_b=np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j]=Q[j]*beta



            p = np.exp(Q_b) / sum(np.exp(Q_b))


            # make choice according to choice probabilities

            a[t] = choose(p)


                # generate the reward based on the chocie

            r[t] = generate_reward(mu, a[t])


            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

        return a,r, new_pos




    def likelihood(self,a,r, alpha, beta):

        """a function that calculates the likelihood of choosing the actual chosen option

        Parameters:

            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

            alpha: float between 0 and 1, learning rate

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5];  # initial value of each option

        T=len(a)

        choice_prob=np.zeros(len(a))

        """loop over trials"""

        for t in range(T):

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_pav(self,a, a_comp, r, alpha, beta):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian learning

        Parameters:

            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward, based on computer's actions

            alpha: float between 0 and 1, learning rate

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5];  # initial value of each option

        T=len(a)

        choice_prob=np.zeros(len(a))

        """loop over trials"""

        for t in range(T):

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values for the option chosen by computer

            delta = r[t] - Q[int(a_comp[t])]


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + alpha * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change(self,a,r, alpha, beta, changepoints, new_pos):

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
                pos_new=pos-1
                Q[pos_new] = 0.5

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        print(NegLL)


        return NegLL

    def likelihood_change_plot_lr(self,a,r, alpha, beta, changepoints, new_pos, base_plot_x, base_plot_y, real_val):

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
                pos_new=pos-1
                Q[pos_new] = 0.5

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        print(NegLL)

        marker_on = [real_val]
        # plt.draw()
        actual_negLL = base_plot_y[int(real_val / 0.01)]
        plt.plot(base_plot_x, base_plot_y, c='blue', markevery=marker_on, label='negative log likelihood')
        plt.scatter(marker_on, [actual_negLL], c='red', s=40)

        plt.title('NegLL for lr= ' + str(real_val))
        plt.xlabel('lr')
        plt.ylabel('NegLL')

        plt.scatter([alpha], [NegLL], s=50, c='purple')

        plt.pause(1)


        return NegLL

    def likelihood_change_plot_temp(self,a,r, alpha, beta, changepoints, new_pos, base_plot_x, base_plot_y, real_val):

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
                pos_new=pos-1
                Q[pos_new] = 0.5

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        print(NegLL)

        marker_on = [real_val]
        # plt.draw()
        actual_negLL = base_plot_y[int(real_val / 0.01)]
        plt.plot(base_plot_x, base_plot_y, c='magenta', markevery=marker_on, label='negative log likelihood')
        plt.scatter(marker_on, [actual_negLL], c='red', s=40)

        plt.title('NegLL for temperature= ' + str(real_val))
        plt.xlabel('temp')
        plt.ylabel('NegLL')

        plt.scatter([beta], [NegLL], s=50, c='purple')

        plt.pause(1)


        return NegLL

    def likelihood_change_pav(self, a, a_comp, r, alpha, beta, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian setting

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

        Q = [0.5, 0.5, 0.5];  # initial value of each option

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
                pos_new=pos-1
                Q[pos_new] = 0.5

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a_comp[t])]


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + alpha * delta

        #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))


        return NegLL


    def likelihood_change_test(self,a,r, alpha, beta, changepoints, new_pos):

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

        Q = [0.5, 0.5, 0.5] # initial value of each option

        T=len(a)

        choice_prob=np.zeros(len(a))



        """loop over trials"""

        for t in range(T):

            """accounting for changepoints"""
            curr_change = 0


             # if its a changepoint, reset the vales of changed options


            if t in changepoints:
                pos_to_change = new_pos[curr_change]


                Q[pos_to_change] = 0.5

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a[t])]


            Q[int(a[t])] = Q[int(a[t])] + alpha * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change_pav_test(self, a, a_comp, r, alpha, beta, changepoints, new_pos):

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

        Q = [0.5, 0.5, 0.5] # initial value of each option

        T=len(a)

        choice_prob=np.zeros(len(a))



        """loop over trials"""

        for t in range(T):

            """accounting for changepoints"""
            curr_change = 0


             # if its a changepoint, reset the vales of changed options


            if t in changepoints:
                pos_to_change = new_pos[curr_change]


                Q[pos_to_change] = 0.5

            # calculate the probability of each choice

            Q_b = np.zeros(len(Q))
            for j in range(len(Q)):
                Q_b[j] = Q[j] * beta

            p = np.exp(Q_b) / sum(np.exp(Q_b))


            choice_prob[t]=p[int(a[t])]


            # update values

            delta = r[t] - Q[int(a_comp[t])]


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + alpha * delta

        #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL




    def fit(self,a,r):

        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood(a, r, x[0], x[1])

        #init_guess_1=random.random()

        init_guess_2=np.random.exponential(1)+1

        init_guess_1 = random.uniform(0, 1)
        #init_guess_2 = random.uniform(2, 12)


        X0 = [init_guess_1, init_guess_2]


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(1,math.inf)])

        NegLL=res.fun

        Xfit=res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2*len(X0) +2 * NegLL

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

        obFunc =lambda x: self.likelihood_pav(a, a_comp, r, x[0], x[1])

        #init_guess_1=random.random()

        init_guess_2=np.random.exponential(1)+1

        init_guess_1 = random.uniform(0, 1)
        #init_guess_2 = random.uniform(2, 12)


        X0 = [init_guess_1, init_guess_2]


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(1,math.inf)])
        #res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL=res.fun

        Xfit=res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2*len(X0) +2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change(self,a,r, changepoints, new_pos):



        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_change(a, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=x[1])

        #obFunc = lambda x: likelihood(a, r, x[0], x[1])

        #init_guess_1=random.random()

        #init_guess_2=np.random.exponential(1)+1

        NegLL = 1000

        for i in range(3):

            init_guess_1 = random.uniform(0.2, 1)
            init_guess_2 = random.uniform(2, 12)

            #init_guess_1 = np.random.normal(loc=0.63, scale=0.2)
            #init_guess_2 = np.random.normal(loc=9, scale=4)

            #init_guess_1_X = get_truncated_normal(mean=0.3083119283, sd=0.1566426521, low=0.2, upp=1)
            #init_guess_1=init_guess_1_X.rvs()

            #init_guess_2_X = get_truncated_normal(mean=7.357686568, sd=4.175147737, low=2, upp=12)
            #init_guess_2=init_guess_2_X.rvs()

            #alpha_X = get_truncated_normal(mean=0.404, sd=0.2739901864, low=0, upp=1)
            #init_guess_1 = alpha_X.rvs()

            #beta_X = get_truncated_normal(mean=6.261, sd=3.91077537, low=1, upp=100)
            #init_guess_2 = beta_X.rvs()

            #init_guess_1 = np.random.normal(loc=0.26051991, scale=0.223883188)
            #init_guess_2 = np.random.normal(loc=13.30012029, scale=11.09702797)

            X0 = [init_guess_1, init_guess_2];


            res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0.2,1),(2, 12)])
            #res = minimize(fun=obFunc, x0=X0, method='SLSQP')

            if res.fun < NegLL:
                NegLL = res.fun

                Xfit = res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_lr_plot(self,a,r, changepoints, new_pos, base_plot_x, baase_plot_y, real_val):



        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_change_plot_lr(a, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=4, base_plot_x=base_plot_x, base_plot_y=baase_plot_y, real_val=real_val)

        #obFunc = lambda x: likelihood(a, r, x[0], x[1])

        init_guess_1=random.random()

        #init_guess_2=np.random.exponential(1)+1

        #init_guess_1 = random.uniform(0.2, 1)
        #init_guess_2 = random.uniform(2, 12)

        #init_guess_1 = np.random.normal(loc=0.63, scale=0.2)
        #init_guess_2 = np.random.normal(loc=9, scale=4)

        #init_guess_1_X = get_truncated_normal(mean=0.3083119283, sd=0.1566426521, low=0.2, upp=1)
        #init_guess_1=init_guess_1_X.rvs()

        #init_guess_2_X = get_truncated_normal(mean=7.357686568, sd=4.175147737, low=2, upp=12)
        #init_guess_2=init_guess_2_X.rvs()



        #init_guess_1 = np.random.normal(loc=0.26051991, scale=0.223883188)
        #init_guess_2 = np.random.normal(loc=13.30012029, scale=11.09702797)

        X0 = [init_guess_1]


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1)])
        #res = minimize(fun=obFunc, x0=X0, method='SLSQP')

        NegLL=res.fun

        Xfit=res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_temp_plot(self,a,r, changepoints, new_pos, base_plot_x, baase_plot_y, real_val):



        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_change_plot_temp(a, r, changepoints=changepoints, new_pos=new_pos,alpha=0.5, beta=x[0], base_plot_x=base_plot_x, base_plot_y=baase_plot_y, real_val=real_val)

        #obFunc = lambda x: likelihood(a, r, x[0], x[1])

        #init_guess_1=random.random()

        init_guess_1=np.random.exponential(1)+1

        #init_guess_1 = random.uniform(0.2, 1)
        #init_guess_2 = random.uniform(2, 12)

        #init_guess_1 = np.random.normal(loc=0.63, scale=0.2)
        #init_guess_2 = np.random.normal(loc=9, scale=4)

        #init_guess_1_X = get_truncated_normal(mean=0.3083119283, sd=0.1566426521, low=0.2, upp=1)
        #init_guess_1=init_guess_1_X.rvs()

        #init_guess_2_X = get_truncated_normal(mean=7.357686568, sd=4.175147737, low=2, upp=12)
        #init_guess_2=init_guess_2_X.rvs()



        #init_guess_1 = np.random.normal(loc=0.26051991, scale=0.223883188)
        #init_guess_2 = np.random.normal(loc=13.30012029, scale=11.09702797)

        X0 = [init_guess_1]


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(1,10)])
        #res = minimize(fun=obFunc, x0=X0, method='SLSQP')

        NegLL=res.fun

        Xfit=res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_pav(self, a, a_comp, r, changepoints, new_pos):



        """a function to fit the parameters to the data in pavlovian learning

        Parameters:
            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by the computer

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_change_pav(a, a_comp, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=x[1])

        #obFunc = lambda x: likelihood(a, r, x[0], x[1])

        #init_guess_1=random.random()

        #init_guess_2=np.random.exponential(1)+1

        init_guess_1 = random.uniform(0.2, 1)
        init_guess_2 = random.uniform(2, 12)

        #init_guess_1 = np.random.normal(loc=0.63, scale=0.2)
        #init_guess_2 = np.random.normal(loc=9, scale=4)

        #init_guess_1_X = get_truncated_normal(mean=0.3083119283, sd=0.1566426521, low=0.2, upp=1)
        #init_guess_1=init_guess_1_X.rvs()

        #init_guess_2_X = get_truncated_normal(mean=7.357686568, sd=4.175147737, low=2, upp=12)
        #init_guess_2=init_guess_2_X.rvs()

        #init_guess_1_X = get_truncated_normal(mean=0.3035292982, sd=0.1600747941, low=0, upp=1)
        #init_guess_1=init_guess_1_X.rvs()

        #init_guess_2_X = get_truncated_normal(mean=7.391652929, sd=4.023906217, low=1, upp=100)
        #init_guess_2=init_guess_2_X.rvs()

        #init_guess_1 = np.random.normal(loc=0.26051991, scale=0.223883188)
        #init_guess_2 = np.random.normal(loc=13.30012029, scale=11.09702797)

        X0 = [init_guess_1, init_guess_2]


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0.2,1),(2, 12)])
        #res = minimize(fun=obFunc, x0=X0, method='SLSQP')

        NegLL=res.fun

        Xfit=res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_test(self,a,r, changepoints, new_pos):



        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_change_test(a, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=x[1])

        #obFunc = lambda x: likelihood(a, r, x[0], x[1])



        init_guess_1 = random.normal(loc=0.26051991, scale=0.223883188)
        init_guess_2 = random.normal(loc=13.30012029, scale=11.09702797)

        X0 = [init_guess_1, init_guess_2];
        LB = (0, 0)
        UB = (1, math.inf)

        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(0,40)])

        NegLL=res.fun

        Xfit=res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC


    def fit_change_test(self, a, a_comp, r, changepoints, new_pos):



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

        obFunc =lambda x: self.likelihood_change_pav_test(a, a_comp, r, changepoints=changepoints, new_pos=new_pos,alpha=x[0], beta=x[1])


        init_guess_1 = random.normal(loc=0.26051991, scale=0.223883188)
        init_guess_2 = random.normal(loc=13.30012029, scale=11.09702797)

        X0 = [init_guess_1, init_guess_2];

        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(0,40)])

        NegLL=res.fun

        Xfit=res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC


"""simulation"""



"""fitting"""

#Xfit,LL=fit(a,r)

#print('Xfit')
#print(Xfit)
#print('LL')
#print(LL)













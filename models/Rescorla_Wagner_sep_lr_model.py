import numpy as np
from models.maths import choose, generate_reward, get_truncated_normal
from scipy.optimize import minimize
import math
import random


class Rescorla_Wagner_sep_lr_model():





    def simulate(self, T, mu, alpha_p, alpha_n, beta):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha_p: float between 0 and 1, learning rate for positive outcomes

                    alpha_n: float between 0 and 1, learning rate for negative outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr=0
            if r[t]==0: #negative (neutral) outcome
                lr=alpha_n
            elif r[t]==1: #positive outcome
                lr=alpha_p

            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + lr * delta

        return a,r

    def simulate_change(self, T, mu, alpha_p, alpha_n, beta, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha_p: float between 0 and 1, learning rate for positive outcomes

                    alpha_n: float between 0 and 1, learning rate for positive outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + lr * delta

        return a,r

    def simulate_change_accuracy(self, T, mu, alpha_p, alpha_n, beta, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model #TODO: try with seperate learning rates and seperate beta for the new trial

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha_p: float between 0 and 1, learning rate for positive outcomes

                    alpha_n: float between 0 and 1, learning rate for negative outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            #classify as best, mid or worst
            if mu[int(a[t])]==max(mu):
                best[t]=1
            elif mu[int(a[t])]==min(mu):
                worst[t]=1
            else:
                mid[t]=1

            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + lr * delta

        return a,r, best, mid, worst

    def simulate_change_test(self, T, mu, alpha_p, alpha_n, beta, changepoints, new_pos, new_probs):
        """a function to simulate the task choices using a rascola wagner model

                Parameters:
                    T: int, number of trials

                    mu: list tof floats, length 3, reward probability of each bandit

                    alpha_p: float between 0 and 1, learning rate for positive outcomes

                    alpha_n: float between 0 and 1, learning rate for negative outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            # update values

            delta = r[t] - Q[int(a[t])]

            Q[int(a[t])] = Q[int(a[t])] + lr * delta

        return a,r, new_pos




    def likelihood(self,a,r, alpha_p, alpha_n, beta):

        """a function that calculates the likelihood of choosing the actual chosen option

        Parameters:

            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

            alpha_p: float between 0 and 1, learning rate for positive outcomes

            alpha_n: float between 0 and 1, learning rate for negative outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            Q[int(a[t])] = Q[int(a[t])] + lr * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL


    def likelihood_pav(self, a, a_comp, r, alpha_p, alpha_n, beta):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian learning

        Parameters:

            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

            alpha_p: float between 0 and 1, learning rate for positive outcomes

            alpha_n: float between 0 and 1, learning rate for negative outcomes

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5]  # initial value of each option

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

            delta = r[t] - Q[int(a_comp[t])]

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + lr * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change(self,a,r, alpha_p, alpha_n, beta, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option

        Parameters:

            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

            alpha_p: float between 0 and 1, learning rate for positive outcomes

            alpha_n: float between 0 and 1, learning rate for negative outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            Q[int(a[t])] = Q[int(a[t])] + lr * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change_pav(self, a, a_comp, r, alpha_p, alpha_n, beta, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian learning

        Parameters:

            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

            alpha_p: float between 0 and 1, learning rate for positive outcomes

            alpha_n: float between 0 and 1, learning rate for negative outcomes

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

            changepoints: list of int, trials at which the stimulus was changed

            new_pos: list of lists, the positions at which the stimulus is being chnaged

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5]  # initial value of each option

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + lr * delta

        #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change_test(self,a,r, alpha_p, alpha_n, beta, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option

        Parameters:

            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

            alpha_p: float between 0 and 1, learning rate for positive outcomes

            alpha_n: float between 0 and 1, learning rate for negative outcomes

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            Q[int(a[t])] = Q[int(a[t])] + lr * delta

            #compute negative log - likelihood

        NegLL = -sum(np.log(choice_prob))

        return NegLL

    def likelihood_change_pav_test(self, a, a_comp, r, alpha_p, alpha_n, beta, changepoints, new_pos):

        """a function that calculates the likelihood of choosing the actual chosen option in pavlovian learning

        Parameters:

            a: list of int, index of the chosen option

            a_comp: list of int, index of the chosen option by computer

            r: list of int, 1 for reward, 0 for no reward

            alpha_p: float between 0 and 1, learning rate for positive outcomes

            alpha_n: float between 0 and 1, learning rate for negative outcomes

            beta: int, 0 to inf, the higher the beta the more determiniistic the choice is

            changepoints: list of int, trials at which the stimulus was changed

            new_pos: list of lists, the positions at which the stimulus is being chnaged

        returns:

        Neg_LL: negative log likelihood"""

        Q = [0.5, 0.5, 0.5]  # initial value of each option

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

            """set the learning rate based on the positive and negative outcomes"""
            lr = 0
            if r[t] == 0:  # negative (neutral) outcome
                lr = alpha_n
            elif r[t] == 1:  # positive outcome
                lr = alpha_p


            Q[int(a_comp[t])] = Q[int(a_comp[t])] + lr * delta

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

        obFunc =lambda x: self.likelihood(a, r, x[0], x[1], x[2])

        #init_guess_1=random.random()
        #init_guess_2 = random.random()
        init_guess_3=np.random.exponential(1)+1

        init_guess_1 = random.uniform(0, 1)
        init_guess_2 = random.uniform(0, 1)
        #init_guess_3 = random.uniform(1, 30)

        X0 = [init_guess_1, init_guess_2, init_guess_3];
        LB = (0, 0)
        UB = (1, math.inf)

        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(0,1),(1,math.inf)])
        #res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL=res.fun

        Xfit=res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC,AIC

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

        obFunc =lambda x: self.likelihood_pav(a, a_comp, r, x[0], x[1], x[2])

        #init_guess_1=random.random()
        #init_guess_2 = random.random()
        init_guess_3=np.random.exponential(1)+1

        init_guess_1 = random.uniform(0, 1)
        init_guess_2 = random.uniform(0, 1)
        #init_guess_3 = random.uniform(1, 30)

        X0 = [init_guess_1, init_guess_2, init_guess_3];
        LB = (0, 0)
        UB = (1, math.inf)

        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(0,1),(1,math.inf)])
        #res = minimize(fun=obFunc, x0=X0, method='L-BFGS-B')

        NegLL=res.fun

        Xfit=res.x

        LL=-NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC,AIC

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

        obFunc =lambda x: self.likelihood_change(a, r, changepoints=changepoints, new_pos=new_pos,alpha_p=x[0], alpha_n=x[1], beta=x[2])


        #init_guess_1=random.random()

        #init_guess_2 = random.random()

        #init_guess_3=np.random.exponential(1)

        NegLL = 1000

        for i in range(3):

            init_guess_1 = random.uniform(0.2, 1)
            init_guess_2 = random.uniform(0.2,1)
            init_guess_3 = random.uniform(2, 12)

            #init_guess_1_X = get_truncated_normal(mean=0.7018621375, sd=0.2555028194, low=0, upp=1)
            #init_guess_1 = init_guess_1_X.rvs()

            #init_guess_2_X = get_truncated_normal(mean=0.3083247317, sd=0.1569015222, low=0, upp=1)
            #init_guess_2 = init_guess_2_X.rvs()

            #init_guess_3_X = get_truncated_normal(mean=5.766115643, sd=2.218477811, low=1, upp=100)
            #init_guess_3 = init_guess_3_X.rvs()

            #init_guess_1 = np.random.normal(loc=0.63, scale=0.2)
            #init_guess_2 = np.random.normal(loc=9, scale=4)

            #init_guess_1_X = get_truncated_normal(mean=0.63, sd=0.25, low=0.1, upp=1)
            #init_guess_1=init_guess_1_X.rvs()

            #init_guess_2_X = get_truncated_normal(mean=8.67, sd=3, low=3, upp=14)
            #init_guess_2=init_guess_2_X.rvs()
            '''
            init_guess_1_X = get_truncated_normal(mean=0.73116224, sd=0.3801940063, low=0, upp=1)
            init_guess_1 = init_guess_1_X.rvs()
    
            init_guess_2_X = get_truncated_normal(mean=0.25169553, sd=0.15508659, low=0, upp=1)
            init_guess_2 = init_guess_2_X.rvs()
    
            init_guess_3_X = get_truncated_normal(mean=3.188618423, sd=1.781965918, low=1, upp=100)
            init_guess_3 = init_guess_3_X.rvs()'''


            X0 = [init_guess_1, init_guess_2, init_guess_3];


            res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0.2,1),(0.2,1),(2,12)])

            if res.fun < NegLL:
                NegLL = res.fun

                Xfit = res.x

        LL = -NegLL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_pav(self, a, a_comp, r, changepoints, new_pos):

        """a function to fit the parameters to the data

        Parameters:
            a: list of int, index of the chosen option

            r: list of int, 1 for reward, 0 for no reward

        Returns:

            Xfit:
            LL:
            BIC:
            """

        obFunc =lambda x: self.likelihood_change_pav(a, a_comp, r, changepoints=changepoints, new_pos=new_pos,alpha_p=x[0], alpha_n=x[1], beta=x[2])


        #init_guess_1=random.random()

        #init_guess_2 = random.random()

        #init_guess_3=np.random.exponential(1)

        init_guess_1 = random.uniform(0.2, 1)
        init_guess_2 = random.uniform(0.2, 1)
        init_guess_3 = random.uniform(2, 12)


        #init_guess_1 = np.random.normal(loc=0.63, scale=0.2)
        #init_guess_2 = np.random.normal(loc=9, scale=4)

        #init_guess_1_X = get_truncated_normal(mean=0.63, sd=0.25, low=0.1, upp=1)
        #init_guess_1=init_guess_1_X.rvs()

        #init_guess_2_X = get_truncated_normal(mean=8.67, sd=3, low=3, upp=14)
        #init_guess_2=init_guess_2_X.rvs()
        '''
        init_guess_1_X = get_truncated_normal(mean=0.73116224, sd=0.3801940063, low=0, upp=1)
        init_guess_1 = init_guess_1_X.rvs()

        init_guess_2_X = get_truncated_normal(mean=0.25169553, sd=0.15508659, low=0, upp=1)
        init_guess_2 = init_guess_2_X.rvs()

        init_guess_3_X = get_truncated_normal(mean=3.188618423, sd=1.781965918, low=1, upp=100)
        init_guess_3 = init_guess_3_X.rvs()'''


        X0 = [init_guess_1, init_guess_2, init_guess_3];


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0.2,1),(0.2,1),(2,12)])

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

        obFunc =lambda x: self.likelihood_change_test(a, r, changepoints=changepoints, new_pos=new_pos,alpha_p=x[0], alpha_n=x[1], beta=x[2])

        #obFunc = lambda x: likelihood(a, r, x[0], x[1])



        #init_guess_1 = random.normal(loc=0.6316284525, scale=0.03535596643)
        #init_guess_2 = random.normal(loc=0.6316284525, scale=0.03535596643)
        #init_guess_3 = random.normal(loc=8.684382077, scale=0.7425075939)

        alpha_X = get_truncated_normal(mean=0.6499729867, sd=0.2368811133, low=0, upp=1)
        init_guess_1 = alpha_X.rvs()

        alpha_X_neg = get_truncated_normal(mean=0.205391265, sd=0.1766854623, low=0, upp=1)
        init_guess_2 = alpha_X_neg.rvs()

        beta_X = get_truncated_normal(mean=5.978394632, sd=3.21938602, low=1, upp=15)
        init_guess_3 = beta_X.rvs()

        X0 = [init_guess_1, init_guess_2, init_guess_3];
        LB = (0, 0)
        UB = (1, math.inf)

        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0,1),(0,1),(0,15)])

        LL=res.fun

        Xfit=res.x

        NegLL = -LL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC

    def fit_change_pav_test(self, a, a_comp, r, changepoints, new_pos):

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

        obFunc =lambda x: self.likelihood_change_pav_test(a, a_comp, r, changepoints=changepoints, new_pos=new_pos,alpha_p=x[0], alpha_n=x[1], beta=x[2])

        init_guess_1 = random.uniform(0.2, 1)
        init_guess_2 = random.uniform(0.2, 1)
        init_guess_3 = random.uniform(2, 12)


        #init_guess_1 = random.normal(loc=0.6316284525, scale=0.03535596643)
        #init_guess_2 = random.normal(loc=0.6316284525, scale=0.03535596643)
        #init_guess_3 = random.normal(loc=8.684382077, scale=0.7425075939)
        '''
        alpha_X = get_truncated_normal(mean=0.6499729867, sd=0.2368811133, low=0, upp=1)
        init_guess_1 = alpha_X.rvs()

        alpha_X_neg = get_truncated_normal(mean=0.205391265, sd=0.1766854623, low=0, upp=1)
        init_guess_2 = alpha_X_neg.rvs()

        beta_X = get_truncated_normal(mean=5.978394632, sd=3.21938602, low=1, upp=15)
        init_guess_3 = beta_X.rvs()'''

        X0 = [init_guess_1, init_guess_2, init_guess_3]


        res=minimize(fun=obFunc,x0=X0, method='L-BFGS-B',bounds=[(0.2,1),(0.2,1),(1,12)])

        LL=res.fun

        Xfit=res.x

        NegLL = -LL
        BIC = len(X0) * np.log(len(a)) + 2 * NegLL
        AIC = 2 * len(X0) + 2 * NegLL

        return Xfit, LL, BIC, AIC


"""simulation"""



"""fitting"""

#Xfit,LL, BIC, AIC=fit(a,r)

#print('Xfit')
#print(Xfit)
#print('LL')
#print(LL)


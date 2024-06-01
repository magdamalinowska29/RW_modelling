import pandas as pd
import random


from models.Rescorla_Wagner_Choice_Kernel_model import Rescorla_Wagner_CK_model

import matplotlib.pyplot as plt
import numpy as np

"""importing changepoint data"""

path_change='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_example_results_new.csv'

data_all_env=pd.read_csv(path_change, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

changepoints=data_all_env['Changepoint']
changepoints=changepoints.tolist()

first_changepoint=changepoints[0]

for change_n in range(len(changepoints)):
    changepoints[change_n] = changepoints[change_n] -first_changepoint
    #changepoints[change_n]=changepoints[change_n] + 1 # substracting the number of the first trial in the condition, so i can run the simulation with the data for each condition


new_pos=data_all_env['Positions']
new_pos=new_pos.tolist()

new_probs=data_all_env['Probabilities']
new_probs=new_probs.tolist()


"""setting up some simulation variables"""

count=300 #number of parameters combinations

alpha_sim=[]
beta_sim=[]

alpha_c_sim=[]
beta_c_sim=[]

acc_rew=np.zeros(count)

acc_rew_mid=np.zeros(count)
acc_rew_worst=np.zeros(count)

acc_rew_bad=np.zeros(count)

acc_rew_mid_bad=np.zeros(count)
acc_rew_worst_bad=np.zeros(count)

sim_n=100 # number of simulations for each combination

alpha=0.5
beta=8

for i in range(count):
    #alpha=random.uniform(0,1)
    alpha_c=random.uniform(0,1)
    #beta = random.uniform(0, 20)
    beta_c=random.uniform(0,20)
    """simulation"""

    param_acc=np.zeros(sim_n)
    param_acc_mid = np.zeros(sim_n)
    param_acc_worst=np.zeros(sim_n)

    param_acc_bad = np.zeros(sim_n)
    param_acc_mid_bad = np.zeros(sim_n)
    param_acc_worst_bad = np.zeros(sim_n)

    for sim in range(sim_n):

        RW_CK=Rescorla_Wagner_CK_model()

        a, r, best, mid, worst = RW_CK.simulate_change_accuracy(180, [0.2, 0.5, 0.8], alpha, beta, alpha_c, beta_c, changepoints, new_pos, new_probs)

        param_acc[sim] = sum(best) / len(r)

        param_acc_mid[sim] = sum(mid) / len(r)

        param_acc_worst[sim] = sum(worst) / len(r)

        if beta_c < 5:

            param_acc_bad[sim] = sum(best) / len(r)

            param_acc_mid_bad[sim] = sum(mid) / len(r)

            param_acc_worst_bad[sim] = sum(worst) / len(r)

        else:

            param_acc_bad[sim] = None

            param_acc_mid_bad[sim] = None

            param_acc_worst_bad[sim] = None

    alpha_sim.append(alpha)
    beta_sim.append(beta)

    alpha_c_sim.append(alpha_c)
    beta_c_sim.append(beta_c)

    acc_rew[i]=sum(param_acc)/sim_n
    acc_rew_mid[i] = sum(param_acc_mid) / sim_n
    acc_rew_worst[i] = sum(param_acc_worst) / sim_n

    acc_rew_bad[i] = sum(param_acc_bad) / sim_n
    acc_rew_mid_bad[i] = sum(param_acc_mid_bad) / sim_n
    acc_rew_worst_bad[i] = sum(param_acc_worst_bad) / sim_n



plt.scatter(alpha_sim,acc_rew, c="green")
plt.scatter(alpha_sim,acc_rew_bad, c="gray")
plt.title("best option")
plt.xlabel("learning rate")
plt.grid()
plt.show()

plt.scatter(beta_sim,acc_rew,c="green")
plt.scatter(beta_sim,acc_rew_bad,c="gray")
plt.title("best option")
plt.xlabel("temperature")
plt.grid()
plt.show()

plt.scatter(alpha_c_sim,acc_rew, c="green")
plt.scatter(alpha_c_sim,acc_rew_bad, c="gray")
plt.title("best option")
plt.xlabel("learning rate")
plt.grid()
plt.show()

plt.scatter(beta_c_sim,acc_rew,c="green")
plt.scatter(beta_c_sim,acc_rew_bad,c="gray")
plt.title("best option")
plt.xlabel("temperature")
plt.grid()
plt.show()

plt.scatter(alpha_sim,acc_rew_mid,c="yellow")
plt.scatter(alpha_sim,acc_rew_mid_bad,c="gray")
plt.title("2nd best option")
plt.xlabel("learning rate")
plt.grid()
plt.show()

plt.scatter(beta_sim,acc_rew_mid,c="yellow")
plt.scatter(beta_sim,acc_rew_mid_bad,c="gray")
plt.title("2nd best option")
plt.xlabel("temperature")
plt.grid()
plt.show()

plt.scatter(alpha_c_sim,acc_rew_mid,c="yellow")
plt.scatter(alpha_c_sim,acc_rew_mid_bad,c="gray")
plt.title("2nd best option")
plt.xlabel("learning rate")
plt.grid()
plt.show()

plt.scatter(beta_c_sim,acc_rew_mid,c="yellow")
plt.scatter(beta_c_sim,acc_rew_mid_bad,c="gray")
plt.title("2nd best option")
plt.xlabel("temperature")
plt.grid()
plt.show()

plt.scatter(alpha_sim,acc_rew_worst,c="red")
plt.scatter(alpha_sim,acc_rew_worst_bad,c="gray")
plt.title("worst option")
plt.xlabel("learning rate")
plt.grid()
plt.show()

plt.scatter(beta_sim,acc_rew_worst,c="red")
plt.scatter(beta_sim,acc_rew_worst_bad,c="gray")
plt.title("worst option")
plt.xlabel("temperature")
plt.grid()
plt.show()

plt.scatter(alpha_c_sim,acc_rew_worst,c="red")
plt.scatter(alpha_c_sim,acc_rew_worst_bad,c="gray")
plt.title("worst option")
plt.xlabel("learning rate")
plt.grid()
plt.show()

plt.scatter(beta_c_sim,acc_rew_worst,c="red")
plt.scatter(beta_c_sim,acc_rew_worst_bad,c="gray")
plt.title("worst option")
plt.xlabel("temperature")
plt.grid()
plt.show()





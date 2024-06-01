import pandas as pd
import random

from models.Rescorla_Wagner_model import Rescorla_Wagner_model

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

path_act='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_example_results_action_observation.csv'
path_change='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_example_results_new.csv'

data_all_trials=pd.read_csv(path_act)

actions=data_all_trials['Action']
actions=actions.tolist()

observations=data_all_trials['Observation']
observations=observations.tolist()
observ_2=np.zeros(len(observations))
for obs_ind in range(len(observations)):
    if observations[obs_ind]==2:
        observ_2[obs_ind]=0
observations=observ_2



stim_1_prob=data_all_trials['stim1_prob']
stim_1_prob=stim_1_prob.tolist()

stim_2_prob=data_all_trials['stim2_prob']
stim_2_prob=stim_2_prob.tolist()

stim_3_prob=data_all_trials['stim3_prob']
stim_3_prob=stim_3_prob.tolist()

data_all_env=pd.read_csv(path_change, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

#changepoints=data_all_env['Changepoint']
#changepoints=changepoints.tolist()

changepoints=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426, 432, 438, 444, 450, 456, 462, 468, 474, 480, 486, 492, 498]

#changepoints=[0, 5, 10, 15, 20, 25, 30, 35, 36, 40, 45, 50, 55, 60, 65, 70, 72, 75, 80, 85, 90, 95, 100, 105, 108, 110, 115, 120, 125, 130, 135, 140]
#changepoints=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 464, 468, 472, 476, 480, 484, 488, 492, 496]

#changepoints=[0, 7, 14, 21, 28, 35, 36, 42, 49, 56, 63, 70, 72, 77, 84, 91, 98, 105, 108, 112, 119, 126, 133, 140, 144, 147, 154, 161, 168, 175, 180, 182, 189, 196, 203, 210, 216, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 288, 294, 301, 308, 315, 322, 324, 329, 336, 343, 350, 357, 360, 364, 371, 378, 385, 392, 396, 399, 406, 413, 420, 427, 432, 434, 441, 448, 455, 462, 468, 469, 476, 483, 490, 497]



first_changepoint=changepoints[0]

for change_n in range(len(changepoints)):
    changepoints[change_n] = changepoints[change_n] -first_changepoint
    #changepoints[change_n]=changepoints[change_n] + 1 # substracting the number of the first trial in the condition, so i can run the simulation with the data for each condition


#new_pos=data_all_env['Positions']
#new_pos=new_pos.tolist()

#new_pos=[[0, 1, 2], [1], [2], [1], [1], [0], [0], [2], [0, 1, 2], [2], [1], [2], [1], [2], [2], [0], [0, 1, 2], [0], [0], [2], [0], [1], [0], [1], [0, 1, 2], [2], [1], [0], [1], [2], [0], [2]]
#new_pos=[[0, 1, 2], [2], [2], [1], [1], [2], [2], [2], [1], [0, 1, 2], [2], [2], [2], [1], [1], [1], [0], [0], [0, 1, 2], [2], [1], [1], [1], [0], [2], [2], [0], [0, 1, 2], [2], [0], [1], [1], [0], [2], [2], [1]]
#new_pos=[[1, 2, 3], [2], [1], [1], [2], [2], [2], [1], [1], [1, 2, 3], [2], [3], [3], [3], [1], [2], [2], [3], [1, 2, 3], [2], [1], [2], [3], [1], [3], [2], [2], [1, 2, 3], [2], [3], [1], [2], [1], [3], [2], [2], [1, 2, 3], [2], [3], [2], [1], [2], [2], [2], [3], [1, 2, 3], [1], [2], [3], [1], [3], [1], [3], [1], [1, 2, 3], [3], [2], [2], [3], [3], [3], [3], [2], [1, 2, 3], [3], [3], [1], [3], [3], [1], [3], [1], [1, 2, 3], [2], [1], [1], [3], [3], [1], [3], [1], [1, 2, 3], [3], [2], [1], [2], [3], [3], [2], [2], [1, 2, 3], [2], [3], [3], [3], [2], [2], [2], [3], [1, 2, 3], [1], [2], [3], [2], [2], [1], [2], [2], [1, 2, 3], [3], [2], [2], [3], [1], [1], [2], [1], [1, 2, 3], [1], [2], [1], [3], [2], [3], [3]]
new_pos=[[1, 2, 3], [3], [1], [2], [1], [2], [1, 2, 3], [2], [2], [2], [2], [3], [1, 2, 3], [1], [2], [1], [1], [3], [1, 2, 3], [3], [1], [1], [3], [2], [1, 2, 3], [2], [3], [2], [3], [3], [1, 2, 3], [3], [2], [3], [2], [2], [1, 2, 3], [1], [2], [1], [3], [2], [1, 2, 3], [2], [3], [1], [3], [2], [1, 2, 3], [3], [1], [2], [1], [3], [1, 2, 3], [3], [3], [1], [1], [2], [1, 2, 3], [2], [3], [1], [3], [3], [1, 2, 3], [1], [2], [3], [3], [2], [1, 2, 3], [1], [1], [1], [3], [1], [1, 2, 3], [3], [1], [2], [1], [2]]



#new_probs=data_all_env['Probabilities']
#new_probs=new_probs.tolist()


new_probs=[[0.1, 0.1, 0.5], [0.3], [0.9], [0.7], [0.9], [0.3], [0.7, 0.9, 0.5], [0.3], [0.1], [0.7], [0.3], [0.9], [0.7, 0.7, 0.3], [0.9], [0.3], [0.1], [0.1], [0.3], [0.3, 0.5, 0.7], [0.1], [0.9], [0.7], [0.1], [0.7], [0.7, 0.3, 0.3], [0.9], [0.1], [0.7], [0.1], [0.5], [0.3, 0.5, 0.5], [0.1], [0.7], [0.1], [0.1], [0.9], [0.3, 0.5, 0.9], [0.7], [0.1], [0.5], [0.3], [0.9], [0.5, 0.9, 0.9], [0.3], [0.7], [0.1], [0.5], [0.1], [0.9, 0.7, 0.9], [0.5], [0.1], [0.3], [0.7], [0.5], [0.5, 0.9, 0.7], [0.1], [0.3], [0.7], [0.5], [0.9], [0.5, 0.9, 0.1], [0.7], [0.9], [0.3], [0.7], [0.9], [0.9, 0.5, 0.7], [0.1], [0.5], [0.5], [0.3], [0.9], [0.3, 0.1, 0.1], [0.9], [0.5], [0.1], [0.7], [0.1], [0.5, 0.7, 0.7], [0.3], [0.9], [0.5], [0.7], [0.1]]

#new_probs=[[0.7, 0.7, 0.7], [0.1], [0.9], [0.3], [0.7], [0.1], [0.3], [0.1], [0.1, 0.5, 0.9], [0.9], [0.1], [0.7], [0.1], [0.1], [0.1], [0.7], [0.1, 0.9, 0.9], [0.9], [0.3], [0.5], [0.5], [0.5], [0.5], [0.7], [0.7, 0.3, 0.9], [0.3], [0.5], [0.9], [0.7], [0.3], [0.5], [0.1]]
#new_probs=[[0.5, 0.3, 0.9], [0.7], [0.7], [0.3], [0.5], [0.9], [0.9], [0.7], [0.1], [0.7, 0.5, 0.5], [0.7], [0.9], [0.9], [0.5], [0.7], [0.9], [0.9], [0.1], [0.5, 0.7, 0.9], [0.7], [0.7], [0.5], [0.1], [0.1], [0.7], [0.3], [0.3], [0.3, 0.3, 0.1], [0.3], [0.5], [0.7], [0.3], [0.1], [0.1], [0.1], [0.5]]
#new_probs=[[0.1, 0.5, 0.1], [0.3], [0.9], [0.7], [0.5], [0.7], [0.3], [0.9], [0.5], [0.9, 0.7, 0.3], [0.5], [0.7], [0.1], [0.7], [0.3], [0.1], [0.5], [0.9], [0.7, 0.3, 0.9], [0.1], [0.5], [0.7], [0.1], [0.9], [0.3], [0.5], [0.1], [0.7, 0.7, 0.9], [0.3], [0.1], [0.5], [0.7], [0.3], [0.5], [0.9], [0.1], [0.9, 0.7, 0.9], [0.5], [0.1], [0.7], [0.3], [0.9], [0.5], [0.9], [0.7], [0.1, 0.5, 0.3], [0.9], [0.7], [0.5], [0.3], [0.1], [0.9], [0.5], [0.1], [0.3, 0.1, 0.5], [0.9], [0.5], [0.1], [0.7], [0.9], [0.7], [0.5], [0.7], [0.9, 0.7, 0.9], [0.5], [0.1], [0.5], [0.9], [0.1], [0.3], [0.9], [0.5], [0.9, 0.5, 0.1], [0.3], [0.5], [0.7], [0.5], [0.9], [0.7], [0.5], [0.1], [0.3, 0.5, 0.9], [0.1], [0.9], [0.7], [0.3], [0.5], [0.1], [0.9], [0.3], [0.1, 0.3, 0.1], [0.7], [0.5], [0.3], [0.9], [0.3], [0.7], [0.5], [0.7], [0.1, 0.5, 0.1], [0.3], [0.3], [0.9], [0.7], [0.5], [0.1], [0.7], [0.3], [0.1, 0.9, 0.7], [0.1], [0.3], [0.9], [0.5], [0.7], [0.3], [0.1], [0.9], [0.9, 0.1, 0.3], [0.5], [0.7], [0.7], [0.1], [0.3], [0.5], [0.1]]

#new_probs=[[0.7, 0.1, 0.9], [0.3], [0.9], [0.7], [0.5], [0.9], [0.3, 0.7, 0.9], [0.5], [0.1], [0.7], [0.1], [0.5], [0.1, 0.3, 0.9], [0.7], [0.5], [0.1], [0.7], [0.9], [0.3, 0.3, 0.1], [0.9], [0.1], [0.7], [0.1], [0.5], [0.1, 0.5, 0.7], [0.1], [0.9], [0.7], [0.5], [0.3], [0.7, 0.7, 0.3], [0.9], [0.5], [0.1], [0.9], [0.5], [0.1, 0.3, 0.7], [0.5], [0.9], [0.1], [0.5], [0.7], [0.9, 0.3, 0.1], [0.5], [0.3], [0.7], [0.1], [0.9], [0.9, 0.7, 0.3], [0.1], [0.5], [0.5], [0.3], [0.7], [0.1, 0.7, 0.9], [0.3], [0.5], [0.9], [0.3], [0.9], [0.7, 0.3, 0.1], [0.9], [0.5], [0.1], [0.3], [0.5], [0.5, 0.5, 0.3], [0.9], [0.1], [0.5], [0.3], [0.7], [0.7, 0.3, 0.1], [0.9], [0.5], [0.9], [0.7], [0.1], [0.5, 0.3, 0.9], [0.7], [0.1], [0.9], [0.5], [0.3]]


"""change imported data to match the model"""

a=[]
r=[]
for act in range(len(actions)):

    a.append(actions[act]-1)

    if observations[act]==1:
        r.append(1)
    elif observations[act]==2:
        r.append(0)

count=6

alpha_sim=[]
beta_sim=[]
alpha_fit=[]
beta_fit=[]

alpha_sim_bad=[]
beta_sim_bad=[]

alpha_fit_bad=[]
beta_fit_bad=[]

for i in range(count):

    print(i)
    alpha=random.uniform(0.2,1)
    beta=random.uniform(2,12) #draw values from th distribution based on the subject fits
    #beta=np.random.exponential(15)

    #alpha = np.random.normal(loc=0.26051991, scale=0.223883188)
    #beta = np.random.normal(loc=13.30012029, scale=11.09702797)

    #alpha_X=get_truncated_normal(mean=0.3083119283, sd=0.1566426521, low=0.2, upp=1)
    #alpha=alpha_X.rvs()

    #beta_X=get_truncated_normal(mean=7.357686568, sd=4.175147737, low=2, upp=12)
    #beta=beta_X.rvs()

    #alpha_X=get_truncated_normal(mean=0.2727163683, sd=0.2497232234, low=0.1, upp=1)
    #alpha=alpha_X.rvs()

    #beta_X=get_truncated_normal(mean=14.49181034, sd=11.86140905, low=1, upp=100)
    #beta=beta_X.rvs()


    RW=Rescorla_Wagner_model()
    """simulation"""
    a,r=RW.simulate_change(180, [0.5,0.5,0.5], alpha, beta, changepoints, new_pos, new_probs)
    #a, r = RW.simulate(180, [0.5, 0.5, 0.5], alpha, beta)
    #a,r,new_pos_1=RW.simulate_change_test(144, [0.5,0.5,0.5],alpha,beta, changepoints, new_pos, new_probs)

    #a,r=RW.simulate(144, [0.5,0.5,0.5],alpha,beta)
    """fitting"""

    #Xfit, LL, BIC, AIC = RW.fit_change_test(a, r, changepoints, new_pos_1)
    Xfit,LL, BIC, AIC=RW.fit_change(a,r, changepoints,new_pos)
    #Xfit, LL, BIC, AIC = RW.fit(a, r)

    #Xfit,LL,BIC, AIC=RW.fit(a,r)

    if abs(alpha-Xfit[0])>0.25:
        alpha_sim_bad.append(alpha)
        alpha_fit_bad.append(Xfit[0])

        beta_sim_bad.append(beta)
        beta_fit_bad.append(Xfit[1])
   # else:

    #    alpha_sim.append(alpha)
     #   beta_sim.append(beta)
      #  alpha_fit.append(Xfit[0])
       # beta_fit.append(Xfit[1])
    alpha_sim.append(alpha)
    beta_sim.append(beta)
    alpha_fit.append(Xfit[0])
    beta_fit.append(Xfit[1])

#print(alpha_sim_bad)
#print(alpha_fit_bad)
#print(alpha_sim)
#print(alpha_fit)

print(np.corrcoef(alpha_sim, alpha_fit))
print(np.corrcoef(beta_sim, beta_fit))

results=pearsonr(alpha_sim,alpha_fit)
print(results.statistic)
print(results.pvalue)

results_2=pearsonr(beta_sim,beta_fit)
print(results_2.statistic)
print(results_2.pvalue)

plt.scatter(alpha_sim, alpha_fit, c='cyan')
#plt.show()
#plt.scatter(alpha_sim_bad,alpha_fit_bad,c='grey')
plt.grid()
plt.title('learning rate')
plt.show()
plt.scatter(beta_sim,beta_fit,c='magenta')
plt.title('temperature')
#plt.show()
#plt.scatter(beta_sim_bad,beta_fit_bad,c='grey')

plt.grid()
plt.show()

print('Xfit')
print(Xfit)
print('LL')
print(LL)

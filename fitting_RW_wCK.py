import pandas as pd

from models.Rescorla_Wagner_Choice_Kernel_model import Rescorla_Wagner_CK_model


path_act='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_i_example_results_action_observation.csv'
path_change='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_i_example_results_new.csv'

data_all_trials=pd.read_csv(path_act)

actions=data_all_trials['Action']
actions=actions.tolist()

observations=data_all_trials['Observation']
observations=observations.tolist()

stim_1_prob=data_all_trials['stim1_prob']
stim_1_prob=stim_1_prob.tolist()

stim_2_prob=data_all_trials['stim2_prob']
stim_2_prob=stim_2_prob.tolist()

stim_3_prob=data_all_trials['stim3_prob']
stim_3_prob=stim_3_prob.tolist()

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

"""change imported data to match the model's input"""

a=[]
r=[]
for act in range(len(actions)):

    a.append(actions[act]-1)

    if observations[act]==1:
        r.append(1)
    elif observations[act]==2:
        r.append(0)

RW_CK=Rescorla_Wagner_CK_model()


"""simulation"""
#a,r, best, mid, worst=simulate_change_accuracy(1000, [0.5,0.5,0.5],0.7,10,0.2,5, changepoints, new_pos, new_probs)

#a,r=simulate(144, [0.5,0.5,0.5],0.7,20)
"""fitting"""

Xfit,LL, BIC, AIC=RW_CK.fit_change(a,r, changepoints,new_pos)

#Xfit,LL, BIC, AIC=fit(a,r)

print('Xfit')
print(Xfit)
print('LL')
print(LL)
print('BIC')
print(BIC)
print('AIC')
print(AIC)


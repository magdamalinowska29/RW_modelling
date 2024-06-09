import pandas as pd

from models.Rescorla_Wagner_Choice_Kernel_model import Rescorla_Wagner_CK_model
import os
from os import listdir


directory = "C:/Users/magda/PycharmProjects/RW_modelling/data/data_action_pe_p/"  # Replace with your directory.

list_trial=listdir(directory)

paths_act=[]

for file_name in list_trial:
    filepath = os.path.join(directory, file_name)
    paths_act.append(filepath)

directory = "C:/Users/magda/PycharmProjects/RW_modelling/data/data_change_pe_p/"  # Replace with your directory.

list_trial=listdir(directory)

paths_change=[]

for file_name in list_trial:
    filepath = os.path.join(directory, file_name)
    paths_change.append(filepath)

lr_val=[]

lr_CK_val=[]

temp_val=[]

temp_CK_val=[]

LL_val=[]
BIC_val=[]
AIC_val=[]


for n in range(len(paths_act)):
    path_act = paths_act[n]
    path_change = paths_change[n]

    data_all_trials = pd.read_csv(path_act)




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

    """change imported data to match the model"""

    a=[]
    r=[]
    for act in range(len(actions)):

        a.append(actions[act]-1)

        if observations[act]==1:
            r.append(1)
        elif observations[act]==2:
            r.append(0)

    RW=Rescorla_Wagner_CK_model()



    """fitting"""

    Xfit,LL, BIC, AIC=RW.fit_change(a,r, changepoints,new_pos)

    lr_val.append(Xfit[0])
    lr_CK_val.append(Xfit[1])
    temp_val.append(Xfit[2])
    temp_CK_val.append(Xfit[3])

    LL_val.append(LL)
    BIC_val.append(BIC)
    AIC_val.append(AIC)

    print('Xfit')
    print(Xfit)
    print('LL')
    print(LL)
    print('BIC')
    print(BIC)
    print('AIC')
    print(AIC)

dict = {'lr': lr_val, 'lr_CK':lr_CK_val,  'temp': temp_val, 'temp_CK': temp_CK_val, 'LL': LL_val, 'BIC': BIC_val, 'AIC': AIC_val}


df = pd.DataFrame(dict)

df.to_csv('data/fitting_RW_CK_results_pe_i.csv')


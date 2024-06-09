import pandas as pd

from models.Rescorla_Wagner_sep_lr_model import Rescorla_Wagner_sep_lr_model
from os import listdir
import os


directory = "C:/Users/magda/PycharmProjects/RW_modelling/data/data_action_r_p/"  # Replace with your directory.

list_trial=listdir(directory)

paths_act=[]

for file_name in list_trial:
    filepath = os.path.join(directory, file_name)
    paths_act.append(filepath)

directory = "C:/Users/magda/PycharmProjects/RW_modelling/data/data_change_r_p/"  # Replace with your directory.

list_trial=listdir(directory)

paths_change=[]

for file_name in list_trial:
    filepath = os.path.join(directory, file_name)
    paths_change.append(filepath)


lr_val=[]
lr_neg_val=[]
temp_val=[]
LL_val=[]
BIC_val=[]
AIC_val=[]

for n in range(len(paths_act)):
    path_act=paths_act[n]

    path_change=paths_change[n]

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

    action_comp=data_all_trials['Computer_Action']
    action_comp=action_comp.tolist()

    obs_comp=data_all_trials['Computer_Observations']
    obs_comp=obs_comp.tolist()


    """change imported data to match the model"""

    a=[]
    a_comp=[]
    r=[]
    for act in range(len(actions)):

        a.append(actions[act]-1)
        a_comp.append(action_comp[act] - 1)

        if obs_comp[act]==1:
            r.append(1)
        elif obs_comp[act]==2:
            r.append(0)

    RW_sep_lr=Rescorla_Wagner_sep_lr_model()


    """fitting"""

    Xfit,LL, BIC, AIC=RW_sep_lr.fit_change_pav(a, a_comp, r, changepoints,new_pos)



    lr_val.append(Xfit[0])
    lr_neg_val.append(Xfit[1])
    temp_val.append(Xfit[1])
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

dict = {'lr': lr_val, 'lr_neg': lr_neg_val, 'temp': temp_val, 'LL': LL_val, 'BIC': BIC_val, 'AIC': AIC_val}


df = pd.DataFrame(dict)

df.to_csv('data/fitting_RW_sep_lr_results_r_p_pav.csv')



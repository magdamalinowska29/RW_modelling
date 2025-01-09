'''this is a script that transforms behavioural data recorded in the fMRI experiment to the format needed for analysis'''

import pandas as pd
import numpy as np

"""upload the result file as a pandas array"""
csvFile = pd.read_csv('C:/Users/magda/Downloads/RewPerc_behavioral_data/sub001_02046775_fMRItask_V2021_2_3_3_new_2024_Apr_02_1055.csv')
#print(csvFile)

"""extract the columns that conatain the stimuli paths"""
stimOne=csvFile['stim1']
stimTwo=csvFile['stim2']
stimThree=csvFile['stim3']

#drop the NaN rows
stimOne=stimOne.dropna(how='any')
stimTwo=stimTwo.dropna(how='any')
stimThree=stimThree.dropna(how='any')


#convert to a list
stimOne_list=stimOne.tolist()
stimTwo_list=stimTwo.tolist()
stimThree_list=stimThree.tolist()

print(len(stimOne_list))

"""extract the column that informas about the condition"""

Condition_list=csvFile['current_condition']

Condition_list=Condition_list.dropna()
Condition_list=Condition_list.tolist()

print(len(Condition_list))

rew_inst_trials = []
perc_inst_trials=[]


changepoint_list_r_i=[]
changepoint_list_pe_i=[]


new_stim_pos_r_i=[]
new_stim_pos_pe_i=[]


new_stim_probs_r_i=[]
new_stim_probs_pe_i=[]

'''extract the column that infomrs about the start of the block'''
block_info=csvFile['space_for_next.keys']

block_info=block_info.notnull() #convert to boolean


block_info=block_info.drop(block_info.index[0]) #getting rid of the first row, bc it doesnt contain trial info


block_info=block_info.tolist() #converting into a list

#collecting the trial numbers of trials that are at the start of the block

block_start=[0]
for trl_n in range(len(block_info)):

    if block_info[trl_n]:

        block_start.append(trl_n)


"""extract the columns that contain the stimuli probabilities"""

prob_stimOne=csvFile['stim1prob']
prob_stimTwo=csvFile['stim2prob']
prob_stimThree=csvFile['stim3prob']

prob_stimOne=prob_stimOne.dropna()
prob_stimTwo=prob_stimTwo.dropna()
prob_stimThree=prob_stimThree.dropna()

prob_stimOne_list=prob_stimOne.tolist()
prob_stimTwo_list=prob_stimTwo.tolist()
prob_stimThree_list=prob_stimThree.tolist()



"""extract the column that contains persons responses"""

responses=csvFile['key_response.keys']

responses=responses.dropna()

responses=responses.tolist()

#print(responses)

print(responses)
print(len(responses))

subj_action=np.zeros(len(stimOne_list))

"""collecting the observations"""

subj_obs_column=csvFile['correct_answer'] #get the column that contains subjects observations


subj_obs_column=subj_obs_column.dropna()


subj_obs_column=subj_obs_column.tolist()


obs_dict={1:1,0:2, -1:-1}

subj_obs=np.zeros(len(subj_obs_column))

for obs_n in range(len(subj_obs_column)):

    subj_obs[obs_n]=obs_dict[subj_obs_column[obs_n]]

stim1_prob_list=[]
stim2_prob_list=[]
stim3_prob_list=[] #lists to hold the information about the probabilities of each stimulus


'''create a dictionary to hold the current images'''


stim_dict={1:stimOne_list[0],2:stimTwo_list[0],3:stimThree_list[0]}  #note, I'm using these key values, cause they should correspond to the positions in the modelling scripts
#print(stim_dict)

changepoint_list=[] #empty list to fill with the numbers of trials that contain a changepoint
new_stim_probs=[]

new_stim_positions=[]

#loop through the trials
for trial_num in range(len(stimOne_list)):
    #print(trial_num)
    stim_1=stimOne_list[trial_num]
    stim_2 = stimTwo_list[trial_num]
    stim_3 = stimThree_list[trial_num]

    prob_stim1=prob_stimOne_list[trial_num]
    prob_stim2=prob_stimTwo_list[trial_num]
    prob_stim3=prob_stimThree_list[trial_num]

    stim_list=[stim_1,stim_2,stim_3]
    prob_list=[prob_stim1, prob_stim2,prob_stim3]

    new_pos=[] #a list of positions where new stimulus is introduced
    new_probs=[] #a list to hold new probabilities

    """check condition"""

    if Condition_list[trial_num]==0:
        rew_inst_trials.append(trial_num)

    elif Condition_list[trial_num]==1:
        perc_inst_trials.append(trial_num)


    if trial_num in block_start: #check if the current trial is the start of a new block



        #if so, then change the stimulus dictionary to the stimuli from the current trial
        stim_dict = {1: stimOne_list[trial_num], 2: stimTwo_list[trial_num], 3: stimThree_list[trial_num]}
        prob_dict = {1: prob_stimOne_list[trial_num], 2: prob_stimTwo_list[trial_num], 3: prob_stimThree_list[trial_num]}

        new_pos=list(stim_dict.keys()) #all positions in the dictionary get replaced
        new_probs=list(prob_dict.values())

        changepoint_list.append(trial_num) #add the trial to the changepoint list, cause it won't be added later



        new_stim_positions.append(new_pos) #add a list of positions on which stim is replaced
        new_stim_probs.append(new_probs)

        #"""record the probabilities of each option"""

        #stim1_prob_list.append(prob_dict[1])
        #stim2_prob_list.append(prob_dict[2])
        #stim3_prob_list.append(prob_dict[3])


        if trial_num in rew_inst_trials:
            changepoint_list_r_i.append(trial_num)
            new_stim_pos_r_i.append(new_pos)
            new_stim_probs_r_i.append(new_probs)

        elif trial_num in perc_inst_trials:
            changepoint_list_pe_i.append(trial_num)
            new_stim_pos_pe_i.append(new_pos)
            new_stim_probs_pe_i.append(new_probs)


    #create a list of the changepoints
    for id_el in range(len(stim_list)): #loop over the stimuli in the current trial
        #print(el)
        #print(type(el))
        el=stim_list[id_el]
        if el not in stim_dict.values():
            #print('no_change')
            changepoint_list.append(trial_num)
         #   print('change')
            new_stim=el # the identity of new stimulus
            new_prob=prob_list[id_el]

            for val in stim_dict.values(): #loop through the previous stimuli names

                if val not in stim_list: #check if the stimulus is still used

                    key_list = list(stim_dict.keys())
                    val_list = list(stim_dict.values())

                    # print key with our value
                    position = val_list.index(val)
                    key=key_list[position]


                    stim_dict[key]=new_stim
                    prob_dict[key]=new_prob

                    new_pos.append(key)
                    new_probs.append(new_prob)

                    #print(stim_dict)
            new_stim_positions.append(new_pos)
            new_stim_probs.append(new_probs)



            if trial_num in rew_inst_trials:
                changepoint_list_r_i.append(trial_num)
                new_stim_pos_r_i.append(new_pos)
                new_stim_probs_r_i.append(new_probs)


            else:
                changepoint_list_pe_i.append(trial_num)
                new_stim_pos_pe_i.append(new_pos)
                new_stim_probs_pe_i.append(new_probs)


    '''subject response'''
    key_response = responses[trial_num]  # response given in the current trial


    pos_dict = {1: stimOne_list[trial_num], 2: stimTwo_list[trial_num],
                3: stimThree_list[trial_num]}  # this dictionary maps persons response to the aliens identity

    alien_name = pos_dict[key_response]

    key_list_action = list(stim_dict.keys())
    val_list_action = list(stim_dict.values())

    print(trial_num)
    print(val_list_action)
    print(block_start)
    # print key with our value
    position_action = val_list_action.index(alien_name)
    trial_action = key_list_action[position_action]

    subj_action[trial_num] = int(trial_action)


    """record the probabilities of each option"""

    stim1_prob_list.append(prob_dict[1])
    stim2_prob_list.append(prob_dict[2])
    stim3_prob_list.append(prob_dict[3])

"""dividing the data into conditions"""

stim1_prob_r_i=[]
stim1_prob_pe_i=[]

stim2_prob_r_i=[]
stim2_prob_pe_i=[]

stim3_prob_r_i=[]
stim3_prob_pe_i=[]


subj_action_r_i=[]
subj_action_pe_i=[]

comp_action_r_i=[]
comp_action_pe_i=[]

subj_obs_r_i=[]
subj_obs_pe_i=[]

comp_obs_r_i=[]
comp_obs_pe_i=[]

for t_n in range(len(Condition_list)):

    print(t_n)
    if t_n in rew_inst_trials:
        #changepoint_list_r_i.append(changepoint_list[t_n])
        #new_stim_pos_r_i.append(new_stim_positions[t_n])
        subj_action_r_i.append(subj_action[t_n])
        subj_obs_r_i.append(subj_obs[t_n])

        stim1_prob_r_i.append(stim1_prob_list[t_n])
        stim2_prob_r_i.append(stim2_prob_list[t_n])
        stim3_prob_r_i.append(stim3_prob_list[t_n])


    else :
        #changepoint_list_pe_i.append(changepoint_list[t_n])
        #new_stim_pos_pe_i.append(new_stim_positions[t_n])
        subj_action_pe_i.append(subj_action[t_n])

        subj_obs_pe_i.append(subj_obs[t_n])
        stim1_prob_pe_i.append(stim1_prob_list[t_n])
        stim2_prob_pe_i.append(stim2_prob_list[t_n])
        stim3_prob_pe_i.append(stim3_prob_list[t_n])

"""saving the results in csv file"""

dict = {'Changepoint': changepoint_list, 'Positions': new_stim_positions, 'Probabilities': new_stim_probs}

df = pd.DataFrame(dict)

# saving the dataframe
df.to_csv('data/sub004_example_results_new.csv')

dict_2={'Action':subj_action, 'Observation': subj_obs, "stim1_prob": stim1_prob_list,"stim2_prob":stim2_prob_list,"stim3_prob": stim3_prob_list}

df_2=pd.DataFrame(dict_2)
df_2.to_csv('data/sub004_example_results_action_observation.csv')

"""saving the results in csv files based on condition"""

dict_r_i = {'Changepoint': changepoint_list_r_i, 'Positions': new_stim_pos_r_i, 'Probabilities': new_stim_probs_r_i}

df_r_i = pd.DataFrame(dict_r_i)

# saving the dataframe
df_r_i.to_csv('data/sub004_r_i_example_results_new.csv')

dict_2_r_i={'Action':subj_action_r_i, 'Observation': subj_obs_r_i,'stim1_prob': stim1_prob_r_i, 'stim2_prob':stim2_prob_r_i,'stim3_prob': stim3_prob_r_i}

df_2_r_i=pd.DataFrame(dict_2_r_i)
df_2_r_i.to_csv('data/sub004_r_i_example_results_action_observation.csv')

"""saving the results in csv file"""

dict_pe_i = {'Changepoint': changepoint_list_pe_i, 'Positions': new_stim_pos_pe_i,'Probabilities': new_stim_probs_pe_i}

df_pe_i = pd.DataFrame(dict_pe_i)

# saving the dataframe
df_pe_i.to_csv('data/sub001_pe_i_example_results_new.csv')

dict_2_pe_i={'Action':subj_action_pe_i, 'Observation': subj_obs_pe_i,'stim1_prob': stim1_prob_pe_i, 'stim2_prob':stim2_prob_pe_i,'stim3_prob': stim3_prob_pe_i}

df_2_pe_i=pd.DataFrame(dict_2_pe_i)
df_2_pe_i.to_csv('data/sub001_pe_i_example_results_action_observation.csv')










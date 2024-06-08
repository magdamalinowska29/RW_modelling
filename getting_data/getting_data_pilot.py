import pandas as pd
import numpy as np

file_list=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-30_14h49.52.558.csv',
        'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-06-01_19h28.15.309.csv']
    #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-30_12h12.34.721.csv']
    #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-28_19h52.05.108.csv']
    #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-28_14h39.37.033.csv']
    #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-29_20h19.05.288.csv',
            #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-29_17h48.57.222.csv']
            #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-28_14h03.07.799.csv']
           # 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-28_14h39.37.033.csv']
            #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-25_11h42.30.286.csv',
           #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-27_17h53.02.337.csv',
           #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-25_12h42.39.315.csv',
           #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-25_18h22.35.977.csv',
           #'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/undefined_test_2_2024-05-26_10h41.19.362.csv']

for file_n in range(len(file_list)):

    """upload the result file as a pandas array"""
    csvFile = pd.read_csv(file_list[file_n])
    #print(csvFile)

    '''get the sona-id'''

    sona_id_col=csvFile['survey.sonaID']
    sona_id_col=sona_id_col.tolist()

    sona_id=int(sona_id_col[1])
    #sona_id=1111
    """find the tutorial trials"""

    tut_info=csvFile['tutorial_trial_loop.ran']

    tut_info=tut_info.notnull() #convert to boolean

    tut_info=tut_info.tolist()

    len_tut_inf=len(tut_info)
    tut_info=tut_info[28:len_tut_inf]

    print(tut_info)

    tut_trial=[]
    for trl in range(len(tut_info)):

        if tut_info[trl]:

            tut_trial.append(trl)

    print(tut_trial)




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

    stimOne_list_clean=[]
    stimTwo_list_clean=[]
    stimThree_list_clean=[]


    for stim_ind in range(len(stimOne_list)):

        if stim_ind not in tut_trial:
            stimOne_list_clean.append(stimOne_list[stim_ind])
            stimTwo_list_clean.append(stimTwo_list[stim_ind])
            stimThree_list_clean.append(stimThree_list[stim_ind])

    stimOne_list=stimOne_list_clean
    stimTwo_list=stimTwo_list_clean
    stimThree_list=stimThree_list_clean




    """extract the column that informas about the condition"""

    Condition_list=csvFile['current_condition']

    Condition_list=Condition_list.dropna()
    Condition_list=Condition_list.tolist()

    #print(Condition_list)

    rew_inst_trials = []
    rew_pav_trials=[]
    perc_inst_trials=[]
    perc_pav_trials=[]

    changepoint_list_r_i=[]
    changepoint_list_r_p=[]
    changepoint_list_pe_i=[]
    changepoint_list_pe_p=[]

    new_stim_pos_r_i=[]
    new_stim_pos_r_p=[]
    new_stim_pos_pe_i=[]
    new_stim_pos_pe_p=[]

    new_stim_probs_r_i=[]
    new_stim_probs_r_p=[]
    new_stim_probs_pe_i=[]
    new_stim_probs_pe_p=[]

    '''extract the column that infomrs about the start of the block'''
    block_info=csvFile['space_for_next.keys']

    block_info=block_info.notnull() #convert to boolean


    block_info=block_info.drop(block_info.index[0]) #getting rid of the first row, bc it doesnt contain trial info


    block_info=block_info.tolist() #converting into a list

    lent_inf=len(block_info)
    block_info=block_info[31:lent_inf] #dropping first 31 rows, cause they don't contain trial info

    #print(block_info)

    #collecting the trial numbers of trials that are at the start of the block

    block_start=[]
    for trl_n in range(len(block_info)):

        if block_info[trl_n]:

            block_start.append(trl_n)

    block_start=[0,36,72,108,144,180,216,252,288,324,360,396,432,468,504,540, 576, 612, 648, 684, 720] #TODO: figure out how to get it automaticlly from the data

    print(block_start)

    """extract the columns that contain the stimuli probabilities"""

    prob_stimOne=csvFile['stim1prob']
    prob_stimTwo=csvFile['stim2prob']
    prob_stimThree=csvFile['stim3prob']

    #drop nans
    prob_stimOne=prob_stimOne.dropna()
    prob_stimTwo=prob_stimTwo.dropna()
    prob_stimThree=prob_stimThree.dropna()

    prob_stimOne_list=prob_stimOne.tolist()
    prob_stimTwo_list=prob_stimTwo.tolist()
    prob_stimThree_list=prob_stimThree.tolist()

    #print(prob_stimTwo_list)
    #print(prob_stimOne_list)
    #print(prob_stimThree_list)

    """extract the column that contains persons responses"""

    responses=csvFile['key_response.keys']

    responses=responses.dropna()

    responses=responses.tolist()

    #print(responses)

    subj_action=np.zeros(len(stimOne_list))

    """extract the column that contains computer responses"""

    resp_comp=csvFile['machineAns']


    resp_comp=resp_comp.dropna()
    resp_comp=resp_comp.tolist()

    resp_comp_clean=[]


    for resp_c_ind in range(len(resp_comp)):

        if resp_c_ind not in tut_trial:
            resp_comp_clean.append(resp_comp[resp_c_ind])

    resp_comp=resp_comp_clean



    comp_action=np.zeros(len(stimOne_list))

    """collecting the observations"""

    subj_obs_column=csvFile['correct_answer'] #get the column that contains subjects observations
    comp_obs_column=csvFile['machine_correct_answer'] #same for comupter answers

    #subj_obs_column=subj_obs_column.drop(subj_obs_column.index[0])
    #comp_obs_column=comp_obs_column.drop(comp_obs_column.index[0])

    subj_obs_column=subj_obs_column.dropna()
    comp_obs_column=comp_obs_column.dropna()

    subj_obs_column=subj_obs_column.tolist()
    comp_obs_column=comp_obs_column.tolist()

    obs_dict={1:1,0:2}
    subj_obs=np.zeros(len(subj_obs_column))
    comp_obs=np.zeros(len(comp_obs_column))

    for obs_n in range(len(subj_obs_column)):

        subj_obs[obs_n]=obs_dict[subj_obs_column[obs_n]]
        comp_obs[obs_n] = obs_dict[comp_obs_column[obs_n]]


    stim1_prob_list=[]
    stim2_prob_list=[]
    stim3_prob_list=[] #lists to hold the information about the probabilities of each stimulus

    print(subj_obs)
    print(len(subj_obs))
    print(comp_obs)
    print(len(comp_obs))

    '''create a dictionary to hold the current images'''


    stim_dict={1:stimOne_list[0],2:stimTwo_list[0],3:stimThree_list[0]}  #note, I'm using these key values, cause they should correspond to the positions in the modelling scripts
    #print(stim_dict)

    changepoint_list=[] #empty list to fill with the numbers of trials that contain a changepoint
    new_stim_probs=[]

    new_stim_positions=[] #a list to fill with positions in which the stimulus is replaced (these are not positions on the screen, but symbolic positions that are used in modelling

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
            rew_pav_trials.append(trial_num)

        elif Condition_list[trial_num]==2:
            perc_inst_trials.append(trial_num)

        elif Condition_list[trial_num]==3:
            perc_pav_trials.append(trial_num)

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

            elif trial_num in rew_pav_trials:
                changepoint_list_r_p.append(trial_num)
                new_stim_pos_r_p.append(new_pos)
                new_stim_probs_r_p.append(new_probs)

            elif trial_num in perc_inst_trials:
                changepoint_list_pe_i.append(trial_num)
                new_stim_pos_pe_i.append(new_pos)
                new_stim_probs_pe_i.append(new_probs)

            elif trial_num in perc_pav_trials:
                changepoint_list_pe_p.append(trial_num)
                new_stim_pos_pe_p.append(new_pos)
                new_stim_probs_pe_p.append((new_probs))

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
              #      print('these are the values in the dictionary: ')
                    #print(stim_dict.values())
                    if val not in stim_list: #check if the stimulus is still used
                        #print('stays')
               #         print('insert new')
                        #print('this is the old dictionary:')
                        #print(stim_dict)
                        key_list = list(stim_dict.keys())
                        val_list = list(stim_dict.values())

                        # print key with our value
                        position = val_list.index(val)
                        key=key_list[position]
                #        print('this is the key: '+str(key))
                        #print('this is the new dictionary: ')


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

                elif trial_num in rew_pav_trials:
                    changepoint_list_r_p.append(trial_num)
                    new_stim_pos_r_p.append(new_pos)
                    new_stim_probs_r_p.append(new_probs)

                elif trial_num in perc_inst_trials:
                    changepoint_list_pe_i.append(trial_num)
                    new_stim_pos_pe_i.append(new_pos)
                    new_stim_probs_pe_i.append(new_probs)

                elif trial_num in perc_pav_trials:
                    changepoint_list_pe_p.append(trial_num)
                    new_stim_pos_pe_p.append(new_pos)
                    new_stim_probs_pe_p.append(new_probs)

        '''subject response'''
        key_response = responses[trial_num]  # response given in the current trial

        pos_dict = {'left': stimOne_list[trial_num], 'down': stimTwo_list[trial_num],
                    'right': stimThree_list[trial_num]}  # this dictionary maps persons response to the aliens identity

        alien_name = pos_dict[key_response]

        key_list_action = list(stim_dict.keys())
        val_list_action = list(stim_dict.values())

        # print key with our value
        position_action = val_list_action.index(alien_name)
        trial_action = key_list_action[position_action]

        subj_action[trial_num] = int(trial_action)

        '''computer response'''
        key_resp_comp = resp_comp[trial_num]  # response given in the current trial

        pos_dict_comp = {'stimLeft': stimOne_list[trial_num], 'stimCenter': stimTwo_list[trial_num],
                         'stimRight': stimThree_list[
                             trial_num]}  # this dictionary maps persons response to the aliens identity

        alien_name_comp = pos_dict_comp[key_resp_comp]

        key_list_action_comp = list(stim_dict.keys())
        val_list_action_comp = list(stim_dict.values())

        # print key with our value
        position_action_comp = val_list_action_comp.index(alien_name_comp)
        trial_action_comp = key_list_action_comp[position_action_comp]

        comp_action[trial_num] = int(trial_action_comp)

        """record the probabilities of each option"""

        stim1_prob_list.append(prob_dict[1])
        stim2_prob_list.append(prob_dict[2])
        stim3_prob_list.append(prob_dict[3])

    """dividing the data into conditions"""

    stim1_prob_r_i=[]
    stim1_prob_r_p=[]
    stim1_prob_pe_i=[]
    stim1_prob_pe_p=[]

    stim2_prob_r_i=[]
    stim2_prob_r_p=[]
    stim2_prob_pe_i=[]
    stim2_prob_pe_p=[]

    stim3_prob_r_i=[]
    stim3_prob_r_p=[]
    stim3_prob_pe_i=[]
    stim3_prob_pe_p=[]



    subj_action_r_i=[]
    subj_action_r_p=[]
    subj_action_pe_i=[]
    subj_action_pe_p=[]

    comp_action_r_i=[]
    comp_action_r_p=[]
    comp_action_pe_i=[]
    comp_action_pe_p=[]

    subj_obs_r_i=[]
    subj_obs_r_p=[]
    subj_obs_pe_i=[]
    subj_obs_pe_p=[]

    comp_obs_r_i=[]
    comp_obs_r_p=[]
    comp_obs_pe_i=[]
    comp_obs_pe_p=[]

    for t_n in range(len(Condition_list)):

        print(t_n)
        if t_n in rew_inst_trials:
            #changepoint_list_r_i.append(changepoint_list[t_n])
            #new_stim_pos_r_i.append(new_stim_positions[t_n])
            subj_action_r_i.append(subj_action[t_n])
            comp_action_r_i.append(comp_action[t_n])
            subj_obs_r_i.append(subj_obs[t_n])
            comp_obs_r_i.append(comp_obs[t_n])
            stim1_prob_r_i.append(stim1_prob_list[t_n])
            stim2_prob_r_i.append(stim2_prob_list[t_n])
            stim3_prob_r_i.append(stim3_prob_list[t_n])

        elif t_n in rew_pav_trials:
            #changepoint_list_r_p.append(changepoint_list[t_n])
            #new_stim_pos_r_p.append(new_stim_positions[t_n])
            subj_action_r_p.append(subj_action[t_n])
            comp_action_r_p.append(comp_action[t_n])
            subj_obs_r_p.append(subj_obs[t_n])
            comp_obs_r_p.append(comp_obs[t_n])
            stim1_prob_r_p.append(stim1_prob_list[t_n])
            stim2_prob_r_p.append(stim2_prob_list[t_n])
            stim3_prob_r_p.append(stim3_prob_list[t_n])

        elif t_n in perc_inst_trials:
            #changepoint_list_pe_i.append(changepoint_list[t_n])
            #new_stim_pos_pe_i.append(new_stim_positions[t_n])
            subj_action_pe_i.append(subj_action[t_n])
            comp_action_pe_i.append(comp_action[t_n])
            subj_obs_pe_i.append(subj_obs[t_n])
            comp_obs_pe_i.append(comp_obs[t_n])
            stim1_prob_pe_i.append(stim1_prob_list[t_n])
            stim2_prob_pe_i.append(stim2_prob_list[t_n])
            stim3_prob_pe_i.append(stim3_prob_list[t_n])

        elif t_n in perc_pav_trials:
            #changepoint_list_pe_p.append(changepoint_list[t_n])
            #new_stim_pos_pe_p.append(new_stim_positions[t_n])
            subj_action_pe_p.append(subj_action[t_n])
            comp_action_pe_p.append(comp_action[t_n])
            subj_obs_pe_p.append(subj_obs[t_n])
            comp_obs_pe_p.append(comp_obs[t_n])
            stim1_prob_pe_p.append(stim1_prob_list[t_n])
            stim2_prob_pe_p.append(stim2_prob_list[t_n])
            stim3_prob_pe_p.append(stim3_prob_list[t_n])



    """saving the results in csv file"""

    dict = {'Changepoint': changepoint_list, 'Positions': new_stim_positions, 'Probabilities': new_stim_probs}

    df = pd.DataFrame(dict)

    # saving the dataframe
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id)+ '_example_results_new.csv'

    df.to_csv(nam)

    dict_2={'Action':subj_action,'Computer_Action': comp_action, 'Observation': subj_obs,'Computer_Observations': comp_obs, "stim1_prob": stim1_prob_list,"stim2_prob":stim2_prob_list,"stim3_prob": stim3_prob_list}

    df_2=pd.DataFrame(dict_2)
    nam_2='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) +  '_example_results_action_observation.csv'
    df_2.to_csv(nam_2)

    """saving the results in csv files based on condition"""

    dict_r_i = {'Changepoint': changepoint_list_r_i, 'Positions': new_stim_pos_r_i, 'Probabilities': new_stim_probs_r_i}

    df_r_i = pd.DataFrame(dict_r_i)

    # saving the dataframe

    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_r_i_example_results_new.csv'
    df_r_i.to_csv(nam)

    dict_2_r_i={'Action':subj_action_r_i,'Computer_Action': comp_action_r_i, 'Observation': subj_obs_r_i,'Computer_Observations': comp_obs_r_i,'stim1_prob': stim1_prob_r_i, 'stim2_prob':stim2_prob_r_i,'stim3_prob': stim3_prob_r_i}

    df_2_r_i=pd.DataFrame(dict_2_r_i)
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_r_i_example_results_action_observation.csv'
    df_2_r_i.to_csv(nam)

    """saving the results in csv file"""

    dict_r_p = {'Changepoint': changepoint_list_r_p, 'Positions': new_stim_pos_r_p, 'Probabilities': new_stim_probs_r_p}

    df_r_p = pd.DataFrame(dict_r_p)

    # saving the dataframe
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_r_p_example_results_new.csv'
    df_r_p.to_csv(nam)

    dict_2_r_p={'Action':subj_action_r_p,'Computer_Action': comp_action_r_p, 'Observation': subj_obs_r_p,'Computer_Observations': comp_obs_r_p,'stim1_prob': stim1_prob_r_p, 'stim2_prob':stim2_prob_r_p,'stim3_prob': stim3_prob_r_p}

    df_2_r_p=pd.DataFrame(dict_2_r_p)
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_r_p_example_results_action_observation.csv'
    df_2_r_p.to_csv(nam)

    """saving the results in csv file"""

    dict_pe_i = {'Changepoint': changepoint_list_pe_i, 'Positions': new_stim_pos_pe_i,'Probabilities': new_stim_probs_pe_i}

    df_pe_i = pd.DataFrame(dict_pe_i)

    # saving the dataframe
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_pe_i_example_results_new.csv'
    df_pe_i.to_csv(nam)

    dict_2_pe_i={'Action':subj_action_pe_i,'Computer_Action': comp_action_pe_i, 'Observation': subj_obs_pe_i,'Computer_Observations': comp_obs_pe_i,'stim1_prob': stim1_prob_pe_i, 'stim2_prob':stim2_prob_pe_i,'stim3_prob': stim3_prob_pe_i}

    df_2_pe_i=pd.DataFrame(dict_2_pe_i)
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_pe_i_example_results_action_observation.csv'
    df_2_pe_i.to_csv(nam)

    """saving the results in csv file"""

    dict_pe_p = {'Changepoint': changepoint_list_pe_p, 'Positions': new_stim_pos_pe_p, 'Probabilities': new_stim_probs_pe_p}

    df_pe_p = pd.DataFrame(dict_pe_p)

    # saving the dataframe
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_pe_p_example_results_new.csv'
    df_pe_p.to_csv(nam)

    dict_2_pe_p={'Action':subj_action_pe_p,'Computer_Action': comp_action_pe_p, 'Observation': subj_obs_pe_p,'Computer_Observations': comp_obs_pe_p,'stim1_prob': stim1_prob_pe_p, 'stim2_prob':stim2_prob_pe_p,'stim3_prob': stim3_prob_pe_p}

    df_2_pe_p=pd.DataFrame(dict_2_pe_p)
    nam='C:/Users/magda/PycharmProjects/Rascola_Wagner/data/' + str(sona_id) + '_pe_p_example_results_action_observation.csv'
    df_2_pe_p.to_csv(nam)














import pandas as pd
import random


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr



path_act= 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_p_example_results_action_observation.csv'
path_change= 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_p_example_results_new.csv'

data_all_trials=pd.read_csv(path_act)

data_all_env=pd.read_csv(path_change, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings



stim_1_prob=data_all_trials['stim1_prob']
stim_1_prob=stim_1_prob.tolist()

stim_2_prob=data_all_trials['stim2_prob']
stim_2_prob=stim_2_prob.tolist()

stim_3_prob=data_all_trials['stim3_prob']
stim_3_prob=stim_3_prob.tolist()

comp_actions=data_all_trials['Computer_Action']
comp_actions=comp_actions.tolist()


comp_obs=data_all_trials['Computer_Observations']
comp_obs=comp_obs.tolist()

path_act_i= 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_i_example_results_action_observation.csv'
path_change_i= 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_i_example_results_new.csv'

data_all_trials_i=pd.read_csv(path_act_i)

data_all_env_i=pd.read_csv(path_change_i, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

actions=data_all_trials_i['Action']
#actions=data_all_trials_i['Computer_Action']
actions=actions.tolist()



stim_1_prob_i=data_all_trials_i['stim1_prob']
stim_1_prob_i=stim_1_prob_i.tolist()

stim_2_prob_i=data_all_trials_i['stim2_prob']
stim_2_prob_i=stim_2_prob_i.tolist()

stim_3_prob_i=data_all_trials_i['stim3_prob']
stim_3_prob_i=stim_3_prob_i.tolist()



observations=data_all_trials_i['Observation']
#observations=data_all_trials_i['Computer_Observations']
observations=observations.tolist()

def part_block(act,stim_1_prob,stim_2_prob,stim_3_prob, block_end=144, block_start=0):
    correct_ans = []

    prob_trial = [0, 0, 0]

    # these variables will count how many times each option was chosen
    best_choice = 0
    mid_choice = 0
    worst_choice = 0

    trial_ind = 0  # how many trials so far

    accuracy_best = []
    accuracy_mid = []
    accuracy_worst = []

    for ans_ind in range(block_start,block_end):

        trial_ind += 1  # add one trial

        prob_trial = [stim_1_prob[ans_ind], stim_2_prob[ans_ind],
                      stim_3_prob[ans_ind]]  # list of probabilities in the trial

        dummy = int(act[
                        ans_ind]) - 1  # have to substract 1 from the actual action value, so we can use it as an index to find the probability in the list

        prob_current = prob_trial[dummy]  # the probability of the option that was chosen


        # now find out weathe th echosen option was the one with the best, 2nd best or the worst probability

        if prob_current == max(prob_trial):

            correct_ans.append(2)  # if it's the best option, add 2 to the list that holds the accuracy of the outcomes

            best_choice += 1

        elif prob_current == min(prob_trial):

            correct_ans.append(0)

            worst_choice += 1
        else:
            correct_ans.append(1)

            mid_choice += 1



    return best_choice, mid_choice, worst_choice, correct_ans

def count_rewards(correct_ans, obs, trl_start):

    rew_best=0
    rew_mid=0
    rew_worst=0

    neg_best = 0
    neg_mid = 0
    neg_worst = 0



    for trl in range(len(correct_ans)):

        trl_obs=trl+block_start

        if correct_ans[trl]==2:

            if obs[trl_obs]==1:

                rew_best+=1
            else:
                neg_best+=1

        elif correct_ans[trl]==1:

            if obs[trl_obs] == 1:

                rew_mid += 1
            else:
                neg_mid += 1

        elif correct_ans[trl]==0:

            if obs[trl_obs] == 1:

                rew_worst += 1
            else:
                neg_worst += 1

    return rew_best, rew_mid, rew_worst

n_blocks=5
n_trials=36

block_start=0

best_count_p=0
mid_count_p=0
worst_count_p=0

rew_best_p=0
rew_mid_p=0
rew_worst_p=0

best_count_c=0
mid_count_c=0
worst_count_c=0

rew_best_c=0
rew_mid_c=0
rew_worst_c=0


for block in range(n_blocks):


    block_end=(block+1)*n_trials


    best_count_p_curr, mid_count_p_curr, worst_count_p_curr, correct_ans_p=part_block(actions,stim_1_prob_i, stim_2_prob_i, stim_3_prob_i, block_end, block_start)

    best_count_p=best_count_p_curr
    mid_count_p = mid_count_p_curr
    worst_count_p = worst_count_p_curr


    rew_best_p_curr, rew_mid_p_curr, rew_worst_p_curr=count_rewards(correct_ans_p,observations, block_start)

    rew_best_p=rew_best_p_curr
    rew_mid_p = rew_mid_p_curr
    rew_worst_p = rew_worst_p_curr

    best_count_c_curr, mid_count_c_curr, worst_count_c_curr, correct_ans_c=part_block(comp_actions,stim_1_prob, stim_2_prob, stim_3_prob, block_end, block_start)

    best_count_c = best_count_c_curr
    mid_count_c = mid_count_c_curr
    worst_count_c = worst_count_c_curr


    rew_best_c_curr, rew_mid_c_curr, rew_worst_c_curr=count_rewards(correct_ans_c,comp_obs, block_start)

    rew_best_c = rew_best_c_curr
    rew_mid_c = rew_mid_c_curr
    rew_worst_c = rew_worst_c_curr

    '''plotting'''

    x = [1, 2]

    x2 = [1.2, 2.2]
    x3 = [1.4, 2.4]
    ax = plt.subplot(111)
    ax.bar(x, [best_count_p, best_count_c], width=0.2, color='mediumseagreen', alpha=0.5, align='center',
           label='correct')
    ax.bar(x, [rew_best_p, rew_best_c], width=0.17, color='mediumseagreen', align='center')
    ax.bar(x2, [mid_count_p, mid_count_c], width=0.2, color='yellow', alpha=0.4, align='center', label='2nd best')
    ax.bar(x2, [rew_mid_p, rew_mid_c], width=0.17, color='yellow', align='center')
    ax.bar(x3, [worst_count_p, worst_count_c], width=0.2, color='red', alpha=0.5, align='center', label='incorrect')
    ax.bar(x3, [rew_worst_p, rew_worst_c], width=0.17, color='red', align='center')
    ax.set_xticks([1.2, 2.2])
    ax.set_xticklabels(['participant', 'machine'])

    ax.set_ylabel('trial count')
    ax.legend()

    title = 'participant 9598, perceptual instrumental vs. pavlovian'
    ax.set_title(title)

    # plt.bar([0,1,2],[best_count_p, mid_count_p,worst_count_p])
    plt.show()

    block_start = block_end




print(worst_count_c)
print(rew_worst_c)






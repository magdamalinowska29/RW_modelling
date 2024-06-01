import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from accuracy_plotting.extract_accuracies import accuracy_data_subject, accuracy_collapsed_over_blocks

#block_length=36
block_length=36
n_blocks=5

def avg_window( acc_list, window_len=1):

    acc_window=np.zeros(len(acc_list))

    acc_window[0]=acc_list[0]
    acc_window[1] = (acc_list[0] + acc_list[1]) / 2

    acc_window[0] = None
    acc_window[1] = None
    acc_window[2] = None
    acc_window[3] = None

    #acc_window[2] =(acc_list[0]+acc_list[1]+ acc_list[2])/3


    for acc_ind in range(4,(len(acc_list))):

        acc_window[acc_ind]=(acc_list[acc_ind-4]+ acc_list[acc_ind-3]+acc_list[acc_ind-2]+acc_list[acc_ind-1]+acc_list[acc_ind])/5

    #acc_window[len(acc_list)-2]=(acc_list[len(acc_list)-2]+acc_list[len(acc_list)-1])/2
    #acc_window[len(acc_list) - 1] = (acc_list[len(acc_list) - 1])

    #acc_window[len(acc_list) - 2] = None
    #acc_window[len(acc_list) - 1] = None
    return acc_window



paths_act=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_pe_i_example_results_action_observation.csv']
'''
paths_act=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_p_example_results_action_observation.csv']'''

paths_change=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_pe_i_example_results_new.csv']
'''
paths_change=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_p_example_results_new.csv']'''

figure, axes= plt.subplots(len(paths_act),6)

figure.suptitle('Perceptual Instrumental- 4 trials avg', fontsize=16)


for partic_n in range(len(paths_act)):

    path_act=paths_act[partic_n]
    path_change=paths_change[partic_n]

    data_all_env = pd.read_csv(path_change, converters={'Positions': pd.eval,
                                                        'Probabilities': pd.eval})  # note, here i have to specify that the positions column will be trated as a series of lists no strings

    changepoints = data_all_env['Changepoint']
    changepoints = changepoints.tolist()

    first_changepoint = changepoints[0]

    for change_n in range(len(changepoints)):
        changepoints[change_n] = changepoints[change_n] - first_changepoint
        changepoints[change_n] = changepoints[
                                     change_n] + 1  # substracting the number of the first trial in the condition, so i can run the simulation with the data for each condition

    new_pos = data_all_env['Positions']
    new_pos = new_pos.tolist()

    new_probs = data_all_env['Probabilities']
    new_probs = new_probs.tolist()

    data_all_trials = pd.read_csv(path_act)

    actions = data_all_trials['Action']
    actions = actions.tolist()

    observations = data_all_trials['Observation']
    observations = observations.tolist()

    stim_1_prob = data_all_trials['stim1_prob']
    stim_1_prob = stim_1_prob.tolist()

    stim_2_prob = data_all_trials['stim2_prob']
    stim_2_prob = stim_2_prob.tolist()

    stim_3_prob = data_all_trials['stim3_prob']
    stim_3_prob = stim_3_prob.tolist()

    change_where = np.zeros(len(changepoints))

    for chan_ind in range(len(changepoints)):

        curr_prob = [stim_1_prob[changepoints[chan_ind]], stim_2_prob[changepoints[chan_ind]],
                     stim_3_prob[changepoints[chan_ind]]]
        # print(curr_prob)
        # print(changepoints)
        # print(new_pos)

        if len(new_pos[chan_ind]) == 3:
            change_where[chan_ind] = 3

        else:

            if curr_prob[new_pos[chan_ind][0] - 1] == max(curr_prob):
                change_where[chan_ind] = 2
            elif curr_prob[new_pos[chan_ind][0] - 1] == min(curr_prob):
                change_where[chan_ind] = 0
            else:
                change_where[chan_ind] = 1

    acc_best, acc_mid, acc_worst = accuracy_data_subject(path_act,path_change, block_length, n_blocks)

    #acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst = accuracy_collapsed_over_blocks(acc_best, acc_mid,
     #                                                                                                 acc_worst,    n_blocks)
    acc_wind = avg_window(acc_best[0])

    print(acc_wind)

    acc_best_wind = [None] * len(acc_best)

    acc_mid_wind = [None] * len(acc_mid)

    acc_worst_wind = [None] * len(acc_worst)

    changepoints_by_block = [None] * n_blocks

    print(changepoints)

    changes_where_by_block = [None] * n_blocks

    ax = axes[partic_n, 0]
    best_x = [None] * len(acc_best_wind)

    for acc_ind in range(len(acc_best_wind)):
        best_x[acc_ind] = acc_ind + 1

    for block in range(len(acc_best_wind)):

        changes = []
        changes_w = []

        for ch in range(len(changepoints)):

            if (block) * block_length <= changepoints[ch] <= (block + 1) * block_length:
                changes.append(changepoints[ch] - block * block_length)
                changes_w.append(change_where[ch])

        changepoints_by_block[block] = changes
        changes_where_by_block[block] = changes_w

        acc_best_curr = avg_window(acc_best[block])

        acc_best_wind[block] = acc_best_curr

        acc_mid_curr = avg_window(acc_mid[block])

        acc_mid_wind[block] = acc_mid_curr

        acc_worst_curr = avg_window(acc_worst[block])

        acc_worst_wind[block] = acc_worst_curr

    acc_best_wind_4 = [None] * n_blocks
    acc_mid_wind_4 = [None] * n_blocks
    acc_worst_wind_4 = [None] * n_blocks

    for p in range(n_blocks):

        acc_best_wind_4_curr = []
        acc_mid_wind_4_curr = []
        acc_worst_wind_4_curr = []

        for chan in range(len(changepoints_by_block[p])):

            for num in range(6):
                chan_ind = changepoints_by_block[p][chan]
                if chan_ind + num <= 36:
                    acc_best_wind_4_curr.append(acc_best_wind[p][chan_ind + num - 1])
                    acc_mid_wind_4_curr.append(acc_mid_wind[p][chan_ind + num - 1])
                    acc_worst_wind_4_curr.append(acc_worst_wind[p][chan_ind + num - 1])

        acc_best_wind_4[p] = acc_best_wind_4_curr
        acc_mid_wind_4[p] = acc_mid_wind_4_curr
        acc_worst_wind_4[p] = acc_worst_wind_4_curr

    acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst = accuracy_collapsed_over_blocks(acc_best_wind_4,
                                                                                                      acc_mid_wind_4,
                                                                                                      acc_worst_wind_4,
                                                                                                      n_blocks)
    for change in range(len(changes)):
        ax.axvline(x=changes[change],  color='gray', linestyle='--',alpha=0.5)

    ax.plot(range(36), acc_over_blocks_best, c='green', marker='o', label='correct', markersize=1)
    ax.plot(range(36), acc_over_blocks_mid, c='yellow', marker='o', label='2nd best', markersize=1)
    ax.plot(range(36), acc_over_blocks_worst, c='red', marker='o', label='incorrect', markersize=1)
    ax.grid()
    #ax.set_xlabel('trials', fontsize=6)
    ax.set_xticks([])
    ax.set_ylabel('choice %', fontsize=6)
    ax.set_yticks([0, 0.25, 0.5, 0.75,1])
    ax.set_title('participant: ' + str(partic_n+1), fontsize=8)

    for i in range(0,5):

        ax = axes[partic_n,i+1]

        block_number = i + 1

        changes_where = changes_where_by_block[i]



        # participant_id="0"

        best = acc_best[i]
        mid = acc_mid[i]
        worst = acc_worst[i]

        best_x = [None] * len(acc_best[i])
        mid_x = [None] * len(acc_best[i])
        worst_x = [None] * len(acc_best[i])

        for acc_ind in range(len(acc_best[i])):
            best_x[acc_ind] = acc_ind + 1

        ax.plot(best_x, acc_best_wind_4[i], c='green', marker='o', label='correct', markersize=1)
        ax.plot(best_x, acc_mid_wind_4[i], c='yellow', marker='o', label='2nd best', markersize=1)
        ax.plot(best_x, acc_worst_wind_4[i], c='red', marker='o', label='incorrect', markersize=1)
        #for change in range(len(changes)):
         #   ax.axvline(x=changes[change], color='gray', linestyle='--', alpha=0.5)
        ax.grid()


        ax.set_xticks([])
        ax.set_yticks([0, 0.5,1])


        ax.set_title('block ' + str(block_number), fontsize=8)

        for change in range(len(changepoints_by_block[i])):
            curr_val = [best[changepoints_by_block[i][change]], mid[changepoints_by_block[i][change]], worst[changepoints_by_block[i][change]]]
            ymax = max(curr_val)

            if changes_where[change] == 0:
                c = 'red'
                ymax = worst[changepoints_by_block[i][change]]

            elif changes_where[change] == 1:
                c = 'yellow'
                ymax = mid[changepoints_by_block[i][change]]
            elif changes_where[change] == 2:
                c = 'green'
                ymax = best[changepoints_by_block[i][change]]
            elif changes_where[change] == 3:
                c = 'gray'

            ymin = 0
            #

            ax.vlines(x=changepoints_by_block[i][change]+2 , ymin=ymin, ymax=ymax, color=c, linestyle='--')

            if change + 1 < len(changepoints_by_block[i]):

                ax.axvspan(changepoints_by_block[i][change]+2 , changepoints_by_block[i][change + 1]+2 , facecolor=c, alpha=0.15)

            else:
                ax.axvspan(changepoints_by_block[i][change]+2 , 36, facecolor=c, alpha=0.15)

plt.show()

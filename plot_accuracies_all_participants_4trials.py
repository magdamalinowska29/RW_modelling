import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from accuracy_plotting.extract_accuracies import accuracy_data_subject, accuracy_collapsed_over_blocks

#block_length=36
block_length=36
n_blocks=5

paths_act=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_p_example_results_action_observation.csv',
            'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_p_example_results_action_observation.csv'
           ]

paths_change=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_p_example_results_new.csv'
              ]







def avg_window( acc_list, window_len=1):

    acc_window=np.zeros(len(acc_list))

    acc_window[0]=acc_list[0]
    acc_window[1] = (acc_list[0] + acc_list[1]) / 2

    acc_window[0] = None
    acc_window[1] = None
    acc_window[2] = None
    acc_window[3] = None

    #acc_window[2] =(acc_list[0]+acc_list[1]+ acc_list[2])/3


    for acc_ind in range(6,(len(acc_list))):

        acc_window[acc_ind]=(acc_list[acc_ind-6] + acc_list[acc_ind-5]+acc_list[acc_ind-4]+ acc_list[acc_ind-3]+acc_list[acc_ind-2]+acc_list[acc_ind-1]+acc_list[acc_ind])/5

    #acc_window[len(acc_list)-2]=(acc_list[len(acc_list)-2]+acc_list[len(acc_list)-1])/2
    #acc_window[len(acc_list) - 1] = (acc_list[len(acc_list) - 1])

    #acc_window[len(acc_list) - 2] = None
    #acc_window[len(acc_list) - 1] = None
    return acc_window


list_best=[None]*len(paths_act)
list_mid=[None]*len(paths_act)
list_worst=[None]*len(paths_act)

'''main loop'''

for parti in range(len(paths_act)):

    data_all_env = pd.read_csv(paths_change[parti], converters={'Positions': pd.eval,
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

    data_all_trials = pd.read_csv(paths_act[parti])

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

    print(change_where)

    acc_best, acc_mid, acc_worst = accuracy_data_subject(paths_act[parti], \
                                                         paths_change[parti], block_length, n_blocks)






    print(acc_best[0])

    acc_wind=avg_window(acc_best[0])

    print(acc_wind)

    acc_best_wind=[None]*len(acc_best)

    acc_mid_wind=[None]*len(acc_mid)

    acc_worst_wind=[None]*len(acc_worst)

    changepoints_by_block=[None]*n_blocks

    print(changepoints)

    changes_where_by_block = [None] * n_blocks

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

    ''' create a list that tells us weather the change was applied to the best, mid or worst option'''




    print(len(acc_best[0]))
    print(len(acc_best[1]))
    print(len(acc_best[2]))
    print(len(acc_best[3]))
    print(changepoints_by_block)

    acc_best_wind_4=[None]*n_blocks
    acc_mid_wind_4=[None]*n_blocks
    acc_worst_wind_4=[None]*n_blocks

    for p in range(n_blocks):

        acc_best_wind_4_curr=[]
        acc_mid_wind_4_curr = []
        acc_worst_wind_4_curr = []

        for chan in range(len(changepoints_by_block[p])):

            for num in range(6):
                chan_ind = changepoints_by_block[p][chan]
                if chan_ind+num<37:


                    acc_best_wind_4_curr.append(acc_best_wind[p][chan_ind+num-1])
                    acc_mid_wind_4_curr.append(acc_mid_wind[p][chan_ind + num - 1])
                    acc_worst_wind_4_curr.append(acc_worst_wind[p][chan_ind + num - 1])

        acc_best_wind_4[p]=acc_best_wind_4_curr
        acc_mid_wind_4[p] = acc_mid_wind_4_curr
        acc_worst_wind_4[p] = acc_worst_wind_4_curr

    print('acc_best_wind')
    print(acc_best_wind[0])
    print(len(acc_best_wind[0]))
    print('acc_best_wind_4')
    print(acc_best_wind_4[0])
    print(len(acc_best_wind_4[0]))
    changepoints_by_block


    print(len(acc_best_wind_4[0]))
    print(len(acc_best_wind_4[1]))
    print(len(acc_best_wind_4[2]))
    print(len(acc_best_wind_4[3]))

    #n_block_new=min([len(acc_best_wind_4[0]),len(acc_best_wind_4[1]),len(acc_best_wind_4[2]),len(acc_best_wind_4[3])])

    #print(n_block_new)

    acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst=accuracy_collapsed_over_blocks(acc_best, acc_mid, acc_worst, n_blocks)

    print(acc_over_blocks_best)
    print(acc_over_blocks_mid)
    print(acc_over_blocks_worst)



    for block in range(len(acc_best_wind)):

        changes=[]
        changes_w=[]

        for ch in range(len(changepoints)):

            if (block)*block_length<=changepoints[ch]<=(block+1)*block_length:

                changes.append(changepoints[ch]-block*block_length)
                changes_w.append(change_where[ch])



        changepoints_by_block[block]=changes
        changes_where_by_block[block]=changes_w

        acc_best_curr=avg_window(acc_best[block])

        acc_best_wind[block]=acc_best_curr

        acc_mid_curr = avg_window(acc_mid[block])

        acc_mid_wind[block] = acc_mid_curr

        acc_worst_curr = avg_window(acc_worst[block])

        acc_worst_wind[block] = acc_worst_curr

        print(len(acc_best[0]))
        print(len(acc_best[1]))
        print(len(acc_best[2]))
        print(len(acc_best[3]))
        print(changepoints_by_block)

        acc_best_wind_4 = [None] * n_blocks
        acc_mid_wind_4 = [None] * n_blocks
        acc_worst_wind_4 = [None] * n_blocks

        for p in range(n_blocks):

            acc_best_wind_4_curr = []
            acc_mid_wind_4_curr = []
            acc_worst_wind_4_curr = []

            for chan in range(len(changepoints_by_block[p])):

                for num in range(4):
                    chan_ind = changepoints_by_block[p][chan]
                    if chan_ind + num < 36:
                        acc_best_wind_4_curr.append(acc_best_wind[p][chan_ind + num - 1])
                        acc_mid_wind_4_curr.append(acc_mid_wind[p][chan_ind + num - 1])
                        acc_worst_wind_4_curr.append(acc_worst_wind[p][chan_ind + num - 1])

            acc_best_wind_4[p] = acc_best_wind_4_curr
            acc_mid_wind_4[p] = acc_mid_wind_4_curr
            acc_worst_wind_4[p] = acc_worst_wind_4_curr

        print('acc_best_wind')
        print(acc_best_wind[0])
        print(len(acc_best_wind[0]))
        print('acc_best_wind_4')
        print(acc_best_wind_4[0])
        print(len(acc_best_wind_4[0]))
        changepoints_by_block

        print(len(acc_best_wind_4[0]))
        print(len(acc_best_wind_4[1]))
        print(len(acc_best_wind_4[2]))
        print(len(acc_best_wind_4[3]))

        # n_block_new=min([len(acc_best_wind_4[0]),len(acc_best_wind_4[1]),len(acc_best_wind_4[2]),len(acc_best_wind_4[3])])

        # print(n_block_new)

        acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst = accuracy_collapsed_over_blocks(
            acc_best, acc_mid, acc_worst, n_blocks)

    list_best[parti]=acc_over_blocks_best

    list_mid[parti] = acc_over_blocks_mid

    list_worst[parti] = acc_over_blocks_worst



print(list_best)
print(list_mid)
print(list_worst)



avg_best=[]
avg_mid=[]
avg_worst=[]

print('here')
print(len(list_best[0]))
print(len(list_mid[0]))
print(len(list_worst[0]))

print(len(list_best[1]))
print(len(list_mid[1]))
print(len(list_worst[1]))

print(len(list_best[2]))
print(len(list_mid[2]))
print(len(list_worst[2]))

for trl in range(len(list_best[0])):

    avg_best_curr=0
    avg_mid_curr = 0
    avg_worst_curr = 0

    for part in range(len(list_best)):

        avg_best_curr+=list_best[part][trl]
        avg_mid_curr += list_mid[part][trl]
        avg_worst_curr += list_worst[part][trl]

    avg_best.append(avg_best_curr/len(list_best))
    avg_mid.append(avg_mid_curr/len(list_best))
    avg_worst.append(avg_worst_curr/len(list_best))

#print(avg_best[0])
#print(avg_mid[15])
#print(avg_worst[30])



def plot_collapsed_over_blocks(participant_id,condition):
    """plot collapsed over blocks"""

    #participant_id="008998"

    changes_block=[1,6,12,18,24,30,36]

    best_x=[None]*len(acc_over_blocks_best)


    for acc_ind in range(len(acc_over_blocks_best)):
        best_x[acc_ind] = acc_ind + 1

    plt.plot(best_x,avg_best, c='green', marker='o', label='correct')
    plt.plot(best_x, avg_mid, c='yellow', marker='o', label='2nd best')
    plt.plot(best_x, avg_worst, c='red', marker='o', label='incorrect')
    #plt.grid()
    #for change in range(len(changes_block)):
     #   plt.axvline(x=changes_block[change],  color='gray', linestyle='--',alpha=0.5)
    plt.legend()
    plt.xlabel('trials')
    plt.ylabel('choice %, accumulative')
    plt.grid()
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.title( 'participant: ' + participant_id + condition)
    plt.show()

plot_collapsed_over_blocks(participant_id="all participants",condition=", reward_pavlovian")









import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from accuracy_plotting.extract_accuracies import accuracy_data_subject, accuracy_collapsed_over_blocks

#block_length=36
block_length=36
n_blocks=4

path_act= 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data_trial_3/9598_pe_p_example_results_action_observation.csv'
path_change= 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data_trial_3/9598_pe_p_example_results_new.csv'

data_all_env=pd.read_csv(path_change, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

changepoints=data_all_env['Changepoint']
changepoints=changepoints.tolist()

first_changepoint=changepoints[0]

for change_n in range(len(changepoints)):
    changepoints[change_n] = changepoints[change_n] -first_changepoint
    changepoints[change_n]=changepoints[change_n] + 1 # substracting the number of the first trial in the condition, so i can run the simulation with the data for each condition


new_pos=data_all_env['Positions']
new_pos=new_pos.tolist()

new_probs=data_all_env['Probabilities']
new_probs=new_probs.tolist()

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

''' create a list that tells us weather the change was applied to the best, mid or worst option'''

change_where=np.zeros(len(changepoints))

for chan_ind in range(len(changepoints)):

    curr_prob=[stim_1_prob[changepoints[chan_ind]], stim_2_prob[changepoints[chan_ind]], stim_3_prob[changepoints[chan_ind]]]
    #print(curr_prob)
    #print(changepoints)
    #print(new_pos)

    if len(new_pos[chan_ind])==3:
        change_where[chan_ind] = 3

    else:

        if curr_prob[new_pos[chan_ind][0]-1]==max(curr_prob):
            change_where[chan_ind]=2
        elif curr_prob[new_pos[chan_ind][0]-1]==min(curr_prob):
            change_where[chan_ind] = 0
        else:
            change_where[chan_ind]=1

print(change_where)







acc_best, acc_mid, acc_worst=accuracy_data_subject(path_act, \
                                                   path_change, block_length, n_blocks)


def plot_per_block(participant_id, condition, acc_best, acc_mid,acc_worst, changepoints, changes_where_by_block):
    for block_ind in range(len(acc_best)):

        changes_block=changepoints[block_ind]
        block_number=block_ind+1



        changes_where=changes_where_by_block[block_ind]

        #participant_id="0"

        best=acc_best[block_ind]
        mid=acc_mid[block_ind]
        worst=acc_worst[block_ind]

        best_x=[None]*len(acc_best[block_ind])
        mid_x=[None]*len(acc_best[block_ind])
        worst_x=[None]*len(acc_best[block_ind])

        for acc_ind in range(len(acc_best[block_ind])):

            best_x[acc_ind]=acc_ind+1




        plt.plot(best_x, best,c='green', marker='o', label='correct')
        plt.plot(best_x,mid, c='yellow',marker='o', label='2nd best')
        plt.plot(best_x,worst,c='red',marker='o', label='incorrect')

        plt.legend(loc='upper left')
        plt.xlabel('trials')
        plt.ylabel('choice %, accumulative')
        plt.title('block '+str(block_number)+', participant: '+participant_id+ condition)

        for change in range(len(changes_block)):
            curr_val = [best[changes_block[change]], mid[changes_block[change]], worst[changes_block[change]]]
            ymax = max(curr_val)

            if changes_where[change]==0:
                c='red'
                ymax = worst[changes_block[change]]

            elif changes_where[change]==1:
                c='yellow'
                ymax=mid[changes_block[change]]
            elif changes_where[change]==2:
                c='green'
                ymax = best[changes_block[change]]
            elif changes_where[change]==3:
                c='gray'



            ymin=0
#


            plt.vlines(x=changes_block[change]+1,ymin=ymin,ymax=ymax, color=c, linestyle='--')

            if change+1<len(changes_block):

                plt.axvspan(changes_block[change]+1, changes_block[change+1]+1, facecolor=c, alpha=0.15)

            else:
                plt.axvspan(changes_block[change] + 1, 36, facecolor=c, alpha=0.15)

        plt.show()



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


print(acc_best[0])

acc_wind=avg_window(acc_best[0])

print(acc_wind)

acc_best_wind=[None]*len(acc_best)

acc_mid_wind=[None]*len(acc_mid)

acc_worst_wind=[None]*len(acc_worst)

changepoints_by_block=[None]*n_blocks

print(changepoints)

changes_where_by_block=[None]*n_blocks



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


#print(changepoints_by_block)

#plot_per_block('0', ', perceptual_instrumental', acc_best_wind, acc_mid_wind,acc_worst_wind, changepoints_by_block, changes_where_by_block)

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

        for num in range(4):
            chan_ind = changepoints_by_block[p][chan]
            if chan_ind+num<36:


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

acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst=accuracy_collapsed_over_blocks(acc_best_wind_4, acc_mid_wind_4, acc_worst_wind_4, n_blocks)

print(acc_over_blocks_best)
print(acc_over_blocks_mid)
print(acc_over_blocks_worst)



def plot_collapsed_over_blocks(participant_id,condition):
    """plot collapsed over blocks"""

    #participant_id="008998"

    changes_block=[1,5,9,13,17,21,25,29]

    best_x=[None]*len(acc_over_blocks_best)


    for acc_ind in range(len(acc_over_blocks_best)):
        best_x[acc_ind] = acc_ind + 1

    plt.plot(best_x,acc_over_blocks_best, c='green', marker='o', label='correct')
    plt.plot(best_x, acc_over_blocks_mid, c='yellow', marker='o', label='2nd best')
    plt.plot(best_x, acc_over_blocks_worst, c='red', marker='o', label='incorrect')
    #plt.grid()
    for change in range(len(changes_block)):
        plt.axvline(x=changes_block[change],  color='gray', linestyle='--',alpha=0.5)
    plt.legend()
    plt.xlabel('trials')
    plt.ylabel('choice %, accumulative')
    plt.title( 'participant: ' + participant_id + condition)
    plt.show()

plot_collapsed_over_blocks(participant_id="8998",condition=", reward_instrumental")







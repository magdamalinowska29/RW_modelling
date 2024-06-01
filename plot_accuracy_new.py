import matplotlib.pyplot as plt

from accuracy_plotting.extract_accuracies import accuracy_data_subject_new, \
    accuracy_collapsed_over_blocks_new

block_length=36
n_blocks=4



acc_best, acc_mid, acc_worst=accuracy_data_subject_new('C:/Users/magda/PycharmProjects/Rascola_Wagner/data_trial_3/9598_r_p_example_results_action_observation.csv', \
                                                       'C:/Users/magda/PycharmProjects/Rascola_Wagner/data_trial_3/9598_r_p_example_results_new.csv', block_length, n_blocks)


acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst=accuracy_collapsed_over_blocks_new(acc_best, acc_mid, acc_worst, n_blocks)

def plot_per_block(participant_id, condition):
    for block_ind in range(len(acc_best)):
        block_number=block_ind+1

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
        plt.grid()
        plt.legend()
        plt.xlabel('trials')
        plt.ylabel('choice %, accumulative')
        plt.title('block '+str(block_number)+', participant: '+participant_id+ condition)
        plt.show()

#plot_per_block('0',', reward_instrumental')

def plot_collapsed_over_blocks(participant_id,condition):
    """plot collapsed over blocks"""

    #participant_id="008998"

    best_x=[None]*len(acc_over_blocks_best)


    for acc_ind in range(len(acc_over_blocks_best)):
        best_x[acc_ind] = acc_ind + 1

    plt.plot(best_x,acc_over_blocks_best, c='green', marker='o', label='correct')
    plt.plot(best_x, acc_over_blocks_mid, c='yellow', marker='o', label='2nd best')
    plt.plot(best_x, acc_over_blocks_worst, c='red', marker='o', label='incorrect')
    plt.grid()
    plt.legend()
    plt.xlabel('trials')
    plt.ylabel('choice %, accumulative')
    plt.title( 'participant: ' + participant_id + condition)
    plt.show()

plot_collapsed_over_blocks(participant_id="009598",condition=", reward_pavlovian")



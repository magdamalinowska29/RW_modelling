import matplotlib.pyplot as plt

from accuracy_plotting.extract_accuracies import accuracies_all_participants_reward

block_length=36
n_blocks=4

paths_act=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_i_example_results_action_observation.csv',\
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_i_example_results_action_observation.csv',\
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_i_example_results_action_observation.csv']

paths_change=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_i_example_results_new.csv',\
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_i_example_results_new.csv',\
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_i_example_results_new.csv']



acc_all_part_win, acc_all_part_loss =accuracies_all_participants_reward(paths_act,paths_change,block_length,n_blocks)

def plot_collapsed_over_blocks(condition):
    """plot collapsed over blocks"""

    #participant_id="008998"

    best_x=[None]*len(acc_all_part_win)


    for acc_ind in range(len(acc_all_part_win)):
        best_x[acc_ind] = acc_ind + 1

    plt.plot(best_x,acc_all_part_win, c='green', marker='o', label='correct')
    plt.plot(best_x, acc_all_part_loss, c='red', marker='o', label='incorrect')

    plt.grid()
    plt.legend()
    plt.xlabel('trials')
    plt.ylabel('choice %, accumulative')
    plt.title( 'all participants, ' + condition)
    plt.show()

plot_collapsed_over_blocks('reward instrumental')
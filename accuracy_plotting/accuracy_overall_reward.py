import matplotlib.pyplot as plt

from accuracy_plotting.extract_accuracies import accuracies_all_participants_reward

block_length=36
n_blocks=4

n_conditions=4

paths_act_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_i_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_i_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_pe_i_example_results_action_observation.csv']

paths_change_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_i_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_i_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_pe_i_example_results_new.csv']


paths_act_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_p_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_p_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_pe_p_example_results_action_observation.csv']

paths_change_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_p_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_p_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_pe_p_example_results_new.csv']


paths_act_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_i_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_i_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_i_example_results_action_observation.csv']

paths_change_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_i_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_i_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_i_example_results_new.csv']


paths_act_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_p_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_p_example_results_action_observation.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_p_example_results_action_observation.csv']

paths_change_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_p_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_p_example_results_new.csv','C:/Users/magda/PycharmProjects/Rascola_Wagner/data/0_r_p_example_results_new.csv']

"""for a single participant"""

paths_act_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_i_example_results_action_observation.csv']
paths_change_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_i_example_results_new.csv']

paths_act_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_p_example_results_action_observation.csv']
paths_change_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_pe_p_example_results_new.csv']

paths_act_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_i_example_results_action_observation.csv']
paths_change_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_i_example_results_new.csv']

paths_act_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_p_example_results_action_observation.csv']
paths_change_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/8998_r_p_example_results_new.csv']

paths_act=[paths_act_pe_i,paths_act_pe_p,paths_act_r_i,paths_act_r_p]

paths_change=[paths_change_pe_i,paths_change_pe_p,paths_change_r_i,paths_change_r_p]

win_overall=[0]*block_length
loss_overall=[0]*block_length

for trial in range(block_length):

    current_win=0
    current_loss=0

    for cond in range(n_conditions):
        acc_all_part_win, acc_all_part_loss=accuracies_all_participants_reward(paths_act[cond], paths_change[cond], block_length, n_blocks)

        current_win+=acc_all_part_win[trial]
        current_loss+=acc_all_part_loss[trial]

    win_overall[trial]=current_win/n_conditions
    loss_overall[trial]=current_loss/n_conditions

best_x=[None]*len(win_overall)

for acc_ind in range(len(win_overall)):
    best_x[acc_ind] = acc_ind + 1

plt.plot(best_x,win_overall, c='green', marker='o', label='correct')

plt.plot(best_x, loss_overall, c='red', marker='o', label='incorrect')
plt.grid()
plt.legend()
plt.xlabel('trials')
plt.ylabel('choice %, accumulative')
plt.title( '008998, all conditions')
plt.show()




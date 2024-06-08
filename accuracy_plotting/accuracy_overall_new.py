import matplotlib.pyplot as plt

from accuracy_plotting.extract_accuracies import accuracies_all_participants_new

block_length=4
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

paths_act_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_i_example_results_action_observation.csv']
paths_change_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_i_example_results_new.csv']

paths_act_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_p_example_results_action_observation.csv']
paths_change_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_pe_p_example_results_new.csv']

paths_act_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_i_example_results_action_observation.csv']
paths_change_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_i_example_results_new.csv']

paths_act_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_p_example_results_action_observation.csv']
paths_change_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/9598_r_p_example_results_new.csv']

paths_act=[paths_act_pe_i,paths_act_pe_p,paths_act_r_i,paths_act_r_p]

paths_change=[paths_change_pe_i,paths_change_pe_p,paths_change_r_i,paths_change_r_p]


best_overall=[0]*block_length
mid_overall=[0]*block_length
worst_overall=[0]*block_length



for trial in range(block_length):

    current_best=0
    current_mid=0
    current_worst=0

    for cond in range(n_conditions):
        acc_all_part_best, acc_all_part_mid, acc_all_part_worst=accuracies_all_participants_new(paths_act[cond], paths_change[cond], block_length, n_blocks)

        current_best+=acc_all_part_best[trial]
        current_mid+=acc_all_part_mid[trial]
        current_worst+=acc_all_part_worst[trial]

    best_overall[trial]=current_best/n_conditions
    mid_overall[trial]=current_mid/n_conditions
    worst_overall[trial]=current_worst/n_conditions

best_x=[None]*len(best_overall)

for acc_ind in range(len(best_overall)):
    best_x[acc_ind] = acc_ind + 1

plt.plot(best_x,best_overall, c='green', marker='o', label='correct')
plt.plot(best_x, mid_overall, c='yellow', marker='o', label='2nd best')
plt.plot(best_x, worst_overall, c='red', marker='o', label='incorrect')
plt.grid()
plt.legend()
plt.xlabel('trials')
plt.ylabel('choice %, accumulative')
plt.title( '009598, all conditions')
plt.show()






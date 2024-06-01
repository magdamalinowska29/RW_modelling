import matplotlib.pyplot as plt

from accuracy_plotting.extract_accuracies import accuracy_data_subject, accuracy_collapsed_over_blocks

#block_length=36
block_length=36
n_blocks=5



#paths_act=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_p_example_results_action_observation.csv',
     #      'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_p_example_results_action_observation.csv',
       #    'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_p_example_results_action_observation.csv',
      #     'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_p_example_results_action_observation.csv',
     #      'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_p_example_results_action_observation.csv']

paths_act=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_pe_p_example_results_action_observation.csv']

#paths_change=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_p_example_results_new.csv',
 #             'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_p_example_results_new.csv',
  #            'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_p_example_results_new.csv',
   #           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_p_example_results_new.csv',
    #          'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_p_example_results_new.csv']

paths_change=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_pe_p_example_results_new.csv']

figure, axes= plt.subplots(len(paths_act),6)

figure.suptitle('Perceptual Pavlovian', fontsize=16)

changes=[1,7,13,19,25,31]


for partic_n in range(len(paths_act)):

    acc_best, acc_mid, acc_worst = accuracy_data_subject(paths_act[partic_n],paths_change[partic_n], block_length, n_blocks)

    acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst = accuracy_collapsed_over_blocks(acc_best, acc_mid,
                                                                                                      acc_worst,
                                                                                                      n_blocks)

    ax = axes[partic_n, 0]
    best_x = [None] * len(acc_over_blocks_best)

    for acc_ind in range(len(acc_over_blocks_best)):
        best_x[acc_ind] = acc_ind + 1

    for change in range(len(changes)):
        ax.axvline(x=changes[change],  color='gray', linestyle='--',alpha=0.5)

    ax.plot(best_x, acc_over_blocks_best, c='green', marker='o', label='correct', markersize=1)
    ax.plot(best_x, acc_over_blocks_mid, c='yellow', marker='o', label='2nd best', markersize=1)
    ax.plot(best_x, acc_over_blocks_worst, c='red', marker='o', label='incorrect', markersize=1)
    ax.grid()
    #ax.set_xlabel('trials', fontsize=6)
    ax.set_xticks([])
    ax.set_ylabel('choice %', fontsize=6)
    ax.set_yticks([0, 0.25, 0.5, 0.75,1])
    ax.set_title('participant: ' + str(partic_n+6), fontsize=8)

    for i in range(0,5):

        ax = axes[partic_n,i+1]

        block_number = i + 1



        # participant_id="0"

        best = acc_best[i]
        mid = acc_mid[i]
        worst = acc_worst[i]

        best_x = [None] * len(acc_best[i])
        mid_x = [None] * len(acc_best[i])
        worst_x = [None] * len(acc_best[i])

        for acc_ind in range(len(acc_best[i])):
            best_x[acc_ind] = acc_ind + 1

        for change in range(len(changes)):
            ax.axvline(x=changes[change], color='gray', linestyle='--', alpha=0.5)

        ax.plot(best_x, best, c='green', marker='o', label='correct', markersize=1)
        ax.plot(best_x, mid, c='yellow', marker='o', label='2nd best', markersize=1)
        ax.plot(best_x, worst, c='red', marker='o', label='incorrect', markersize=1)
        ax.grid()


        ax.set_xticks([])
        ax.set_yticks([0, 0.5,1])


        ax.set_title('block ' + str(block_number), fontsize=8)

plt.show()

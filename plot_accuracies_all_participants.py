import matplotlib.pyplot as plt

from accuracy_plotting.extract_accuracies import accuracies_all_participants

block_length=36
n_blocks=5

paths_act_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_r_p_example_results_action_observation.csv'
               ]

paths_act_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_i_example_results_action_observation.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_r_i_example_results_action_observation.csv'
               ]


paths_act_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_pe_i_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_pe_i_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_pe_i_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_i_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_pe_i_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_pe_i_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_i_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_pe_i_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_pe_i_example_results_action_observation.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_pe_i_example_results_action_observation.csv'
                ]

paths_act_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_pe_p_example_results_action_observation.csv',
               'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_pe_p_example_results_action_observation.csv',
                'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_p_example_results_action_observation.csv',
            'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_pe_p_example_results_action_observation.csv',
           'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_pe_p_example_results_action_observation.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_pe_p_example_results_action_observation.csv'
                ]

paths_change_r_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_p_example_results_new.csv',
                 'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_p_example_results_new.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_r_p_example_results_new.csv'
                  ]

paths_change_r_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_r_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_r_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_r_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_r_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_r_i_example_results_new.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_r_i_example_results_new.csv'
                  ]


paths_change_pe_i=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_pe_i_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_pe_i_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_pe_i_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_i_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_pe_i_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_pe_i_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_i_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_pe_i_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_pe_i_example_results_new.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_pe_i_example_results_new.csv'
                   ]

paths_change_pe_p=['C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4108_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4870_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5233_pe_p_example_results_new.csv',
              'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5620_pe_p_example_results_new.csv',
                  'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5764_pe_p_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_p_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/1111_pe_p_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/4564_pe_p_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5119_pe_p_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5419_pe_p_example_results_new.csv',
                   'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/3580_pe_p_example_results_new.csv',
'C:/Users/magda/PycharmProjects/Rascola_Wagner/data/5629_pe_p_example_results_new.csv'
                   ]

paths_act=[paths_act_r_i, paths_act_r_p, paths_act_pe_i, paths_act_pe_p]

paths_change=[paths_change_r_i, paths_change_r_p, paths_change_pe_i, paths_change_pe_p]



def plot_collapsed_over_blocks(condition, ax):
    """plot collapsed over blocks"""

    #participant_id="008998"

    best_x=[None]*len(acc_all_part_best)


    for acc_ind in range(len(acc_all_part_best)):
        best_x[acc_ind] = acc_ind + 1

    ax.plot(best_x,acc_all_part_best, c='green', marker='o', label='correct')
    ax.plot(best_x, acc_all_part_mid, c='yellow', marker='o', label='2nd best')
    ax.plot(best_x, acc_all_part_worst, c='red', marker='o', label='incorrect')
    ax.grid()
    #ax.legend()
    #ax.set_xlabel('trials')
    ax.set_ylabel('choice %, accumulative')
    ax.set_title( 'all participants, ' + condition)
    ax.set_yticks([0,0.25,0.5,0.75,1])


figure, axes=plt.subplots(2,2)

acc_all_part_best, acc_all_part_mid, acc_all_part_worst=accuracies_all_participants(paths_act_r_i,paths_change_r_i,block_length,n_blocks)

plot_collapsed_over_blocks('Reward instrumental', axes[0,0])

acc_all_part_best, acc_all_part_mid, acc_all_part_worst=accuracies_all_participants(paths_act_r_p,paths_change_r_p,block_length,n_blocks)

plot_collapsed_over_blocks('Reward pavlovian', axes[1,0])

acc_all_part_best, acc_all_part_mid, acc_all_part_worst=accuracies_all_participants(paths_act_pe_p,paths_change_pe_p,block_length,n_blocks)

plot_collapsed_over_blocks('Perceptual pavlovian', axes[1,1])

acc_all_part_best, acc_all_part_mid, acc_all_part_worst=accuracies_all_participants(paths_act_pe_i,paths_change_pe_i,block_length,n_blocks)

plot_collapsed_over_blocks('Perceptual instrumental', axes[0,1])

plt.show()








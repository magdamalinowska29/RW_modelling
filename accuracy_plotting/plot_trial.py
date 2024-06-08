import matplotlib.pyplot as plt
import pandas as pd

"""upload the data from the experiment"""

data_all_trials=pd.read_csv('/data/sub004_r_i_example_results_action_observation.csv')

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

data_all_env=pd.read_csv('/data/sub004_r_i_example_results_new.csv', converters={'Positions':pd.eval, 'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

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

T=len(actions)


correct_ans=[]

prob_trial=[0,0,0]

for ans_ind in range(len(actions)):

    prob_trial=[stim_1_prob[ans_ind],stim_2_prob[ans_ind],stim_3_prob[ans_ind]]

    dummy=int(actions[ans_ind])-1

    prob_current=prob_trial[dummy]
    print(prob_trial)
    print(prob_current)

    if prob_current==max(prob_trial):

        correct_ans.append(2)

    elif prob_current==min(prob_trial):

        correct_ans.append(0)
    else:
        correct_ans.append(1)

print(correct_ans)

def trial_plot():

    fig_1,ax=plt.subplots(nrows=2,ncols=1)

    #heat_map=ax[0].imshow(act_prob, cmap='winter')

    #fig_1.colorbar(heat_map,orientation="horizontal")

    #plt.title("Participant 009598, perceptual pavlovian condition")

    ax[0].scatter(range(1,T+1),actions,c='magenta')

    #make lists of changepoints and positions that were used in the simulation
    changepoints_used=[]
    new_pos_used=[]

    for point_ind in range(len(changepoints)):
        if changepoints[point_ind]<=T:
            changepoints_used.append(changepoints[point_ind])
            new_pos_used.append(new_pos[point_ind])

    xticks_labels=[None]*len(changepoints_used)

    ax[0].set_title("sub004, reward instrumental")
    ax[0].set_yticks([0,1,2,3])
    ax[0].set_xticks(changepoints_used)
    ax[0].set_yticklabels(['Start','Choose 1','Choose 2','Choose 3'])

    pos_dict={0:'Start',1:'1',2:'2',3:'3'}

    for change_ind in range(len(changepoints_used)):

        change_pos=new_pos[change_ind]

        change_list=[]

        for m in change_pos:

            change_list.append(pos_dict[m])

        xticks_labels[change_ind]= str(change_list[0])

        if len(change_pos)==3:
            xticks_labels[change_ind]='block'

    ax[0].set_xticklabels(xticks_labels)

    '''scatter plot with reward outcomes'''

    wins=[]
    wins_ind=[]
    losses=[]
    losses_ind=[]
    mid=[]
    mid_ind=[]
    for outcome_ind in range(len(observations)):

        if correct_ans[outcome_ind]==2:

            wins.append(1)
            wins_ind.append(outcome_ind+1) #adding +1 to account for the o trial being the first trial in plot (so it matches the action plot

        elif correct_ans[outcome_ind]==0:

            losses.append(-1)
            losses_ind.append(outcome_ind+1)

        elif correct_ans[outcome_ind]==1:

            mid.append(0)
            mid_ind.append(outcome_ind+1)





    #ax[1].scatter(range(T),reward_obs_plot, s=100, c='orange')
    #ax[1].set_yticks([1,2])
    #ax[1].set_yticklabels(['Correct','Incorrect'])
    #ax[1].set_title("reward outcomes")

    ax[1].scatter(wins_ind,wins, s=100, c='green')
    ax[1].scatter(losses_ind,losses, s=100, c='red')
    ax[1].scatter(mid_ind, mid, s=100, c='yellow')
    ax[1].set_yticks([-1,0,1])
    ax[1].set_yticklabels(['Incorrect','2nd best','Correct'])
    ax[1].set_title("Outcomes")

    #plt.show()


    plt.show()

"""calling the plotting function"""

trial_plot()

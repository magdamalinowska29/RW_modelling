import pandas as pd


def accuracy_block(actions,stim_1_prob,stim_2_prob,stim_3_prob, block_end=144, block_start=0):
    correct_ans = []

    prob_trial = [0, 0, 0]

    # these variables will count how many times each option was chosen
    best_choice = 0
    mid_choice = 0
    worst_choice = 0

    trial_ind = 0  # how many trials so far

    accuracy_best = []
    accuracy_mid = []
    accuracy_worst = []

    for ans_ind in range(block_start,block_end):

        trial_ind += 1  # add one trial

        prob_trial = [stim_1_prob[ans_ind], stim_2_prob[ans_ind],
                      stim_3_prob[ans_ind]]  # list of probabilities in the trial

        dummy = int(actions[
                        ans_ind]) - 1  # have to substract 1 from the actual action value, so we can use it as an index to find the probability in the list

        prob_current = prob_trial[dummy]  # the probability of the option that was chosen


        # now find out weathe th echosen option was the one with the best, 2nd best or the worst probability

        if prob_current == max(prob_trial):

            correct_ans.append(2)  # if it's the best option, add 2 to the list that holds the accuracy of the outcomes

            best_choice += 1

        elif prob_current == min(prob_trial):

            correct_ans.append(0)

            worst_choice += 1
        else:
            correct_ans.append(1)

            mid_choice += 1

        accuracy_best.append(best_choice / trial_ind)
        accuracy_mid.append(mid_choice / trial_ind)
        accuracy_worst.append(worst_choice / trial_ind)

        if ans_ind >= block_end:
            break



    return accuracy_best,accuracy_mid,accuracy_worst

def accuracy_block(actions,stim_1_prob,stim_2_prob,stim_3_prob, block_end=144, block_start=0):
    correct_ans = []

    prob_trial = [0, 0, 0]

    # these variables will count how many times each option was chosen
    best_choice = 0
    mid_choice = 0
    worst_choice = 0

    trial_ind = 0  # how many trials so far

    accuracy_best = []
    accuracy_mid = []
    accuracy_worst = []

    for ans_ind in range(block_start,block_end):

        trial_ind += 1  # add one trial

        prob_trial = [stim_1_prob[ans_ind], stim_2_prob[ans_ind],
                      stim_3_prob[ans_ind]]  # list of probabilities in the trial

        dummy = int(actions[
                        ans_ind]) - 1  # have to substract 1 from the actual action value, so we can use it as an index to find the probability in the list

        prob_current = prob_trial[dummy]  # the probability of the option that was chosen


        # now find out weathe th echosen option was the one with the best, 2nd best or the worst probability

        if prob_current == max(prob_trial):

            correct_ans.append(2)  # if it's the best option, add 2 to the list that holds the accuracy of the outcomes

            best_choice += 1

        elif prob_current == min(prob_trial):

            correct_ans.append(0)

            worst_choice += 1
        else:
            correct_ans.append(1)

            mid_choice += 1

        accuracy_best.append(best_choice / trial_ind)
        accuracy_mid.append(mid_choice / trial_ind)
        accuracy_worst.append(worst_choice / trial_ind)

        if ans_ind >= block_end:
            break



    return accuracy_best,accuracy_mid,accuracy_worst

def accuracy_block_rewards(actions,observations, block_end=144, block_start=0):


    trial_ind = 0  # how many trials so far

    wins = 0
    losses = 0

    #wins_ind = []
    #losses_ind = []

    accuracy_wins=[]
    accuracy_losses=[]

    for ans_ind in range(block_start,block_end):

        trial_ind += 1  # add one trial





        # now find out weathe th echosen option was the one with the best, 2nd best or the worst probability

        if observations[ans_ind] ==1:
            wins+=1
     #       wins_ind.append(ans_ind + 1)  # adding +1 to account for the o trial being the first trial in plot (so it matches the action plot

        elif observations[ans_ind] ==2:

            losses+=1
      #      losses_ind.append(ans_ind + 1)


        accuracy_wins.append(wins / trial_ind)
        accuracy_losses.append(losses / trial_ind)


        if ans_ind >= block_end:
            break



    return accuracy_wins,accuracy_losses

def accuracy_data_subject(path_act, path_change, block_length, n_blocks):
    """upload the data from the experiment"""

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

    T=len(actions)

    acc_best=[None]*n_blocks
    acc_mid = [None] * n_blocks
    acc_worst = [None] * n_blocks

    for block in range(n_blocks):

        block_end=(block+1)*block_length

        block_start=block*block_length

        acc_best_curr, acc_mid_curr, acc_worst_curr=accuracy_block(actions, stim_1_prob,stim_2_prob,stim_3_prob,block_end, block_start)


        acc_best[block]=acc_best_curr
        acc_mid[block]=acc_mid_curr
        acc_worst[block]=acc_worst_curr


    return acc_best,acc_mid, acc_worst

def accuracy_data_subject_new(path_act, path_change, block_length, n_blocks):
    """upload the data from the experiment"""

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

    data_all_env=pd.read_csv(path_change, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

    changepoints=data_all_env['Changepoint']
    changepoints=changepoints.tolist()

    first_changepoint=changepoints[0]

    for change_n in range(len(changepoints)):
        changepoints[change_n] = changepoints[change_n] -first_changepoint
        #changepoints[change_n]=changepoints[change_n] + 1 # substracting the number of the first trial in the condition, so i can run the simulation with the data for each condition


    new_pos=data_all_env['Positions']
    new_pos=new_pos.tolist()

    new_probs=data_all_env['Probabilities']
    new_probs=new_probs.tolist()

    T=len(actions)

    acc_best=[None]*len(changepoints)
    acc_mid = [None] * len(changepoints)
    acc_worst = [None] *len(changepoints)

    for change in range(len(changepoints)):

        if change==len(changepoints)-1:

            block_end=changepoints[change]+4
        else:
            block_end=changepoints[change+1]

        block_start=changepoints[change]

        acc_best_curr, acc_mid_curr, acc_worst_curr=accuracy_block(actions, stim_1_prob,stim_2_prob,stim_3_prob,block_end, block_start)


        acc_best[change]=acc_best_curr
        acc_mid[change]=acc_mid_curr
        acc_worst[change]=acc_worst_curr


    return acc_best,acc_mid, acc_worst

def accuracy_data_subject_rewards(path_act, path_change, block_length, n_blocks):
    """upload the data from the experiment"""

    data_all_trials=pd.read_csv(path_act)

    actions=data_all_trials['Action']
    actions=actions.tolist()

    observations=data_all_trials['Observation']
    observations=observations.tolist()



    data_all_env=pd.read_csv(path_change, converters={'Positions':pd.eval,'Probabilities':pd.eval}) #note, here i have to specify that the positions column will be trated as a series of lists no strings

    changepoints=data_all_env['Changepoint']
    changepoints=changepoints.tolist()

    first_changepoint=changepoints[0]

    #for change_n in range(len(changepoints)):
     #   changepoints[change_n] = changepoints[change_n] -first_changepoint
      #  changepoints[change_n]=changepoints[change_n] + 1 # substracting the number of the first trial in the condition, so i can run the simulation with the data for each condition


    new_pos=data_all_env['Positions']
    new_pos=new_pos.tolist()

    new_probs=data_all_env['Probabilities']
    new_probs=new_probs.tolist()

    T=len(actions)

    acc_win=[None]*n_blocks
    acc_loss = [None] * n_blocks


    for block in range(n_blocks):


        block_end=(block+1)*block_length

        block_start=block*block_length

        acc_win_curr, acc_loss_curr=accuracy_block_rewards(actions, observations,block_end, block_start)


        acc_win[block]=acc_win_curr
        acc_loss[block]=acc_loss_curr



    return acc_win,acc_loss


def accuracy_collapsed_over_blocks(acc_best, acc_mid, acc_worst, n_blocks):
    acc_over_blocks_best = []
    acc_over_blocks_mid = []
    acc_over_blocks_worst = []

    dum=min([len(acc_best[0]),len(acc_best[1]),len(acc_best[2]),len(acc_best[3]),len(acc_best[4])])

    for trial in range(36):

        best_hold = 0
        mid_hold = 0
        worst_hold = 0

        for block in range(n_blocks):
            best_hold += acc_best[block][trial]
            mid_hold += acc_mid[block][trial]
            worst_hold += acc_worst[block][trial]

        acc_over_blocks_best.append(best_hold / n_blocks)
        acc_over_blocks_mid.append(mid_hold / n_blocks)
        acc_over_blocks_worst.append(worst_hold / n_blocks)

    return acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst

def accuracy_collapsed_over_blocks_new(acc_best, acc_mid, acc_worst, n_blocks):
    acc_over_blocks_best = []
    acc_over_blocks_mid = []
    acc_over_blocks_worst = []

    for trial in range(n_blocks):

        best_hold = 0
        mid_hold = 0
        worst_hold = 0

        for block in range(n_blocks):
            best_hold += acc_best[block][trial]
            mid_hold += acc_mid[block][trial]
            worst_hold += acc_worst[block][trial]

        acc_over_blocks_best.append(best_hold / n_blocks)
        acc_over_blocks_mid.append(mid_hold / n_blocks)
        acc_over_blocks_worst.append(worst_hold / n_blocks)

    return acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst

def accuracy_collapsed_over_blocks_rewards(acc_win, acc_loss, n_blocks):
    acc_over_blocks_win = []
    acc_over_blocks_loss = []


    for trial in range(len(acc_win[0])):

        win_hold = 0
        loss_hold = 0


        for block in range(n_blocks):
            win_hold += acc_win[block][trial]
            loss_hold += acc_loss[block][trial]


        acc_over_blocks_win.append(win_hold / n_blocks)
        acc_over_blocks_loss.append(loss_hold / n_blocks)


    return acc_over_blocks_win, acc_over_blocks_loss

def accuracies_all_participants(paths_act, paths_change, block_length, n_blocks):
    acc_all_part_best=[0]*block_length
    acc_all_part_mid=[0]*block_length
    acc_all_part_worst=[0]*block_length

    acc_best_sum=0
    acc_mid_sum=0
    acc_worst_sum=0

    #loop over participants
    for participant in range(len(paths_act)):

        acc_best, acc_mid, acc_worst=accuracy_data_subject(paths_act[participant],paths_change[participant], block_length,n_blocks)

        acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst = accuracy_collapsed_over_blocks(acc_best, acc_mid,
                                                                                                          acc_worst,
                                                                                                          n_blocks)



        for trial in range(len(acc_over_blocks_worst)):

            acc_trial_best= acc_all_part_best[trial]

            acc_trial_best+=acc_over_blocks_best[trial]

            acc_all_part_best[trial]=acc_trial_best

            #mid

            acc_trial_mid = acc_all_part_mid[trial]

            acc_trial_mid += acc_over_blocks_mid[trial]

            acc_all_part_mid[trial] = acc_trial_mid

            #worst

            acc_trial_worst = acc_all_part_worst[trial]

            acc_trial_worst += acc_over_blocks_worst[trial]

            acc_all_part_worst[trial] = acc_trial_worst

    for trl in range(block_length):

        acc_all_part_best[trl]=acc_all_part_best[trl]/len(paths_act)
        acc_all_part_mid[trl] = acc_all_part_mid[trl] / len(paths_act)
        acc_all_part_worst[trl] = acc_all_part_worst[trl] / len(paths_act)



    return acc_all_part_best, acc_all_part_mid, acc_all_part_worst

def accuracies_all_participants_new(paths_act, paths_change, block_length, n_blocks):
    acc_all_part_best=[0]*block_length
    acc_all_part_mid=[0]*block_length
    acc_all_part_worst=[0]*block_length

    acc_best_sum=0
    acc_mid_sum=0
    acc_worst_sum=0

    #loop over participants
    for participant in range(len(paths_act)):

        acc_best, acc_mid, acc_worst=accuracy_data_subject_new(paths_act[participant],paths_change[participant], block_length,n_blocks)

        acc_over_blocks_best, acc_over_blocks_mid, acc_over_blocks_worst = accuracy_collapsed_over_blocks_new(acc_best, acc_mid,
                                                                                                          acc_worst,
                                                                                                          n_blocks)



        for trial in range(len(acc_over_blocks_worst)):

            acc_trial_best= acc_all_part_best[trial]

            acc_trial_best+=acc_over_blocks_best[trial]

            acc_all_part_best[trial]=acc_trial_best

            #mid

            acc_trial_mid = acc_all_part_mid[trial]

            acc_trial_mid += acc_over_blocks_mid[trial]

            acc_all_part_mid[trial] = acc_trial_mid

            #worst

            acc_trial_worst = acc_all_part_worst[trial]

            acc_trial_worst += acc_over_blocks_worst[trial]

            acc_all_part_worst[trial] = acc_trial_worst

    for trl in range(block_length):

        acc_all_part_best[trl]=acc_all_part_best[trl]/len(paths_act)
        acc_all_part_mid[trl] = acc_all_part_mid[trl] / len(paths_act)
        acc_all_part_worst[trl] = acc_all_part_worst[trl] / len(paths_act)



    return acc_all_part_best, acc_all_part_mid, acc_all_part_worst


def accuracies_all_participants_reward(paths_act, paths_change, block_length, n_blocks):
    acc_all_part_win=[0]*block_length
    acc_all_part_loss=[0]*block_length


    acc_win_sum=0
    acc_loss_sum=0


    #loop over participants
    for participant in range(len(paths_act)):

        acc_win, acc_loss=accuracy_data_subject_rewards(paths_act[participant],paths_change[participant], block_length,n_blocks)

        acc_over_blocks_win, acc_over_blocks_loss = accuracy_collapsed_over_blocks_rewards(acc_win, acc_loss,n_blocks)



        for trial in range(len(acc_over_blocks_loss)):

            acc_trial_win= acc_all_part_win[trial]

            acc_trial_win+=acc_over_blocks_win[trial]

            acc_all_part_win[trial]=acc_trial_win

            #loss

            acc_trial_loss = acc_all_part_loss[trial]

            acc_trial_loss += acc_over_blocks_loss[trial]

            acc_all_part_loss[trial] = acc_trial_loss



    for trl in range(block_length):

        acc_all_part_win[trl]=acc_all_part_win[trl]/len(paths_act)
        acc_all_part_loss[trl] = acc_all_part_loss[trl] / len(paths_act)




    return acc_all_part_win, acc_all_part_loss



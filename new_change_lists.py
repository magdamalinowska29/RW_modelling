import random


n_trials=36
n_blocks=4

n_total=500

pos_probs=[0.1,0.3,0.5,0.7,0.9]

changepoints=[]
new_pos=[]
new_probs=[]

change_n=6

curr_probs=[None]*3

if_change=0
for i in range(n_total):

    if if_change>=change_n:
        if_change=0

    if i in [0, 36, 72, 108,144,180,216,252,288,324,360,396,432,468]:
        changepoints.append(i)
        new_pos.append([1, 2, 3])

        prob1 = random.choice(pos_probs)
        prob2 = random.choice(pos_probs)
        prob3 = random.choice(pos_probs)

        new_probs.append([prob1, prob2, prob3])

        curr_probs=[prob1,prob2,prob3]

    elif if_change==0:

        for j in range(5):

            new_prob_curr=random.choice(pos_probs)

            if new_prob_curr not in curr_probs:

                break

        new_probs.append([new_prob_curr])


        changepoints.append(i)

        new_pos_curr=random.randint(1,3)
        new_pos.append([new_pos_curr])

        curr_probs[new_pos_curr-1]=new_prob_curr

        #new_prob_curr=random.randint(0,4)
        #new_probs.append([pos_probs[new_prob_curr]])

    if_change+=1

print(changepoints)
print(new_pos)
print(new_probs)




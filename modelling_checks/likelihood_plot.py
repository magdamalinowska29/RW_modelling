from matplotlib import pyplot as plt
from models.Rescorla_Wagner_model import Rescorla_Wagner_model


changepoints=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426, 432, 438, 444, 450, 456, 462, 468, 474, 480, 486, 492, 498]
new_pos=[[1, 2, 3], [3], [1], [2], [1], [2], [1, 2, 3], [2], [2], [2], [2], [3], [1, 2, 3], [1], [2], [1], [1], [3], [1, 2, 3], [3], [1], [1], [3], [2], [1, 2, 3], [2], [3], [2], [3], [3], [1, 2, 3], [3], [2], [3], [2], [2], [1, 2, 3], [1], [2], [1], [3], [2], [1, 2, 3], [2], [3], [1], [3], [2], [1, 2, 3], [3], [1], [2], [1], [3], [1, 2, 3], [3], [3], [1], [1], [2], [1, 2, 3], [2], [3], [1], [3], [3], [1, 2, 3], [1], [2], [3], [3], [2], [1, 2, 3], [1], [1], [1], [3], [1], [1, 2, 3], [3], [1], [2], [1], [2]]
new_probs=[[0.1, 0.1, 0.5], [0.3], [0.9], [0.7], [0.9], [0.3], [0.7, 0.9, 0.5], [0.3], [0.1], [0.7], [0.3], [0.9], [0.7, 0.7, 0.3], [0.9], [0.3], [0.1], [0.1], [0.3], [0.3, 0.5, 0.7], [0.1], [0.9], [0.7], [0.1], [0.7], [0.7, 0.3, 0.3], [0.9], [0.1], [0.7], [0.1], [0.5], [0.3, 0.5, 0.5], [0.1], [0.7], [0.1], [0.1], [0.9], [0.3, 0.5, 0.9], [0.7], [0.1], [0.5], [0.3], [0.9], [0.5, 0.9, 0.9], [0.3], [0.7], [0.1], [0.5], [0.1], [0.9, 0.7, 0.9], [0.5], [0.1], [0.3], [0.7], [0.5], [0.5, 0.9, 0.7], [0.1], [0.3], [0.7], [0.5], [0.9], [0.5, 0.9, 0.1], [0.7], [0.9], [0.3], [0.7], [0.9], [0.9, 0.5, 0.7], [0.1], [0.5], [0.5], [0.3], [0.9], [0.3, 0.1, 0.1], [0.9], [0.5], [0.1], [0.7], [0.1], [0.5, 0.7, 0.7], [0.3], [0.9], [0.5], [0.7], [0.1]]





count=101

lrs=[0.1,0.25,0.5,0.75,1]

for j in range(len(lrs)):

    sim_lr=lrs[j]

    RW=Rescorla_Wagner_model()

    a,r= RW.simulate_change(180, [0.5,0.5,0.5], sim_lr, 4, changepoints, new_pos, new_probs)


    lr_val=[]
    neg_val=[]

    lr=0
    actual_negLL=0



    for i in range(count):
        print(i)


        '''fitting'''



        NegLL=RW.likelihood_change(a,r, lr, 4, changepoints, new_pos)

        lr_val.append(lr)

        neg_val.append(NegLL)



        lr+=0.01


    marker_on=[sim_lr]
    actual_negLL=neg_val[int(sim_lr/0.01)]
    plt.plot(lr_val, neg_val, c='blue', markevery=marker_on, label='negative log likelihood')
    plt.scatter(marker_on,[actual_negLL],c='red', s=40)

    plt.title('NegLL for learning rate= ' + str(sim_lr))
    plt.xlabel('lr')
    plt.ylabel('NegLL')



    plt.grid()
    plt.show()
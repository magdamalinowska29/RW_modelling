
from models.Rescorla_Wagner_model import Rescorla_Wagner_model
from models.Rescorla_Wagner_Choice_Kernel_model import Rescorla_Wagner_CK_model
from models.Rescorla_Wagner_sep_lr_model import Rescorla_Wagner_sep_lr_model
from models.Rescorla_Wagner_sep_temp_model import Rescorla_Wagner_sep_temp_model
import numpy as np
from matplotlib import pyplot as plt

from maths import get_truncated_normal

def fit_all(a,r, changepoints, new_pos):



    '''
    a function that fits all the available models to the given data and finds the one with the lowest BIC

    :param a: list of int, simulated actions

    :param r: list of int, simulated reward outcomes

    :return:
    '''

    BIC=[None]*4

    RW=Rescorla_Wagner_model()
    RW_CK=Rescorla_Wagner_CK_model()
    RW_sep_lr=Rescorla_Wagner_sep_lr_model()
    RW_sep_temp=Rescorla_Wagner_sep_temp_model()

    #BIC[0]=1000
    Xfit_1,LL_1, BIC[0], b1= RW.fit_change(a,r,changepoints, new_pos)
    Xfit_2,LL_2, BIC[1],b2  = RW_CK.fit_change(a, r, changepoints, new_pos)
    Xfit_3,LL_3, BIC[2], b3   = RW_sep_lr.fit_change(a, r, changepoints, new_pos)
    #Xfit_4,LL_4,  BIC[3], b4   = RW_sep_temp.fit_change(a, r, changepoints, new_pos)
    #Xfit_1, LL_1,BIC[0],a1 = RW.fit(a, r)
    #Xfit_2, LL_2,BIC[1],a2  = RW_CK.fit(a, r)
    #BIC[1]=0
    #Xfit_3, LL_3,BIC[2],a3 = RW_sep_lr.fit(a, r)
    #BIC[2]=10000
    #Xfit_4, LL_4, b4, BIC[3] = RW_sep_temp.fit(a, r)
    BIC[3]=10000

    model_dict={0:'RW',1:'RW_CK',2:'RW_sep_lr',3:'RW_sep_temp'}

    best_bic=min(BIC)

    best=[0]*4
    for i in range(len(BIC)):

        if BIC[i]==best_bic:
            best[i]=1

    for j in range(len(best)):

        sumed=sum(best)
        if sumed!=0:
            best[j]=best[j]/sumed


    return BIC, best_bic, best, model_dict

def fit_all_AIC(a,r, changepoints, new_pos):

    '''
    a function that fits all the available models to the given data and finds the one with the lowest BIC

    :param a: list of int, simulated actions

    :param r: list of int, simulated reward outcomes

    :return:
    '''

    AIC=[None]*4

    RW=Rescorla_Wagner_model()
    RW_CK=Rescorla_Wagner_CK_model()
    RW_sep_lr=Rescorla_Wagner_sep_lr_model()
    RW_sep_temp=Rescorla_Wagner_sep_temp_model()

    #AIC[0]=1000
    Xfit_1,LL_1, bic1, AIC[0]= RW.fit_change(a,r,changepoints, new_pos)
    Xfit_2,LL_2, bic2,AIC[1]  = RW_CK.fit_change(a, r, changepoints, new_pos)
    Xfit_3,LL_3, bic3, AIC[2]   = RW_sep_lr.fit_change(a, r, changepoints, new_pos)
    Xfit_4,LL_4, bic4, AIC[3]   = RW_sep_temp.fit_change(a, r, changepoints, new_pos)
    #Xfit_1, LL_1,BIC[0],a1 = RW.fit(a, r)
    #Xfit_2, LL_2,BIC[1],a2  = RW_CK.fit(a, r)
    #BIC[1]=0
    #Xfit_3, LL_3,BIC[2],a3 = RW_sep_lr.fit(a, r)
    #BIC[2]=10000
    #Xfit_4, LL_4, b4, BIC[3] = RW_sep_temp.fit(a, r)
    AIC[3]=10000

    model_dict={0:'RW',1:'RW_CK',2:'RW_sep_lr',3:'RW_sep_temp'}

    best_aic=min(AIC)

    best=[0]*4
    for i in range(len(AIC)):

        if AIC[i]==best_aic:
            best[i]=1

    for j in range(len(best)):

        sumed=sum(best)
        if sumed!=0:
            best[j]=best[j]/sumed


    return AIC, best_aic, best, model_dict

def fit_all_LL(a,r, changepoints, new_pos):

    '''
    a function that fits all the available models to the given data and finds the one with the lowest BIC

    :param a: list of int, simulated actions

    :param r: list of int, simulated reward outcomes

    :return:
    '''

    LL=[None]*4

    RW=Rescorla_Wagner_model()
    RW_CK=Rescorla_Wagner_CK_model()
    RW_sep_lr=Rescorla_Wagner_sep_lr_model()
    RW_sep_temp=Rescorla_Wagner_sep_temp_model()

    #LL[0]=0
    Xfit_1,LL[0], bic1,aic1= RW.fit_change(a,r,changepoints, new_pos)
    Xfit_2,LL[1], bic2, aic2  = RW_CK.fit_change(a, r, changepoints, new_pos)
    #LL[1]=0
    Xfit_3,LL[2], bic3,aic3  = RW_sep_lr.fit_change(a, r, changepoints, new_pos)
    #Xfit_4,LL[3], bic4, aic4   = RW_sep_temp.fit_change(a, r, changepoints, new_pos)
    #Xfit_1, LL[0],b1,bic1 = RW.fit(a, r)
    #Xfit_2, LL[1],b2 ,bic2  = RW_CK.fit(a, r)
    #Xfit_3, LL[2],b1,  bic3 = RW_sep_lr.fit(a, r)
    #LL[2]=0
    #Xfit_4, LL[3], b4, bic4 = RW_sep_temp.fit(a, r)
    LL[3]=-1000

    model_dict={0:'RW',1:'RW_CK',2:'RW_sep_lr',3:'RW_sep_temp'}

    best_ll=max(LL)

    best=[0]*4
    for i in range(len(LL)):

        if LL[i]==best_ll:
            best[i]=1

    for j in range(len(best)):

        sumed=sum(best)
        if sumed!=0:


            best[j]=best[j]/sumed


    return LL, best_ll, best, model_dict

'''changepoint data'''

changepoints=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426, 432, 438, 444, 450, 456, 462, 468, 474, 480, 486, 492, 498]


new_pos=[[1, 2, 3], [1], [1], [3], [2], [1], [1, 2, 3], [2], [2], [1], [1], [1], [1, 2, 3], [2], [1], [3], [3], [3], [1, 2, 3], [1], [1], [1], [1], [1], [1, 2, 3], [2], [1], [3], [3], [1], [1, 2, 3], [3], [2], [3], [2], [1], [1, 2, 3], [3], [2], [1], [3], [2], [1, 2, 3], [1], [2], [3], [1], [1], [1, 2, 3], [3], [1], [2], [1], [2], [1, 2, 3], [1], [2], [1], [1], [3], [1, 2, 3], [1], [1], [1], [1], [3], [1, 2, 3], [1], [1], [1], [2], [1], [1, 2, 3], [3], [3], [2], [3], [1], [1, 2, 3], [3], [3], [2], [2], [2]]



new_probs=[[0.3, 0.7, 0.9], [0.1], [0.5], [0.3], [0.1], [0.7], [0.9, 0.7, 0.5], [0.1], [0.7], [0.3], [0.1], [0.3], [0.9, 0.3, 0.3], [0.1], [0.5], [0.5], [0.9], [0.7], [0.3, 0.7, 0.1], [0.5], [0.7], [0.5], [0.3], [0.5], [0.7, 0.9, 0.3], [0.5], [0.9], [0.1], [0.3], [0.1], [0.3, 0.3, 0.7], [0.5], [0.1], [0.7], [0.9], [0.1], [0.7, 0.5, 0.7], [0.3], [0.9], [0.1], [0.5], [0.7], [0.7, 0.7, 0.3], [0.9], [0.1], [0.5], [0.3], [0.1], [0.9, 0.5, 0.1], [0.3], [0.7], [0.1], [0.5], [0.9], [0.5, 0.5, 0.3], [0.7], [0.1], [0.3], [0.9], [0.7], [0.5, 0.7, 0.7], [0.9], [0.5], [0.9], [0.5], [0.9], [0.3, 0.3, 0.7], [0.9], [0.5], [0.9], [0.5], [0.1], [0.7, 0.7, 0.7], [0.5], [0.3], [0.5], [0.9], [0.3], [0.5, 0.3, 0.5], [0.7], [0.9], [0.1], [0.7], [0.1]]



'''simulate data from each model and then fit all models to it'''


CM=np.zeros((4,4))

T=180
a_min=0.2
a_max=1

alpha_pos_val=[]
alpha_neg_val=[]


for count in range(100):

    #RW
    '''
    alpha = random.uniform(a_min, a_max)
    beta = np.random.exponential(1)+1'''

    alpha_X = get_truncated_normal(mean=0.404, sd=0.2739901864, low=0, upp=1)
    alpha = alpha_X.rvs()

    beta_X = get_truncated_normal(mean=6.261, sd=3.91077537, low=1, upp=100)
    beta = beta_X.rvs()


    #beta=random.uniform(2,12)
    RW = Rescorla_Wagner_model()
    #a,r=RW.simulate_change(T,[0.5,0.5,0.5],alpha,beta,changepoints,new_pos,new_probs)
    a, r = RW.simulate(T, [0.5, 0.5, 0.5], alpha, beta)

    #BIC, best_bic, best, model_dict=fit_all(a,r,changepoints,new_pos)
    AIC, best_bic, best, model_dict = fit_all_AIC(a, r, changepoints, new_pos)
    #LL, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)

    for el_ind in range(len(CM[0])): #update the CM matrix for the given model

        CM[0][el_ind]+=best[el_ind]

    #RW_CK

    #set simulation parameters

    #alpha = random.uniform(a_min, a_max)
    #beta=random.uniform(2,12)

    #beta=np.random.exponential(1) + 1
    #alpha_c = random.uniform(a_min, a_max)
    #beta_c=random.uniform(2,12)
    #beta_c=np.random.exponential(1)+1

    alpha_X = get_truncated_normal(mean=0.5664515917, sd=0.3169843578, low=0, upp=1)
    alpha = alpha_X.rvs()

    beta_X = get_truncated_normal(mean=5.174611978, sd=2.975011374, low=1, upp=100)
    beta = beta_X.rvs()

    alpha_X_c = get_truncated_normal(mean=0.3128277583, sd=0.1547298159, low=0, upp=1)
    alpha_c = alpha_X_c.rvs()

    beta_X_c = get_truncated_normal(mean=3.279238993, sd=2.056496848, low=1, upp=100)
    beta_c = beta_X_c.rvs()

    RW_CK = Rescorla_Wagner_CK_model()
    a, r, acc_best, acc_mid, acc_worst = RW_CK.simulate_change_accuracy(T, [0.5, 0.5, 0.5], alpha, beta, alpha_c,beta_c, changepoints, new_pos, new_probs)
    #a, r = RW_CK.simulate(T, [0.5, 0.5, 0.5], alpha, beta, alpha_c,beta_c)

    #BIC, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)
    AIC, best_bic, best, model_dict = fit_all_AIC(a, r, changepoints, new_pos)
    #LL, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)

    for el_ind in range(len(CM[1])):  # update the CM matrix for the given model

        CM[1][el_ind] += best[el_ind]

    # RW_sep_lr

    #alpha = random.uniform(a_min, a_max)
    #alpha_neg = random.uniform(0.4, a_max)
    #alpha_neg=alpha_neg-0.4
    #beta = np.random.exponential(1)+1
    #beta=random.uniform(2,12)



    init_guess_1_X = get_truncated_normal(mean=0.7018621375, sd=0.2555028194, low=0, upp=1)
    alpha = init_guess_1_X.rvs()

    init_guess_2_X = get_truncated_normal(mean=0.3083247317, sd=0.1569015222, low=0, upp=1)
    alpha_neg= init_guess_2_X.rvs()

    init_guess_3_X = get_truncated_normal(mean=5.766115643, sd=2.218477811, low=1, upp=100)
    beta = init_guess_3_X.rvs()

    alpha_pos_val.append(alpha)
    alpha_neg_val.append(alpha_neg)



    RW_sep_lr = Rescorla_Wagner_sep_lr_model()
    a, r = RW_sep_lr.simulate_change(T, [0.5, 0.5, 0.5], alpha, alpha_neg, beta, changepoints, new_pos, new_probs)
    #a, r = RW_sep_lr.simulate(T, [0.5, 0.5, 0.5], alpha, alpha_neg, beta)

    #BIC, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)
    AIC, best_bic, best, model_dict = fit_all_AIC(a, r, changepoints, new_pos)
    #LL, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)

    for el_ind in range(len(CM[2])):  # update the CM matrix for the given model

        CM[2][el_ind] += best[el_ind]

    # RW_spe_temp
    '''
    alpha = random.uniform(a_min, a_max)
    beta = np.random.exponential(1)+1
    beta_new = np.random.exponential(1)+1


    RW_sep_temp = Rescorla_Wagner_sep_temp_model()
    a, r = RW_sep_temp.simulate_change(T, [0.5, 0.5, 0.5], alpha, beta, beta_new, changepoints, new_pos, new_probs)
    #a, r = RW_sep_temp.simulate(180, [0.5, 0.5, 0.5], alpha, beta, beta_new, changepoints, new_pos, new_probs)

    BIC, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)
    #AIC, best_bic, best, model_dict = fit_all_AIC(a, r, changepoints, new_pos)
    # LL, best_bic, best, model_dict = fit_all(a, r, changepoints, new_pos)

    for el_ind in range(len(CM[3])):  # update the CM matrix for the given model

        CM[3][el_ind] += best[el_ind]'''

print(CM)


plt.scatter(alpha_pos_val,alpha_neg_val)
plt.title('learnign rates used for simulation')
plt.xlabel('pos lr')
plt.ylabel('neg lr')
plt.show()




'''changing to inversion matrix'''
IM=np.zeros((3,3))

for m in range(len(CM[0])-1):

    sum=CM[0][m]+CM[1][m]+CM[2][m]

    IM[0][m]=CM[0][m]/sum
    IM[1][m] = CM[1][m] / sum
    IM[2][m] = CM[2][m] / sum

print(IM)




















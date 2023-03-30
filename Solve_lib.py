import numpy as np


def EM(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_0, maxCount, minLike):
    I = np.eye(3)

    vals = []

    alpha = alpha_0
    count = 1
    maxlike = -np.inf
    Log_like_nm1 = -999
    Sigma_F_m1 = np.linalg.inv(Sigma_F)
    Sigma_B_m1 = np.linalg.inv(Sigma_B)
    while(1):

        Theta = np.zeros((6, 6))
        Theta[:3, :3] = Sigma_F_m1 + I * alpha * alpha /((Sigma_C)**2)
        Theta[:3,3:] = Theta[3:,:3] = I * alpha*(1-alpha)/((Sigma_C)**2)
        Theta[3:,3:] = Sigma_B_m1 + I * (1 - alpha)**2/((Sigma_C)**2)

        #Theta = np.array([[Sigma_F_m1 + I * alpha * alpha /((Sigma_C)**2), I * alpha*(1-alpha)/((Sigma_C)**2)],
                #[I * alpha * (1 - alpha)/((Sigma_C)**2), Sigma_B_m1 + I * (1 - alpha)**2/((Sigma_C)**2)]])
        Phi  = np.zeros((6, 1))
        Phi[:3] = np.reshape(Sigma_F_m1 @ mu_F + C*(alpha) / ((Sigma_C)**2),(3,1))
        Phi[3:] = np.reshape(Sigma_B_m1 @ mu_B + C*(1-alpha) / ((Sigma_C)**2),(3,1))
        #Phi = np.array([[Sigma_F_m1 * mu_F + C * alpha /((Sigma_C)**2)], [Sigma_B_m1 * mu_B + C * (1 - alpha)/((Sigma_C)**2)]])

        FB = np.linalg.solve(Theta, Phi)

        F = np.maximum(np.minimum(FB[0:3], 1), 0)
        B = np.maximum(np.minimum(FB[3:6], 1), 0)

        alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]

        F = F.T
        B = B.T

        like_C = - np.sum((np.atleast_2d(C).T -alpha*F-(1-alpha)*B)**2) /((Sigma_C)**2)
        like_fg = (- ((F- np.atleast_2d(mu_F).T).T @ Sigma_F_m1 @ (F-np.atleast_2d(mu_F).T))/2)[0,0]
        like_bg = (- ((B- np.atleast_2d(mu_B).T).T @ Sigma_B_m1 @ (B-np.atleast_2d(mu_B).T))/2)[0,0]
        like = (like_C + like_fg + like_bg)

        if like > maxlike:
            a_best = alpha
            maxLike = like
            fg_best = F.ravel()
            bg_best = B.ravel()

        if count >= maxCount or abs(like-Log_like_nm1) <= minLike:
            break

        Log_like_nm1 = like
        count = count + 1

    return F, B, alpha, like

def Calc(mu_F,Sigma_F,mu_B,Sigma_B,C,sigma_C,alpha_0, maxCount, minLike):

    vals = []

    for i in range(mu_F.shape[0]):
        mu_Fi = mu_F[i]
        Sigma_Fi = Sigma_F[i]

        for j in range(mu_B.shape[0]):
            mu_Bj = mu_B[j]
            Sigma_Bj = Sigma_B[j]

            F, B, alpha, like = EM(mu_Fi,Sigma_Fi,mu_Bj,Sigma_Bj,C,sigma_C,alpha_0, maxCount, minLike)
            val = {'F': F, 'B': B, 'alpha': alpha, 'like': like}
            vals.append(val)

        max_like, max_index = max((v['like'], i) for i, v in enumerate(vals))
        F = vals[max_index]['F']
        B = vals[max_index]['B']
        alpha = vals[max_index]['alpha']
        ################
        ################
    return F, B, alpha

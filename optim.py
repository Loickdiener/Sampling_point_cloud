import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.random as rng
import outils
from geomloss import SamplesLoss
import random
import time
import affichage
import os

sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur= 0.15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def variable_changeante(it, Nmin, Nmax, peri_p, peri_w, mu0, amplificateur):
    turn_p = (((it % 5 != 0 and it < 300) or (it % 2 == 0 and it>300 and it<10000) or (it % 100 > 4 and it>10000)) or (Nmin==Nmax))
    if Nmin !=Nmax:
        if it>300 and peri_p ==0 and it % 200 == 0:
            peri_w = 10
        if it>300 and peri_w ==0 and it % 200 == 100:
            peri_p = 10
        
        if peri_w>0:
            turn_p = False
            peri_w -=1
        if peri_p>0:
            peri_p -=1
            turn_p = True
    
    if it % 50 == 0:
        amplificateur = min(1 + it/50 ,Nmax * 10)
        
    mu = max(1e-5, mu0 * (0.99 ** it))
    return turn_p, peri_p, peri_w, mu, amplificateur

def calc_point_poid_nb_inter(nech, Nmax, d, P_list, w_list, amplificateur, turn_p):
    P_masked = torch.empty((nech, Nmax, d), device=P_list[0].device)
    m_masked = torch.empty((nech, Nmax), device=P_list[0].device)
    nb_points = torch.empty(nech, device=P_list[0].device)

    # Préparation des nuages
    for i, (P, w) in enumerate(zip(P_list, w_list)):
        Pi, mi = outils.get_masked_points(P, w * amplificateur)
        nb_points[i] = (mi.sum())
        mi = mi/(mi.sum())  
        if turn_p:
            P_masked[i] = Pi
            m_masked[i] = mi.detach()
        else:
            P_masked[i] = Pi.detach()
            m_masked[i] = mi
    return P_masked, m_masked, nb_points


def dist_min_par_ech(nech, P_list,P_masked):
    min_intra_dist = torch.empty(nech, device=P_list[0].device)
    
    for i in range(nech):
        dist_pour_intra = torch.cdist(P_masked[i], P_masked[i])
        k, j = torch.triu_indices(dist_pour_intra.size()[0], dist_pour_intra.size()[0], offset=1)
        intra_all_dist = dist_pour_intra[k,j]
        min_intra_dist[i] = outils.quasimin(intra_all_dist, 0.001)
    return min_intra_dist

ofr_comp = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur= 0.1,
    backend="tensorized"

)
def calc_min_dist_inter_nuage(nech, P_masked, m_masked, Nmin, Nmax, jln_mth, echdist = []):
    i, j = torch.triu_indices(nech, nech, offset=1)
    
    P_i = P_masked[i]        # (num_pairs, M, d)
    P_j = P_masked[j]
    
    w_i = m_masked[i]         # (num_pairs, M)
    w_j = m_masked[j]
    # calcul Sinkhorn pairwise (batché)
    if Nmin != Nmax:
        all_dist = sinkhorn(w_i, P_i, w_j, P_j)
    else:
        all_dist = sinkhorn(P_i, P_j)
    
    if jln_mth:
        return ofr_comp(all_dist.view(-1,1), echdist.view(-1,1))
    else:
        return torch.min(all_dist)





def calc_loss_p(a,b,d,P_list, pds0, loss_dist, pena_rep, nech, inert_pena, mu0):
    regp = torch.mean(torch.stack([outils.borne(P ,a ,b , d) for P in P_list]))
    return (- pds0 * loss_dist + nech/5 * inert_pena + mu0 * regp + pena_rep)


def calc_loss_w(Nmin, Nmax, nb_points, K, P_list, nbp_possible, temp,
                w_list, nech, ct, it, alpha, beta, gamma, delta, loss_dist, mu
                , pds0, pds1, pds2, pds3):
    ct +=1
    regw = torch.mean(torch.stack([outils.bornew((P-Nmin)/(Nmax-Nmin)) for P in nb_points]))
    # Régularisation taille
    if torch.isnan(regw):
        penalty_size = 0
        for nb in nb_points:
            penalty_size += torch.relu(Nmin - nb)  
    effect = torch.zeros(K, device=P_list[0].device)
    for i in range(K):
        effect[i] = outils.soft_count_near(nb_points,nbp_possible[i],temp) 
        
    dif_att_rll = torch.stack([( (effect.sum()/K) - efi  )**2 for efi in effect])
    pena = torch.mean(dif_att_rll)
    
    repul_z = 0
    for w in w_list:
        repul_z += outils.w_penal(w)
    repul_z = repul_z/nech
    
    if ct == 1 or ct% 50 == 0 and it<2000:
         with torch.no_grad():
             pds1 = torch.min((beta/alpha) * torch.abs(loss_dist)/pena, torch.tensor(.1)) * pds0
             pds2 = torch.min((gamma/alpha) * torch.abs(loss_dist)/regw,  torch.tensor(mu)) * pds0
             pds3 = torch.min((delta/alpha) * torch.abs(loss_dist)/repul_z, torch.tensor(1e-5)) * pds0
    
    if torch.isnan(pds2) and not torch.isnan(regw):
        with torch.no_grad():
            pds2 = torch.min((gamma/alpha) * torch.abs(loss_dist)/regw,  torch.tensor(mu)) * pds0
            print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")

    if torch.isnan(regw):
        return (- pds0 * loss_dist + pds0 * 1000 * penalty_size + pds1 * pena + pds3 * repul_z ), ct, pds1, pds2, pds3
    else:   
        return (- pds0 * loss_dist + pds1 * pena +  max((20/(it+1)), 1e-2)* pds2 * regw + pds3 * repul_z), ct, pds1, pds2, pds3


#echdist a absolument fournir si 
def optim_boucl(P_list, w_list, Nmin, Nmax, nech, echdist = [], a = torch.tensor([0,0]), b = torch.tensor([1,1]), 
                d = 2, mu0 = 7e-4, born_disper_inf = 0, temp = 2.3, plot_hist = False, inert_pena_ch = True, 
                jln_mth = False, tol = 1e-6, repul_param = 0):
    
    if jln_mth and len(echdist) == 0:
        raise ValueError("echdist doit etre ranseigner et non vide pour cette methode(jln_mth)")
    #initialisation de toutes les variable necessaire au bon fonctionnement de la boucle d'optimisation 
    #ainsi que celle permettant de recuperer les historique de certaine de nos variable
    it = 0
    peri_p = 0
    peri_w = 0
    alpha = 0.70
    beta = 0.20
    gamma = 0.03
    delta =0.07
    pds0 = min(np.log(nech+1e-8), 10)
    prev_loss = 1000
    prev_ploss = 1000
    prev_wloss = 1000
    prev_prev_wloss = 10000
    prev_prev_ploss = 10000
    prev_loss_dist = 10000
    hst_lossw = []
    amplificateur = 1
    cntlds = 0
    hist_loss_cl = []
    ct = 0
    K = Nmax - Nmin + 1
    cnt = 0
    pds1 = pds2 = pds3 = 1
    nbp_possible = torch.linspace(Nmin, Nmax, Nmax - Nmin +1, device=device)
    
    optimizer_P = torch.optim.AdamW(
        [P for P in P_list],
        lr=1e-3)
    
    if Nmin != Nmax:
        optimizer_w = torch.optim.AdamW(
            [w for w in w_list],
            lr=1e-3)
    
    if len(a)== d and len(b) == d:
        born_disper = 1.2 * Nmax**(-1/d) *( torch.prod(b-a))**(1/d)
    else: 
        born_disper = 1.2 * Nmax**(-1/d)
    
    while it<30000 :
        turn_p, peri_p, peri_w, mu, amplificateur = variable_changeante(it, Nmin, Nmax, peri_p, peri_w, mu0, amplificateur)
        
        if turn_p:
            optimizer = optimizer_P
        else:
            optimizer = optimizer_w
        

        optimizer.zero_grad()

        P_masked, m_masked, nb_points = calc_point_poid_nb_inter(nech, Nmax, d,
                                        P_list, w_list, amplificateur, turn_p)
        
        loss_dist = calc_min_dist_inter_nuage(nech, P_masked, m_masked, Nmin, Nmax, jln_mth)
        
        min_intra_dist = dist_min_par_ech(nech, P_list,P_masked)
        
        if inert_pena_ch:
            inert_pena, born_disper_inf, born_disper = outils.cvm_uniform_loss(min_intra_dist,a = born_disper_inf, b =born_disper)
        else:
            inert_pena = 0
        
        if turn_p:
            if repul_param > 1e-5:
                pena_rep = torch.stack([outils.repulsion_penalty3(P, repul_param) for P in P_list]).mean()
            else:
                pena_rep = 0
            loss = calc_loss_p(a,b,d,P_list, pds0, loss_dist, pena_rep, nech, inert_pena, mu0)
            if np.abs(prev_ploss - loss.item())<1e-5:
                peri_w = 10
                
            hist_loss_cl.append(loss.item())
        else:
            loss, ct, pds1, pds2, pds3 = calc_loss_w(Nmin, Nmax, nb_points, K, P_list, nbp_possible, temp,
                            w_list, nech, ct, it, alpha, beta, gamma, delta, loss_dist, mu
                            , pds0, pds1, pds2, pds3)

            if np.abs(prev_wloss - loss.item())<1e-5:
                peri_p = 10
         

        if it > 800 :
            if turn_p:
                if torch.abs(loss - prev_ploss) + np.abs(prev_wloss - prev_prev_wloss)< tol:
                    break
            else :
                if torch.abs(loss - prev_wloss) + np.abs(prev_ploss - prev_prev_ploss)< tol:
                    break
                
            if torch.abs(loss_dist - prev_loss_dist)< tol/2 and it > 1500:
                cntlds +=1
                if cntlds >7:
                    break
            else:
                cntlds = 0
        
        if peri_w >0 and torch.abs(loss - prev_wloss)<tol*10:
            peri_w = 0
            
        if peri_p >0 and torch.abs(loss - prev_ploss)<tol*10:
            peri_p = 0
        
        loss.backward()

        if turn_p: 
            torch.nn.utils.clip_grad_norm_(
                [p for p in (P_list)],
                torch.min(b))
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in (w_list)],
                1.)
            
        optimizer.step()

        with torch.no_grad():
            for P in P_list:
                P.clamp_(a + 1e-6, b - 1e-6)
            
          
        if cnt>49:
            print(f"it {it} | loss {prev_ploss + prev_wloss:.6f} | loss w {prev_wloss:.6f} | loss cl {prev_ploss:.6f}")
            print(P_list[0].device)
            print(torch.cuda.memory_allocated() / 1e6, "MB")
            cnt = 0
        cnt += 1
        if turn_p:  
            prev_prev_ploss = prev_ploss
            prev_loss = loss.item() + prev_wloss + 1 * peri_p
            prev_ploss = loss.item()
        else:
            prev_prev_wloss = prev_wloss
            prev_loss += prev_wloss 
            prev_wloss = loss.item()
            
            hst_lossw.append(prev_wloss)
        prev_loss_dist = loss_dist
        it +=1
    
    if not jln_mth:
        sorted_intra_min, indice = torch.sort(min_intra_dist)
        indice = indice.cpu()
    
        P_list = [P_list[i] for i in indice]
        w_list = [w_list[i] for i in indice]
        
    if plot_hist:
        plt.hist(min_intra_dist.detach().cpu().numpy())
        plt.show()
        
        plt.plot(hst_lossw[500:])
        plt.show()
        
        plt.plot(hist_loss_cl[500:])
        plt.show()
    return P_list, w_list
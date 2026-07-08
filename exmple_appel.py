import numpy as np
import torch
import affichage
import initialisation
import optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mise_enforme_borne(a, b):
    temp = b - a
    retour_param = torch.min(temp)
    if retour_param<0:
        raise ValueError("b doit etre la borne sup et ne pas etre egal a zeros")
    return torch.zeros(len(a)), temp/retour_param, retour_param
    
def remise_a_niveau(a, b, a_reel, retour_param, nech, P_list):
    a = a_reel
    for i in range(nech):
        P_list[i] = P_list[i]*retour_param + a
    b = b*retour_param
    b += a
    return a,b, P_list
    

#comportement pouvant etre etrange si une des dimention est beaucoup plus petites que les autre de l'ordre de 1 pour 100 voir 1 pour 1000
def creation_dun_ech(Nmin, Nmax, nech,jln_mth = False, a = torch.tensor([0,0]), b = torch.tensor([1,1]), d= 2,nrechlhs = 60000, 
                     mu0 = 7e-4, born_disper_inf = 0, temp = 2.3, plot_hist = False, inert_pena_ch = True, tol = 1e-6, repul_param = 0,
                     all_opt = True, aff = True, for_torch = True, seed = None, veux_coin = False, lhs = False, aff_repart = True,
                     dossier = "resultat_optim", save = False,  export_all = False, aff_fin_nage = True, aff_sup_nuage = True):
    
    a_reel = a.clone()
    a, b, retour_param = mise_enforme_borne(a, b)
    P_list, echdist = initialisation.initialisation(Nmin, Nmax, nech,jln_mth, a, b, d, nrechlhs, all_opt, aff, for_torch, seed, veux_coin, lhs)
    if Nmin != Nmax:
        w_list = [(torch.rand(Nmax, requires_grad=True, device=device)) for _ in range(nech)]

        for w in w_list:
            ind = np.random.choice(np.arange(0, Nmax), size= np.random.randint(0, Nmax - Nmin), replace=False)
        
            with torch.no_grad():
                w[ind] = -w[ind]
    P_list, w_list = optim. optim_boucl(P_list, w_list, Nmin, Nmax, nech, echdist, a, b, 
                    d, mu0, born_disper_inf, temp, plot_hist, inert_pena_ch, 
                    jln_mth, tol, repul_param)
    
    a, b, P_list = remise_a_niveau(a, b, a_reel, retour_param, nech, P_list)
    P_final = affichage.traitement_et_aff(P_list, w_list, nech, Nmin, Nmax, d, a, b, export_all,
                      aff_fin_nage, aff_sup_nuage, save, dossier, aff_repart)
    
    return P_final #si tu veux en faire qqc directement ici
    
    
    
    
    

if __name__ == '__main__':
    start = time.perf_counter()
    torch.set_default_dtype(torch.float32) #a set toujours avant d'appeler la fonction attention beaucoup plus rappide en float32 qu'en float64
    P_list_f = creation_dun_ech(Nmin = 24, Nmax = 30, nech = 100, a = torch.tensor([15,-30]), b = torch.tensor([23,-22]), d = 2,
                     plot_hist = True, inert_pena_ch = True, )
    end = time.perf_counter()
    print(f"Temps d'exécution : {end - start:.6f} secondes")
    
    